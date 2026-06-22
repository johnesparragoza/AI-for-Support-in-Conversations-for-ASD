import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import re
from gtts import gTTS
import io
import os

# fetch the token from environment variables (HF Spaces secrets) or Streamlit Secrets.
# os.getenv is checked first because HF Spaces exposes secrets as env vars, and
# st.secrets raises when no secrets.toml file exists (which is the case on Spaces).
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    try:
        hf_token = st.secrets.get("HF_TOKEN")
    except Exception:
        hf_token = None

# helper: fading function ---
def get_faded_prompt(words, fade_level):
    """Return the narrative with the last `fade_level` words replaced by blanks."""
    if fade_level == 0:
        return " ".join(words)
    faded_words = [
        word if i < len(words) - fade_level else "___"
        for i, word in enumerate(words)
    ]
    return " ".join(faded_words)

# app UI 
# Page configuration (sets favicon, title, and center page)
st.set_page_config(
    page_title="Image-to-Narrative",
    layout="centered"
)

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Chewy&display=swap');
    </style>
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style="font-family: 'Chewy', cursive; color: #2E7D32;">AImage Narrator 🪄</h1>
        <p style='font-size: 1.2rem;'>Welcome! Upload your image and get a friendly, narrated description!</p>
    </div>
""", unsafe_allow_html=True)

# app tutorial — welcoming figure on the left, steps inside a cloud illustration on the right
buddy_img = os.path.join(os.path.dirname(__file__), "puzzle-buddy.png")
intro_left, intro_right = st.columns([1, 2], vertical_alignment="center")
with intro_left:
    st.image(buddy_img, width="stretch")
with intro_right:
    st.markdown("""
    <style>
        .tutorial-sky {
            background: transparent;
            padding: 2.5rem 1rem 3rem 1rem;
            border-radius: 24px;
            margin-bottom: 1.5rem;
        }
        .cloud-card {
            position: relative;
            background: #ffffff;
            border-radius: 90px;
            padding: 2.2rem 2.6rem;
            max-width: 480px;
            margin: 0 auto;
            box-shadow: 0 12px 28px rgba(0,0,0,0.08);
        }
        /* fluffy puffs around the cloud */
        .cloud-card::before {
            content: "";
            position: absolute;
            z-index: -1;
            background: #ffffff;
            border-radius: 50%;
            width: 95px; height: 95px;
            top: -35px; left: 55px;
            box-shadow: 130px -12px 0 12px #fff, 270px 4px 0 -6px #fff;
        }
        .cloud-card::after {
            content: "";
            position: absolute;
            z-index: -1;
            background: #ffffff;
            border-radius: 50%;
            width: 75px; height: 75px;
            bottom: -28px; right: 60px;
            box-shadow: -150px 16px 0 6px #fff, -320px 6px 0 -8px #fff;
        }
        .cloud-card h3 {
            font-family: 'Chewy', cursive;
            color: #2E7D32;
            text-align: center;
            margin: 0 0 1rem 0;
        }
        .cloud-steps {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        .cloud-steps li {
            font-size: 1.1rem;
            line-height: 1.6;
            margin: 0.55rem 0;
            color: #37474F;
        }
        .cloud-steps li span {
            margin-right: 0.5rem;
        }
    </style>
    <div class='tutorial-sky'>
        <div class='cloud-card'>
            <h3>How it works ☁️</h3>
            <ul class='cloud-steps'>
                <li><span></span>- Upload a photo/file or take a picture!</li>
                <li><span></span>- The AI model will create a short narrative.</li>
                <li><span></span>- Practice filling in the blanks as words fade out.</li>
                <li><span></span>- Enjoy!</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# user uploads image 
uploaded_file = st.file_uploader(
    "Choose an image file (jpg, jpeg, png)...", 
    type=["jpg", "jpeg", "png"],
    help="Upload a clear photo (jpg, jpeg, png). Max size: 5MB."
)

# load model and processor (cache to avoid reloading) 
@st.cache_resource
def load_model():
    import transformers
    if not hasattr(transformers.PreTrainedModel, "_supports_sdpa"):
        transformers.PreTrainedModel._supports_sdpa = True

    model_id = "visheratin/MC-LLaVA-3b"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        token=hf_token,
        low_cpu_mem_usage=True,  # keeps peak RAM near final size; matters on the 16GB CPU Space
        attn_implementation="eager"  # <-- Bypasses the internal SDPA check causing the error
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model

# cleaner response code for TTS
def clean_response(text, prompt):
    # Remove prompt and special tokens from the generated output
    cleaned = text.replace(prompt, "")
    cleaned = re.sub(r"<\|.*?\|>", "", cleaned)
    return cleaned.strip()

if uploaded_file is not None: # open and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your uploaded image", width='stretch')
    
    # caption input with tooltip
    caption = st.text_input(
        "Optional: Add your own caption for this image",
        "",
        help="Add a short description, or leave blank for automatic caption."
    )
    if not caption:
        caption = " "  # Avoid empty caption in prompt

    # Streamlit reruns this whole script on every widget interaction — including
    # the Fade buttons and the caption box. Without a guard, model.generate()
    # (60s+ on CPU) would re-run on every click, greying out the page mid-rerun
    # and making fading unusable. So we generate only when the image or caption
    # actually changes, cache the narrative in session_state, and let fade clicks
    # just re-render the cached text instantly.
    file_id = getattr(uploaded_file, "file_id", None) or uploaded_file.name
    narrative_key = (file_id, caption)
    if st.session_state.get("narrative_key") != narrative_key:
        # load the cached model and processor
        processor, model = load_model()

        # build/edit prompt
        prompt = (
        "<|im_start|>user\n"
        "<image>\n"
        "Describe this image with a single first-person sentence.\n"
        "Make it short and simple, no more than 10 words.\n"
        "Use first-person perspective (e.g., 'I see...', 'I feel...').\n"
        f"Caption: {caption}\n"
        "Narrative:\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
            )
        # preparing inputs for the model
        inputs = processor(prompt, [image], model, return_tensors="pt")
        inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        # Generate narrative
        with st.spinner("Generating narrative… (first run also downloads the model, ~1 min)"):
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    # KV caching re-enabled: requirements.txt pins the transformers 4.40
                    # stack, where past_key_values.seen_tokens still exists, so MC-LLaVA's
                    # remote code path works with the cache on. This is the speed win over
                    # the old use_cache=False workaround.
                    use_cache=True,
                    do_sample=False,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
        generated_text = processor.tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract only the generated narrative
        narrative_parts = re.split(r"<\|im_start\|>assistant", generated_text)
        narrative = narrative_parts[-1] if len(narrative_parts) > 1 else generated_text
        narrative = narrative.replace("<|im_end|>", "").strip()

        st.session_state.narrative = narrative
        st.session_state.narrative_key = narrative_key
        st.session_state.fade_level = 0  # reset fading whenever a new narrative is made

    # use the cached narrative; fade clicks reach here without regenerating
    narrative = st.session_state.narrative
    words = narrative.split()
    
    # state for Fade Level
    if "fade_level" not in st.session_state:
        st.session_state.fade_level = 0
    max_fade = len(words)

    # buttons for fading
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        if st.button("Fade Less", help="Show more of the prompt (remove one blank)"):
            if st.session_state.fade_level > 0:
                st.session_state.fade_level -= 1
    with col3:
        if st.button("Fade More", help="Fade one more word from the end"):
            if st.session_state.fade_level < max_fade:
                st.session_state.fade_level += 1

    faded_prompt = get_faded_prompt(words, st.session_state.fade_level)
    st.markdown("**Your Practice Prompt:**")
    st.success(faded_prompt)
    
    # TTS Button 
    if st.button("🔊 Read Aloud"):
        tts = gTTS(narrative)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        st.audio(mp3_fp, format="audio/mp3")

    # optional, let a supporter/teacher reveal the full narrative if needed
    with st.expander("Show full narrative (for supporter/teacher use)"):
        st.info(narrative)
else:
    st.info("Upload a picture you want to talk about! (jpg, jpeg, or png).")

st.markdown("---")
st.caption("Powered by MC-LLaVA-3b and Streamlit. For best results, use clear photos with obvious subjects!")
