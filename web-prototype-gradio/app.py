import base64
import os
import re
import sys
import tempfile
import time

# Force line-buffered, flushed stdout/stderr so HF run logs show our boot
# progress in real time even if the process later hangs or is killed during
# startup (Python block-buffers stdout when not attached to a TTY).
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


def _boot(msg):
    # Write to stderr: HF Spaces run logs surface stderr reliably, whereas
    # stdout can be swallowed by block buffering.
    print(f"[BOOT] {msg}", file=sys.stderr, flush=True)


_boot("starting imports")

import gradio as gr
import spaces  # HF ZeroGPU; this import is a no-op outside ZeroGPU environments
import torch

# --- ZeroGPU pins gradio==4.44.0 (its build recipe overrides README sdk_version),
# and 4.44.0's bundled gradio_client 1.3.0 crashes get_api_info() with
#   "TypeError: argument of type 'bool' is not iterable"
# on gr.Image()'s boolean schema (additionalProperties: true|false). gradio #11084.
# Short-circuit the schema->type converters on bool schemas. (The companion
# starlette TemplateResponse incompatibility is fixed by pinning starlette in
# requirements.txt, not here.)
import gradio_client.utils as _gc_utils

_orig_json_to_py = _gc_utils._json_schema_to_python_type
_orig_get_type = _gc_utils.get_type


def _safe_json_to_py(schema, defs=None):
    if isinstance(schema, bool):
        return "bool"
    return _orig_json_to_py(schema, defs)


def _safe_get_type(schema):
    if isinstance(schema, bool):
        return "bool"
    return _orig_get_type(schema)


_gc_utils._json_schema_to_python_type = _safe_json_to_py
_gc_utils.get_type = _safe_get_type
_boot("gradio_client patched")

from gtts import gTTS
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

_boot("imports complete")

# HF Spaces exposes secrets as env vars. The token is optional (the model is
# public) but we pass it through if present to avoid any rate limiting.
hf_token = os.getenv("HF_TOKEN")

MODEL_ID = "visheratin/MC-LLaVA-3b"


# --- Model loaded at MODULE level on cuda ---------------------------------
# ZeroGPU requires this: outside @spaces.GPU functions a CUDA *emulation* mode
# is active, so .to("cuda") works at import time, and the real GPU is attached
# only while a @spaces.GPU function runs. Lazy-loading inside the function is
# explicitly discouraged by the ZeroGPU docs (CUDA transfers are optimized for
# startup placement).
import transformers
if not hasattr(transformers.PreTrainedModel, "_supports_sdpa"):
    transformers.PreTrainedModel._supports_sdpa = True

_boot("loading processor + model...")
try:
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True, token=hf_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,  # real GPU path — fp16 is ~6.5GB and fast
        trust_remote_code=True,
        token=hf_token,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to("cuda")
    _boot("model loaded OK")
except Exception:
    import traceback
    _boot("MODEL LOAD FAILED:")
    traceback.print_exc()
    sys.stderr.flush()
    raise


# --- server-side generation timing ----------------------------------------
# Logs to the HF Space "Logs" tab only — invisible to end users. Writes to
# STDERR because HF run logs reliably surface stderr; stdout is block-buffered
# and can be swallowed. Format is identical to the Streamlit app so the two
# environments can be compared apples-to-apples (same model/prompt/
# max_new_tokens; only CPU+fp32 vs ZeroGPU+fp16 differ).
#
# IMPORTANT (ZeroGPU): this is called from on_generate (the MAIN gradio process),
# NOT from inside the @spaces.GPU function. ZeroGPU forks the GPU call into a
# subprocess whose stderr does not reliably reach the Space run logs, so the
# timing is measured inside the GPU function and returned, then logged here.
# Delete this and its call site to strip instrumentation.
def log_generation_timing(env, device, dtype, input_len, new_tokens, elapsed):
    tok_per_s = new_tokens / elapsed if elapsed > 0 else float("nan")
    print(
        f"[TIMING] env={env} device={device} dtype={dtype} "
        f"input_tokens={input_len} new_tokens={new_tokens} "
        f"elapsed={elapsed:.2f}s tok_per_s={tok_per_s:.2f}",
        file=sys.stderr,
        flush=True,
    )


def _mascot_data_uri():
    """Return puzzle-buddy.png as a base64 data URI, or '' if it's missing.

    Embedding the mascot inline (rather than via gr.Image) lets us style it
    freely inside the welcome HTML and avoids Gradio's image-component chrome
    (upload/download buttons, borders). Returns '' so the template can simply
    omit the <img> if the asset isn't present.
    """
    path = os.path.join(os.path.dirname(__file__), "puzzle-buddy.png")
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return ""


def get_faded_prompt(words, fade_level):
    """Return the narrative with the last `fade_level` words replaced by blanks."""
    if fade_level <= 0:
        return " ".join(words)
    return " ".join(
        word if i < len(words) - fade_level else "___"
        for i, word in enumerate(words)
    )


# --- The one GPU-bound call -----------------------------------------------
# duration is the max GPU seconds ZeroGPU reserves; a short, honest estimate
# keeps queue priority high for other visitors. 64 new tokens on this hardware
# is a few seconds, so 60s is generous headroom.
@spaces.GPU(duration=60)
def _generate_narrative(image, caption):
    prompt = (
         "<|im_start|>user\n"
        "<image>\n"
        "Describe this image in one short first-person sentence, as if YOU are "
        "the person in the picture doing the action.\n"
        "Examples: 'I am riding my bike.' 'I am baking cookies.' 'I am playing in the park.'\n"
        "Keep it under 10 words.\n"
        "Caption: {caption}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "I"
    )
    inputs = processor(prompt, [image], model, return_tensors="pt")
    inputs = {
        k: v.to(model.device) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }
    input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else None
    with torch.no_grad():
        # CUDA kernels are async; sync before/after so elapsed reflects real GPU
        # completion, not just the launch-return time.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = model.generate(
            **inputs,
            max_new_tokens=64,
            use_cache=True,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    new_tokens = len(output[0]) - input_len if input_len is not None else len(output[0])
    generated_text = processor.tokenizer.decode(output[0], skip_special_tokens=True)
    parts = re.split(r"<\|im_start\|>assistant", generated_text)
    narrative = parts[-1] if len(parts) > 1 else generated_text
    narrative = narrative.replace("<|im_end|>", "").strip()
    # Return timing metrics so the MAIN process (on_generate) can log them; see
    # log_generation_timing for why we can't log from inside this GPU subprocess.
    timing = {
        "device": str(model.device),
        "dtype": str(model.dtype),
        "input_len": input_len,
        "new_tokens": new_tokens,
        "elapsed": elapsed,
    }
    return narrative, timing


# --- Gradio event handlers (run on CPU; only _generate_narrative hits GPU) -
def on_generate(image, caption):
    if image is None:
        raise gr.Error("Please upload an image first.")
    narrative, t = _generate_narrative(image.convert("RGB"), caption or " ")
    log_generation_timing(
        "gradio-zerogpu", t["device"], t["dtype"],
        t["input_len"], t["new_tokens"], t["elapsed"],
    )
    words = narrative.split()
    return (
        narrative,                                   # state: full narrative
        gr.update(value=narrative, visible=True),    # practice prompt box
        # fade slider: reset to 0, max = word count
        gr.update(minimum=0, maximum=len(words), value=0, visible=True),
        gr.update(value=narrative, visible=True),    # supporter accordion text
    )


def on_fade(narrative, fade_level):
    return get_faded_prompt(narrative.split(), int(fade_level))


def on_read_aloud(narrative):
    if not narrative:
        raise gr.Error("Generate a narrative first.")
    tts = gTTS(narrative)
    fp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tts.write_to_fp(fp)
    fp.close()
    return fp.name


# --- Friendly, ASD-accessible theme & styling -----------------------------
# Mirrors the welcoming look of the Streamlit web-prototype: a calming green
# palette, the rounded "Chewy" display font for the title, the puzzle-buddy
# mascot, and a fluffy cloud "How it works" card. Soft base theme gives us the
# rounded corners / gentle shadows; the rest is layered on via custom CSS so it
# reads the same in Gradio 4.44.0 as it does in Streamlit.
THEME = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="green",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Quicksand"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#ffffff",
    button_large_radius="18px",
    button_small_radius="14px",
    block_radius="20px",
    block_shadow="0 8px 20px rgba(0,0,0,0.06)",
)

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Chewy&display=swap');

.gradio-container { max-width: 920px !important; margin: 0 auto !important; }

/* Welcome header */
#app-header { text-align: center; padding: 1.6rem 0 0.4rem 0; }
#app-header h1 {
    font-family: 'Chewy', cursive;
    color: #2E7D32;
    font-size: 2.9rem;
    margin: 0;
    line-height: 1.1;
}
#app-header p { font-size: 1.25rem; color: #37474F; margin: 0.5rem 0 0 0; }

/* Intro: mascot + cloud "how it works" card */
#intro-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1.5rem;
    flex-wrap: wrap;
    padding: 1.2rem 0 0.5rem 0;
}
#intro-row .mascot { width: 160px; max-width: 30vw; height: auto; }
.cloud-card {
    position: relative;
    background: #ffffff;
    border-radius: 90px;
    padding: 2rem 2.6rem;
    max-width: 460px;
    box-shadow: 0 12px 28px rgba(0,0,0,0.08);
}
.cloud-card::before {
    content: ""; position: absolute; z-index: -1; background: #ffffff;
    border-radius: 50%; width: 95px; height: 95px; top: -35px; left: 55px;
    box-shadow: 130px -12px 0 12px #fff, 270px 4px 0 -6px #fff;
}
.cloud-card::after {
    content: ""; position: absolute; z-index: -1; background: #ffffff;
    border-radius: 50%; width: 75px; height: 75px; bottom: -28px; right: 60px;
    box-shadow: -150px 16px 0 6px #fff, -320px 6px 0 -8px #fff;
}
.cloud-card h3 {
    font-family: 'Chewy', cursive; color: #2E7D32;
    text-align: center; margin: 0 0 0.8rem 0; font-size: 1.4rem;
}
.cloud-card ul { list-style: none; padding: 0; margin: 0; }
.cloud-card li {
    font-size: 1.08rem; line-height: 1.6; margin: 0.5rem 0; color: #37474F;
}

/* Make the practice-prompt box read like the calm green "success" panel
   the Streamlit app uses, so the part learners focus on stands out gently. */
#practice-box textarea {
    background: #e8f5e9 !important;
    color: #1b5e20 !important;
    font-size: 1.3rem !important;
    line-height: 1.7 !important;
    border-radius: 16px !important;
    text-align: center;
}
#footer-note { text-align: center; color: #607D8B; padding-top: 0.5rem; }

/* Safety net: keep text readable (dark) even if a dark-mode preference slips
   through before the force-light redirect applies. */
.gradio-container textarea,
.gradio-container input[type="text"] { color: #263238 !important; }
"""

_mascot_uri = _mascot_data_uri()
_mascot_html = (
    f"<img class='mascot' src='{_mascot_uri}' alt='Puzzle Buddy mascot'/>"
    if _mascot_uri
    else ""
)

# Pin the app to light mode. The Soft theme keeps dark-mode's white text even
# though we force a white background, which makes text invisible for viewers
# whose browser/OS prefers dark mode. Forcing ?__theme=light keeps text dark.
_FORCE_LIGHT_JS = """
() => {
    const url = new URL(window.location.href);
    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.replace(url.href);
    }
}
"""

with gr.Blocks(
    title="AImage Narrator", theme=THEME, css=CUSTOM_CSS, js=_FORCE_LIGHT_JS
) as demo:
    gr.HTML(
        "<div id='app-header'>"
        "<h1>AImage Narrator 🪄</h1>"
        "<p>Welcome! Upload your image and get a friendly, narrated description.</p>"
        "</div>"
        "<div id='intro-row'>"
        f"{_mascot_html}"
        "<div class='cloud-card'>"
        "<h3>How it works ☁️</h3>"
        "<ul>"
        "<li>📷 Upload a photo or take a picture.</li>"
        "<li>✨ The AI writes a short, friendly narrative.</li>"
        "<li>🧩 Practice filling in the blanks as words fade out.</li>"
        "<li>🎉 Enjoy!</li>"
        "</ul>"
        "</div>"
        "</div>"
    )

    narrative_state = gr.State("")

    with gr.Row(equal_height=False):
        with gr.Column():
            image_in = gr.Image(type="pil", label="📷 Upload an image (jpg, jpeg, png)")
            caption_in = gr.Textbox(
                label="Optional: add your own caption",
                placeholder="Leave blank for an automatic caption.",
            )
            generate_btn = gr.Button("✨ Generate narrative", variant="primary", size="lg")
        with gr.Column():
            practice_box = gr.Textbox(
                label="🧩 Your Practice Prompt",
                visible=False,
                interactive=False,
                elem_id="practice-box",
            )
            fade_slider = gr.Slider(
                label="Fade words (drag right to hide more from the end)",
                minimum=0, maximum=1, step=1, value=0, visible=False,
            )
            read_btn = gr.Button("🔊 Read Aloud")
            audio_out = gr.Audio(label="Narration", visible=True)
            with gr.Accordion("Show full narrative (supporter/teacher use)", open=False):
                full_text = gr.Textbox(label="", visible=False, interactive=False)

    generate_btn.click(
        on_generate,
        inputs=[image_in, caption_in],
        outputs=[narrative_state, practice_box, fade_slider, full_text],
    )
    fade_slider.change(
        on_fade, inputs=[narrative_state, fade_slider], outputs=[practice_box]
    )
    read_btn.click(on_read_aloud, inputs=[narrative_state], outputs=[audio_out])

    gr.HTML(
        "<div id='footer-note'>Powered by MC-LLaVA-3b on ZeroGPU. "
        "For best results, use clear photos with obvious subjects.</div>"
    )

_boot("blocks built; launching gradio")
demo.launch()
