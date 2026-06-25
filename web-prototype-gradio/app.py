import os
import re
import sys
import tempfile

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
        "Describe this image with a single first-person sentence.\n"
        "Make it short and simple, no more than 10 words.\n"
        "Use first-person perspective (e.g., 'I see...', 'I feel...').\n"
        f"Caption: {caption}\n"
        "Narrative:\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = processor(prompt, [image], model, return_tensors="pt")
    inputs = {
        k: v.to(model.device) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=64,
            use_cache=True,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    generated_text = processor.tokenizer.decode(output[0], skip_special_tokens=True)
    parts = re.split(r"<\|im_start\|>assistant", generated_text)
    narrative = parts[-1] if len(parts) > 1 else generated_text
    return narrative.replace("<|im_end|>", "").strip()


# --- Gradio event handlers (run on CPU; only _generate_narrative hits GPU) -
def on_generate(image, caption):
    if image is None:
        raise gr.Error("Please upload an image first.")
    narrative = _generate_narrative(image.convert("RGB"), caption or " ")
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


with gr.Blocks(title="AImage Narrator") as demo:
    gr.Markdown(
        "# AImage Narrator 🪄\n"
        "Upload an image, get a friendly first-person narrative, then practice "
        "by fading words out."
    )

    narrative_state = gr.State("")

    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="Upload an image (jpg, jpeg, png)")
            caption_in = gr.Textbox(
                label="Optional: add your own caption",
                placeholder="Leave blank for an automatic caption.",
            )
            generate_btn = gr.Button("Generate narrative", variant="primary")
        with gr.Column():
            practice_box = gr.Textbox(
                label="Your Practice Prompt", visible=False, interactive=False
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

    gr.Markdown(
        "---\nPowered by MC-LLaVA-3b on ZeroGPU. For best results, use clear "
        "photos with obvious subjects."
    )

_boot("blocks built; launching gradio")
demo.launch()
