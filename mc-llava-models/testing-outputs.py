"""
test_model_outputs.py
----------------------
Standalone harness for inspecting what MC-LLaVA-3b actually returns, *outside*
of Streamlit. Runs the model exactly the way app.py does, but dumps the raw
output in several forms so you can see what your extraction code is really
working with.

Why this exists:
  app.py decodes with skip_special_tokens=True and then splits on
  "<|im_start|>assistant". If those markers are registered as special tokens,
  decode() strips them, the split finds nothing, and you silently get the whole
  prompt echoed back. This harness decodes BOTH ways so you can confirm which
  case you're in, and compares two extraction strategies side by side.

Usage:
  # one image, ALL built-in prompt variants in a single model load (the iteration loop)
  python testing-outputs.py --image path/to/photo.jpg

  # just the activity-focused variants you're tuning
  python testing-outputs.py --image photo.jpg --variant activity_simple,activity_action

  # a whole folder of images, default = all variants
  python testing-outputs.py --image path/to/folder/

  # add a caption, save results to JSON
  python testing-outputs.py --image photo.jpg --caption "my dog" --output results.json

  # try your own prompt template (use {caption} as a placeholder); overrides --variant
  python testing-outputs.py --image photo.jpg --prompt-file my_prompt.txt

Edit the PROMPT_VARIANTS dict below to add/tweak wordings, then re-run. Each run
prints a per-variant diagnostic block plus a compact COMPARISON table at the end.

Env:
  Set HF_TOKEN in your environment if the model repo needs auth.
"""

import argparse
import io
import json
import os
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_ID = "visheratin/MC-LLaVA-3b"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _chat(body):
    """Wrap an instruction body in MC-LLaVA's ChatML user/assistant turns."""
    return (
        "<|im_start|>user\n"
        "<image>\n"
        f"{body}"
        "Caption: {caption}\n"
        "Narrative:\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# A registry of prompt variants so you can A/B many wordings in ONE model load
# (loading the 3B model is the slow part — don't pay it per prompt). Iterate by
# editing the bodies below, then run:  python testing-outputs.py --image foo.jpg
#
#   "baseline"  -> exactly what app.py ships today (present-tense "I see..."),
#                  kept as a control to compare every new wording against.
#   The "activity_*" variants push toward the goal: a first-person PAST-TENSE
#   sentence describing the activity, e.g. "I played soccer with my friends."
PROMPT_VARIANTS = {
    # Control: current production wording (present-tense perception).
    "baseline": _chat(
        "Describe this image with a single first-person sentence.\n"
        "Make it short and simple, no more than 10 words.\n"
        "Use first-person perspective (e.g., 'I see...', 'I feel...').\n"
    ),
    # Past-tense activity, closest edit to the baseline.
    "activity_simple": _chat(
        "Describe the activity in this image as one first-person, past-tense sentence.\n"
        "Say what I did, as if I did the activity myself.\n"
        "Make it short and simple, no more than 10 words.\n"
        "Start with 'I' and a past-tense verb (e.g., 'I played...', 'I made...').\n"
    ),
    # Action-focused: point the model at the person and what they're doing.
    "activity_action": _chat(
        "Look at what the person in this image is doing.\n"
        "Write one short sentence describing that activity in first-person past tense,\n"
        "as if I did it myself. Begin with 'I' and a past-tense action verb.\n"
        "No more than 10 words.\n"
    ),
    # Minimal: terse instruction; small models often follow short prompts best.
    "activity_minimal": _chat(
        "What activity is happening here? Answer in one short first-person,\n"
        "past-tense sentence that starts with 'I'.\n"
    ),
}

# Default prompt mirrors app.py so outputs match what you see in the app.
DEFAULT_PROMPT_TEMPLATE = PROMPT_VARIANTS["baseline"]


def load_model(force_cpu=False):
    """Load processor + model the same way app.py does."""
    import transformers
    if not hasattr(transformers.PreTrainedModel, "_supports_sdpa"):
        transformers.PreTrainedModel._supports_sdpa = True

    hf_token = os.getenv("HF_TOKEN")
    use_cuda = torch.cuda.is_available() and not force_cpu
    device = "cuda" if use_cuda else "cpu"

    print(f"Loading {MODEL_ID} on {device} (this can take a minute)...")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True, token=hf_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        trust_remote_code=True,
        token=hf_token,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)
    return processor, model


def extract_via_split(generated_text):
    """app.py's current strategy: split the decoded string on the assistant marker."""
    parts = re.split(r"<\|im_start\|>assistant", generated_text)
    narrative = parts[-1] if len(parts) > 1 else generated_text
    return narrative.replace("<|im_end|>", "").strip(), len(parts)


def extract_via_slice(output_ids, input_len, tokenizer):
    """Alternative strategy: drop the prompt tokens by length, decode only the rest.

    For multimodal models the input_ids length may not line up perfectly with
    what generate() prepends (image-token expansion), so treat this as a
    diagnostic comparison, not gospel.
    """
    gen_only = output_ids[input_len:]
    text = tokenizer.decode(gen_only, skip_special_tokens=True)
    return text.strip()


def run_one(image_path, caption, prompt_template, processor, model, max_new_tokens,
            variant_name=None, image=None):
    if image is None:
        image = Image.open(image_path).convert("RGB")
    prompt = prompt_template.format(caption=caption if caption else " ")

    inputs = processor(prompt, [image], model, return_tensors="pt")
    inputs = {
        k: (v.to(model.device) if torch.is_tensor(v) else v)
        for k, v in inputs.items()
    }
    input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else None

    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    tok = processor.tokenizer
    output_ids = output[0]
    raw_with_special = tok.decode(output_ids, skip_special_tokens=False)
    raw_without_special = tok.decode(output_ids, skip_special_tokens=True)

    # app.py extracts from the skip_special_tokens=True version:
    narrative_split, n_parts = extract_via_split(raw_without_special)
    # also try splitting the WITH-special version, in case markers were stripped:
    narrative_split_special, n_parts_special = extract_via_split(raw_with_special)
    narrative_slice = (
        extract_via_slice(output_ids, input_len, tok) if input_len is not None else None
    )

    return {
        "image": str(image_path),
        "variant": variant_name,
        "caption": caption,
        "elapsed_sec": round(elapsed, 1),
        "total_output_tokens": len(output_ids),
        "input_token_len": input_len,
        "markers_survive_decode": "<|im_start|>assistant" in raw_without_special,
        "raw_with_special_tokens": raw_with_special,
        "raw_without_special_tokens": raw_without_special,
        "extract_split_on_clean": narrative_split,          # what app.py currently does
        "extract_split_on_clean_nparts": n_parts,
        "extract_split_on_raw": narrative_split_special,     # split before stripping specials
        "extract_split_on_raw_nparts": n_parts_special,
        "extract_by_token_slice": narrative_slice,           # length-based alternative
    }


def pretty_print(r):
    bar = "=" * 70
    print(f"\n{bar}\nIMAGE: {r['image']}")
    if r.get("variant"):
        print(f"VARIANT: {r['variant']}")
    if r["caption"]:
        print(f"CAPTION: {r['caption']!r}")
    print(f"time: {r['elapsed_sec']}s | output tokens: {r['total_output_tokens']} "
          f"| input len: {r['input_token_len']}")
    print(f"assistant marker survives skip_special_tokens=True? "
          f"{r['markers_survive_decode']}")
    print("-" * 70)
    print("RAW (skip_special_tokens=False):")
    print(repr(r["raw_with_special_tokens"]))
    print("-" * 70)
    print("RAW (skip_special_tokens=True):")
    print(repr(r["raw_without_special_tokens"]))
    print("-" * 70)
    print(f"app.py extraction  -> {r['extract_split_on_clean']!r} "
          f"(split parts={r['extract_split_on_clean_nparts']})")
    print(f"split on raw       -> {r['extract_split_on_raw']!r} "
          f"(split parts={r['extract_split_on_raw_nparts']})")
    print(f"token-slice        -> {r['extract_by_token_slice']!r}")
    print(bar)


def gather_images(path):
    p = Path(path)
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    return [p]


def build_variants(args):
    """Resolve which (name -> template) prompt variants to run this invocation."""
    if args.prompt_file:
        # A custom template file overrides the registry entirely.
        return {Path(args.prompt_file).stem: Path(args.prompt_file).read_text()}
    if args.variant == "all":
        return dict(PROMPT_VARIANTS)
    chosen = {}
    for name in args.variant.split(","):
        name = name.strip()
        if name not in PROMPT_VARIANTS:
            raise SystemExit(
                f"Unknown variant {name!r}. Available: {', '.join(PROMPT_VARIANTS)} (or 'all')."
            )
        chosen[name] = PROMPT_VARIANTS[name]
    return chosen


def print_comparison(results):
    """Compact end-of-run table: one line per (image, variant) -> narrative.

    This is the view you actually iterate on — scan it to see which wording
    produces the first-person past-tense activity sentences you want.
    """
    if not results:
        return
    bar = "#" * 70
    print(f"\n{bar}\nCOMPARISON (extraction app.py uses)\n{bar}")
    last_img = None
    for r in results:
        if r["image"] != last_img:
            print(f"\n{Path(r['image']).name}:")
            last_img = r["image"]
        print(f"  [{r.get('variant') or 'custom':<16}] {r['extract_split_on_clean']!r}")


def main():
    ap = argparse.ArgumentParser(description="Inspect raw MC-LLaVA-3b outputs.")
    ap.add_argument("--image", required=True, help="Image file or a folder of images.")
    ap.add_argument("--caption", default="", help="Optional caption to inject into the prompt.")
    ap.add_argument(
        "--variant",
        default="all",
        help="Comma-separated prompt variant name(s) to run, or 'all' (default). "
             f"Available: {', '.join(PROMPT_VARIANTS)}.",
    )
    ap.add_argument("--prompt-file", help="Text file with a prompt template ({caption} placeholder). "
                                          "Overrides --variant.")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--output", help="Write all results to this JSON file.")
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    args = ap.parse_args()

    variants = build_variants(args)

    images = gather_images(args.image)
    if not images:
        print(f"No images found at {args.image}")
        return

    print(f"Running {len(variants)} variant(s) x {len(images)} image(s): "
          f"{', '.join(variants)}")
    processor, model = load_model(force_cpu=args.cpu)

    results = []
    for img in images:
        try:
            # Decode the image once per image, reuse across all variants.
            image = Image.open(img).convert("RGB")
        except Exception as e:
            print(f"\n[ERROR opening] {img}: {e}")
            continue
        for vname, template in variants.items():
            try:
                r = run_one(img, args.caption, template, processor, model,
                            args.max_new_tokens, variant_name=vname, image=image)
                results.append(r)
                pretty_print(r)
            except Exception as e:
                print(f"\n[ERROR] {img} [{vname}]: {e}")

    print_comparison(results)

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nSaved {len(results)} result(s) to {args.output}")


if __name__ == "__main__":
    main()