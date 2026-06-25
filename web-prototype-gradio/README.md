---
title: AImage Narrator (Gradio + ZeroGPU)
emoji: 🪄
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# AImage Narrator — Gradio + ZeroGPU

Experimental ZeroGPU port of the Streamlit `web-prototype`. Upload an image,
get a short first-person narrative from MC-LLaVA-3b, then practice by fading
words out. Includes text-to-speech read-aloud.

## Deploying this as a ZeroGPU Space

1. Create a **new** Space at https://huggingface.co/new-space
   - **SDK:** Gradio
   - **Hardware:** select **ZeroGPU** (requires a PRO subscription on the owning account)
2. Push these files (`app.py`, `requirements.txt`, `README.md`) to the Space repo.
   The README front matter above is what tells the Space it's a Gradio app.
3. (Optional) Add an `HF_TOKEN` secret under *Settings → Variables and secrets*.
   The model is public, so this is only needed to avoid download rate limits.

The existing Streamlit Space is unaffected — this is a parallel deployment so you
can fall back to it if ZeroGPU doesn't work out.

## Known risk to validate

ZeroGPU requires **torch ≥ 2.8 / Python 3.12**, whereas the model was originally
validated on torch 2.2 / Python 3.11. We keep `transformers==4.40.2` to preserve
MC-LLaVA's working remote-code path. If the model errors on the newer torch, the
first thing to try is setting `use_cache=False` in `model.generate(...)` (slower,
but sidesteps the cache-format mismatch that newer transformers/torch can trigger
in the remote code).
