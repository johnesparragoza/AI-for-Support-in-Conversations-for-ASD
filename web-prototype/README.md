---
title: Image Narrator
emoji: 🖼️
colorFrom: purple
colorTo: indigo
sdk: streamlit
sdk_version: 1.58.0
app_file: app.py
pinned: false
---

# Image Narrator

Upload an image and get a friendly, first-person narrative description, with a
prompt-fading practice mode and text-to-speech read-aloud. Built for
conversation practice and social support (ASD support project).

Powered by [`visheratin/MC-LLaVA-3b`](https://huggingface.co/visheratin/MC-LLaVA-3b)
and Streamlit.

## Notes

- Runs on the free CPU tier (16 GB RAM). The 3B model loads on CPU, so the first
  request after a cold start is slow (~30–90s) while the model loads and runs.
- `HF_TOKEN` is optional (the model is public). If you set it, add it under
  **Settings → Variables and secrets** as a secret named `HF_TOKEN`.
