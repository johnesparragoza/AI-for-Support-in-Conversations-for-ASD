---
title: Image Narrator
emoji: 🖼️
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
short_description: Image-to-narrative with prompt fading and read-aloud
---

# Image Narrator

Upload an image and get a friendly, first-person narrative description, with a
prompt-fading practice mode and text-to-speech read-aloud. Built for
conversation practice and social support (ASD support project).

Powered by [`visheratin/MC-LLaVA-3b`](https://huggingface.co/visheratin/MC-LLaVA-3b)
and Streamlit, served via Docker on the free CPU tier.

## Notes

- Runs on the free CPU tier (16 GB RAM). The 3B model loads on CPU, so the first
  request after a cold start is slow (~30–90s) while the model loads and runs.
- `HF_TOKEN` is optional (the model is public). If you set it, add it under
  **Settings → Variables and secrets** as a secret named `HF_TOKEN`.
- The container runs `app.py` (see `Dockerfile`).
