---
title: Smart CCTV Investigator
emoji: ":camera:"
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Smart CCTV Investigator

YOLO-World object detection + face recognition identity matching, with a VLM
(Groq / Ollama) narrative and 0-100 threat scoring per frame.

## Run locally

```bash
streamlit run src/app.py   # from the repo root
```

## Deploy (Hugging Face Spaces)

Docker SDK, free CPU tier (16 GB RAM). Paste a Groq API key in the app sidebar
at runtime, or set `GROQ_API_KEY` as a Space secret.
