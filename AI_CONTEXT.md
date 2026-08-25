# AI Context — Smart CCTV Investigator

## Overview
- **Purpose**: A generative AI pipeline that transforms unstructured video footage into searchable, verifiable insights with anomaly detection and facial re-identification.
- **Stack**: Python, OpenCV, YOLO11, face_recognition, Ollama, Groq (Llama 4 Scout), ChromaDB, LangGraph
- **Status**: Active Development (Watchdog Agent & Premium UI Integrated)
- **Version**: 4.1.0
- **Last Updated**: 2026-04-28

## File Structure
```
/
├── src/
│   ├── app.py              # Main Streamlit UI
│   ├── vision_test.py      # YOLO-World initialization & testing
│   ├── face_engine.py      # Biometric identity matching
│   ├── vlm_engine.py       # Groq/Ollama narrative generation
│   ├── rag_engine.py       # ChromaDB event logging
├── data/
│   ├── known_faces/        # Profiles for re-identification
│   ├── videos/             # Source .mp4 files
│   ├── db/                 # ChromaDB persistence
├── models/
│   ├── custom_yolo_world.pt # Saved model with custom classes
```

## Key Components
| Component | File | Purpose | Status |
|-----------|------|---------|--------|
| Vision Pipeline | src/app.py | YOLO-World trigger (detects faces, weapons, etc.) | ✅ Done |
| Face Engine | src/face_engine.py | Handles identity matching against known_faces | ✅ Done |
| VLM Descriptor | src/vlm_engine.py | Parses frames into JSON via Groq/Ollama | ✅ Done |
| Watchdog Agent | [TBD] | Scores events for anomaly | 🛠️ In Progress |
| RAG DB Setup | src/rag_engine.py | Embeds events into ChromaDB | ✅ Done |
| LangGraph | [TBD] | Resolves natural language user queries | 🛠️ In Progress |

## Environment Variables
| Variable | Description | Required? |
|----------|-------------|-----------|
| OLLAMA_HOST | URL for local Ollama instance (default: localhost:11434) | No |
| GROQ_API_KEY| Key for Llama-4-Scout narrative generation | Yes (for Cloud) |

## Deployment
- Target: Hugging Face Spaces, Docker SDK, free CPU tier (16 GB RAM)
- `Dockerfile` at repo root: python:3.11-slim + cmake/build-essential (dlib), serves `src/app.py` on port 7860
- `README.md` carries HF Space metadata (`sdk: docker`, `app_port: 7860`)
- YOLO weights auto-download (`yolov8m-worldv2.pt`) on first boot; `*.pt` stays git-ignored
- Live Stream webcam page does not work in the cloud (server-side `cv2.VideoCapture(0)` has no camera); image/video upload pages do
- Use the Groq provider in the cloud; local Ollama is not available in the container

## Known Issues
- Environment: Requires `numpy<2.0` for pandas compatibility.

## Next Steps
- [ ] Implement advanced anomaly detection logic (Watchdog Agent)
- [ ] Connect LangGraph for complex natural language queries
- [ ] Optimize real-time live stream processing speed
