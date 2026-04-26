# AI Context — Smart CCTV Investigator

## Overview
- **Purpose**: A generative AI pipeline that transforms unstructured video footage into searchable, verifiable insights with anomaly detection and facial re-identification.
- **Stack**: Python, OpenCV, YOLO11, face_recognition, Ollama (Moondream/LLaVA/LLaMA), ChromaDB, LangGraph
- **Status**: In Development
- **Version**: 0.1.0
- **Last Updated**: 2026-04-25

## File Structure
```
/
├── src/
│   ├── vision_test.py      # YOLO-World initialization & testing
├── data/
│   ├── known_faces/        # Profiles for re-identification
│   ├── videos/             # Source .mp4 files
│   ├── output/             # Processed events/frames
├── models/
│   ├── custom_yolo_world.pt # Saved model with custom classes
```

## Key Components
| Component | File | Purpose |
|-----------|------|---------|
| Vision Pipeline | src/vision_test.py | YOLO-World trigger (detects faces, weapons, etc.) |
| Face Engine | src/face_engine.py | Handles identity matching against known_faces |
| VLM Descriptor | [TBD] | Parses frames into JSON via Ollama |
| Watchdog Agent | [TBD] | Scores events for anomaly |
| RAG DB Setup | [TBD] | Embeds events into ChromaDB |
| LangGraph | [TBD] | Resolves natural language user queries |

## Environment Variables
| Variable | Description | Required? |
|----------|-------------|-----------|
| OLLAMA_HOST | URL for local Ollama instance (default: localhost:11434) | No |

## API Endpoints
N/A (Local script execution for MVP)

## Known Issues
- None yet

## Next Steps
- [ ] Implement Face Recognition + YOLO pre-processing step
- [ ] Connect single frame to Ollama Moondream for JSON extraction
- [ ] Build Watchdog prompt and RAG pipeline
