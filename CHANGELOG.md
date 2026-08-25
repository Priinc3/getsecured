# Changelog

Format: [YYYY-MM-DD] | [vX.X.X] | [Type: Added/Fixed/Changed/Removed]

---

## [Unreleased]
- Dockerfile: added git to apt deps; `mkdir -p models` before baking — bake step now passes end-to-end
- Fixed live-server errors: bake `custom_yolo_world.pt` at Docker build (runtime no longer needs CLIP for `set_classes`); migrated 6x `use_column_width` → `use_container_width` in app.py
- Deployed to Render (`cctv-investigator`, Docker runtime, free plan) — live at https://cctv.princeprojects.in
- Hostinger DNS: added CNAME `cctv` → `cctv-investigator.onrender.com`
- Dockerfile: CPU-only torch installed before ultralytics, serial dlib compile — fixes OOM build failures on small builders
- Streamlit binds to `$PORT` (Render compat), falls back to 7860
- Added `Dockerfile` (python:3.11-slim + cmake/dlib build deps) and `.dockerignore` for Hugging Face Spaces Docker SDK deployment, port 7860
- Added `README.md` with HF Space metadata (`sdk: docker`, `app_port: 7860`)
- Removed `chromadb`, `sentence-transformers`, `langgraph` from requirements.txt (unused, no imports anywhere)
- Created `src/face_engine.py` for biometric identity matching
- Implemented `FaceEngine` class with automated indexing of `data/known_faces/`
- Created project directory structure (`src`, `data`, `models`)
- Implemented `src/vision_test.py` to initialize YOLO-World with custom classes
- Classes defined: face, car, bike, person, phone, laptop, knife, gun
- Updated AI_CONTEXT with new file structure

---

## [0.1.0] — 2026-04-25

### Added
- Project initialized
- Brainstorming session completed
- Spec design, AI_CONTEXT, and CHANGELOG created

### Technical Notes
- Selected Hybrid pipeline (YOLO trigger + VLM descriptor) for efficiency
- Selected Watchdog agent approach for immediate anomaly detection
- Added Face Re-identification step to vision pipeline
