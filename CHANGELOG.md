# Changelog

Format: [YYYY-MM-DD] | [vX.X.X] | [Type: Added/Fixed/Changed/Removed]

---

## [Unreleased]
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
