# Smart CCTV Investigator - System Design Spec

## 1. Overview
The Smart CCTV Investigator transforms raw video footage into a searchable, queryable database using Generative AI, computer vision, and local LLMs. 

## 2. Architecture & Data Flow 
The pipeline consists of the following layers designed for the MVP (testing with a pre-recorded .mp4 file):

### 2.1 Vision Processing Layer (The Trigger)
- **Object Detection:** Ultralytics YOLOv11 nano model processes the .mp4. If an object (person, car) is detected for > 2 seconds, it triggers an event and extracts the clearest frame.
- **Face Recognition (Identity Matching):** 
  - Runs a face recognition matching process against a pre-populated `known_faces/` directory.
  - If a match is found (e.g., "Rahul"), it stores the subject identity to be passed to the LLM.

### 2.2 Event Structuring Layer (The Context)
- **Descriptor Model:** The extracted frame, along with any recognized identities, is passed to a local Vision-Language Model (Moondream or LLaVA via Ollama).
- **Output:** The VLM outputs structured JSON containing `{timestamp, object_id, action, location_context, duration}`.

### 2.3 Anomaly Detection (Watchdog Agent)
- The structured JSON is passed to a fast local LLM (Watchdog Agent).
- The agent compares the event against a system prompt defining "Normal" behavior.
- Tags the event with `Threat Level: High/Low` and an explanation if unusual.

### 2.4 Vector Search & Storage (Building Memory)
- Events and anomaly tags are converted to human-readable strings.
- Embedded using `SentenceTransformers` and stored in a local `ChromaDB`.

### 2.5 Query Agent (LangGraph)
- Resolves user natural language queries (e.g., "Show me unusual activity last night").
- Connects to ChromaDB to retrieve relevant events and reasons over them using a Local LLM to formulate an answer.

## 3. Tech Stack
- **Vision:** OpenCV, YOLO11 (`ultralytics`), `face_recognition`
- **Vision LLM:** Moondream / LLaVA (via `Ollama`)
- **Text LLM:** LLaMA / Mistral / Qwen (via `Ollama`)
- **Memory:** ChromaDB, SentenceTransformers
- **Routing/Orchestration:** LangGraph
