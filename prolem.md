# 🧠 Smart CCTV Investigator  
### GenAI + RAG + LangGraph + Local LLM Project

---

# 🚀 Overview

The **Smart CCTV Investigator** is an intelligent surveillance system that transforms raw video footage into **searchable, explainable, and queryable insights** using Generative AI.

Instead of manually watching hours of footage, users can interact with the system using natural language:

> “Show me suspicious activity near the entrance after midnight.”

This system combines:
- Computer Vision (event detection)
- RAG (retrieval of relevant events)
- Local LLM (reasoning + explanation)
- LangGraph (multi-step decision workflows)

---

# 🎯 Core Objective

Convert unstructured CCTV footage → structured knowledge → intelligent querying.

---

# 🧩 Problem Statement

Traditional CCTV systems:
- Record everything, understand nothing
- Require manual review
- Provide only basic alerts (motion detection)
- Cannot answer complex questions

---

# 💡 Solution

A system that:
1. Detects and logs events from video feeds
2. Converts events into structured textual descriptions
3. Stores them in a searchable vector database
4. Uses an LLM to answer user queries intelligently

---

# ⚙️ System Architecture

## 1. Video Input Layer
- Sources:
  - RTSP streams (IP cameras)
  - DVR/NVR recordings
  - Uploaded video files

---

## 2. Vision Processing Layer

### Tasks:
- Object Detection (person, vehicle, bag, etc.)
- Activity Recognition (enter, exit, loitering)

### Output Example:
```
[Timestamp: 02:13 AM]
Person detected entering through main gate.
Stayed for 3 minutes.
```

---

## 3. Event Structuring Layer

Convert raw detections into meaningful logs:

| Field        | Example                        |
|-------------|-------------------------------|
| Timestamp   | 02:13 AM                      |
| Object      | Person                        |
| Action      | Entering                      |
| Location    | Main Gate                     |
| Duration    | 3 minutes                     |

---

## 4. Embedding + Storage (RAG)

- Convert events → embeddings
- Store in:
  - FAISS / Chroma DB

This allows:
- Semantic search (not just keywords)

---

## 5. LangGraph Agent Flow

### Flow Steps:

1. **Query Understanding**
   - Identify intent (search / analyze / compare)

2. **Retriever**
   - Fetch relevant events from vector DB

3. **Reasoning Layer**
   - Analyze patterns
   - Compare timestamps
   - Detect anomalies

4. **Response Generation**
   - Generate human-like explanation

---

## 6. Local LLM

Runs fully offline:
- Interprets queries
- Performs reasoning
- Generates answers

---

# 🏠 Use Cases

---

## 🏡 1. Home Security

### Features:
- Detect late-night intrusions
- Track repeated visitors
- Identify unusual patterns

### Example Queries:
- “Did anyone enter after 1 AM?”
- “Show all unknown visitors this week”
- “Was the door left open?”

---

## 🏢 2. Office Monitoring

### Features:
- Track employee entry/exit
- Detect unauthorized access
- Monitor attendance patterns

### Example Queries:
- “Who stayed after office hours?”
- “Was there activity on Sunday?”
- “List all entries between 8–9 AM”

---

## 🏭 3. Industrial Security

### Features:
- Restricted zone monitoring
- Equipment usage tracking
- Safety violation detection

### Example Queries:
- “Did anyone enter restricted area?”
- “Was helmet missing in zone B?”
- “Show unusual movement near machines”

---

## 🛍️ 4. Retail Analytics

### Features:
- Customer movement tracking
- Peak hours analysis
- Behavior understanding

### Example Queries:
- “When is the store most crowded?”
- “Where do customers spend most time?”
- “Did anyone leave without purchasing?”

---

## 🚓 5. Advanced Security / Investigation

### Features:
- Multi-day tracking of individuals
- Suspicious behavior detection
- Timeline reconstruction

### Example Queries:
- “Did the same person appear multiple times?”
- “Show all activity before the incident”
- “Find loitering behavior near gate”

---

# 🧠 Intelligent Capabilities

---

## 🔍 Semantic Search
Find events even if wording differs:
- “intruder” → “unknown person detected”

---

## ⏱️ Temporal Reasoning
Understands time-based queries:
- “after midnight”
- “last 3 days”

---

## 🧩 Pattern Detection
- Repeated visits
- Unusual durations
- Deviations from normal behavior

---

## 🧠 Context Awareness
- Understands location + object + time together

---

# 🛠️ Tech Stack

### LLM:
- Local models (Mistral / LLaMA)

### Framework:
- LangGraph

### Vision:
- YOLOv8 / OpenCV

### Vector DB:
- FAISS / Chroma

### Embeddings:
- Sentence Transformers

---

# 🔥 Advanced Features (Next Level)

---

## 👤 Face Re-Identification
Track same person across multiple timestamps

---

## ⚠️ Suspicious Behavior Detection
- Loitering
- Repeated visits
- Odd hours movement

---

## 📄 Auto Report Generator
Generate incident reports:
```
Summary:
A person entered at 2:13 AM and stayed for 3 minutes.
No exit detected.
```

---

## 🔔 Smart Alerts
- “Person detected in restricted area”
- “Unknown face detected at night”

---

# 🧪 Challenges

---

## ⚠️ Accuracy
- False detections
- Lighting conditions

---

## ⚠️ Performance
- Real-time processing is heavy

---

## ⚠️ Storage
- Large video + embeddings

---

## ⚠️ Privacy
- Must handle data securely

---

# 🚀 Future Scope

- Multi-camera correlation
- Real-time alerts with GenAI reasoning
- Integration with IoT devices
- Voice-based querying

---

# 🧠 Why This Project Matters

This is not just a project.

It is:
- A real-world problem solver
- A combination of AI domains (CV + NLP + Systems)
- A product-level idea

---

# 🏁 Conclusion

The Smart CCTV Investigator transforms surveillance from passive recording → active intelligence.

It reduces manual effort, increases security, and enables decision-making through AI.

---

**Tagline:**  
> “Don’t watch footage. As