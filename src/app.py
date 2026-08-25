import os
# Fix for macOS OpenCV authorization/threading issue
os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLOWorld
from face_engine import FaceEngine
from vlm_engine import VLMEngine
import shutil
import time
from datetime import datetime
from watchdog_agent import WatchdogAgent

# Page configuration
st.set_page_config(
    page_title="CCTV Smart Investigator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for a Premium Look
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    
    <style>
    /* Global Styles */
    * { font-family: 'Outfit', sans-serif; }
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #020617 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    section[data-testid="stSidebar"] .stMarkdown h1 {
        color: #6366f1;
        font-weight: 700;
        letter-spacing: -1px;
    }
    
    /* Glassmorphism Cards */
    div.stButton > button {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4);
        border: none;
        color: white;
    }
    
    /* File Uploader Style */
    [data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.5);
        border: 2px dashed rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 2rem;
        backdrop-filter: blur(10px);
    }
    
    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(to right, #ffffff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Navigation Radio Fix */
    div[data-testid="stWidgetLabel"] p {
        font-weight: 600;
        color: #94a3b8;
    }
    
    /* Metric Boxes */
    div[data-testid="stMetricValue"] {
        color: #6366f1;
    }
    
    /* Info/Success Boxes */
    .stAlert {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        backdrop-filter: blur(5px);
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_yolo():
    if os.path.exists("models/custom_yolo_world.pt"):
        return YOLOWorld("models/custom_yolo_world.pt")
    model = YOLOWorld("yolov8m-worldv2.pt")
    model.set_classes(["face", "car", "bicycle", "motorcycle", "person", "cell phone", "laptop", "knife", "gun"])
    return model

@st.cache_resource
def load_face_engine():
    return FaceEngine()

@st.cache_resource
def load_vlm_engine(model_name, provider, api_key):
    return VLMEngine(model=model_name, provider=provider, api_key=api_key)

@st.cache_resource
def load_watchdog(api_key):
    return WatchdogAgent(api_key=api_key)

# Sidebar
with st.sidebar:
    st.title("🛡️ Investigator")
    page = st.radio("Navigation", ["🔍 General Detection", "🎬 Video Analysis", "📡 Live Stream", "👤 Biometrics", "📁 Known Faces"])
    
    st.divider()
    st.subheader("🧠 VLM Configuration")
    vlm_provider = st.selectbox("Provider", ["local", "groq", "nvidia"], index=1) # Default to Groq
    
    if vlm_provider == "local":
        vlm_model = st.selectbox("Model", ["moondream", "llama3.2-vision", "llava"], index=0)
        nim_key = None
    elif vlm_provider == "groq":
        vlm_model = st.text_input("Groq Model", value="meta-llama/llama-4-scout-17b-16e-instruct")
        nim_key = st.text_input("Groq API Key", type="password", value="key")
    else:
        vlm_model = st.text_input("NIM Model", value="nvidia/llama-3.2-nv-vision-70b-v1")
        nim_key = st.text_input("NVIDIA API Key", type="password", value="key")
    
    st.divider()
    if st.button("🔄 Refresh Models"):
        st.cache_resource.clear()
        st.success("Cache cleared!")
        st.rerun()
    st.info("MVP v0.1.0 - Local AI Engine")

# --- PAGE: LIVE STREAM ---
if page == "📡 Live Stream":
    st.title("Live AI Monitoring")
    
    if 'run_live' not in st.session_state: st.session_state.run_live = False
    if 'capturing_remaining' not in st.session_state: st.session_state.capturing_remaining = 0

    col_ctrl, col_idx, col_throttle, col_vlm_t = st.columns([1, 1, 1, 1])
    cam_idx = col_idx.selectbox("Camera Index", [0, 1, 2], index=0)
    ai_interval = col_throttle.slider("YOLO Interval (s)", 0.0, 5.0, 0.2, step=0.1)
    vlm_interval = col_vlm_t.slider("VLM Interval (s)", 1, 60, 10)
    
    if 'last_vlm_nar' not in st.session_state: st.session_state.last_vlm_nar = "System initialized. No events recorded yet."

    if not st.session_state.run_live:
        if col_ctrl.button("🚀 Start Live Camera"):
            st.session_state.run_live = True; st.rerun()
    else:
        if col_ctrl.button("🛑 Stop Camera"):
            st.session_state.run_live = False; st.rerun()

    col_live, col_stats = st.columns([3, 1])
    FRAME_WINDOW = col_live.image([])
    
    with col_stats:
        st.markdown("### 🎓 Live Training")
        existing_p = [d for d in os.listdir("data/known_faces") if os.path.isdir(os.path.join("data/known_faces", d))] if os.path.exists("data/known_faces") else []
        train_choice = st.selectbox("Choose Person", ["-- New Person --"] + existing_p)
        train_name = st.text_input("Name").strip() if train_choice == "-- New Person --" else train_choice
        num_photos = st.slider("Photos", 1, 30, 10)
        if st.button("📸 Batch Train"):
            if train_name: 
                st.session_state.capturing_remaining = num_photos
                st.session_state.current_train_name = train_name
            else: st.error("Enter a name.")
        
        st.divider()
        stats_placeholder = st.empty()
    
    if st.session_state.run_live:
        yolo = load_yolo(); face_eng = load_face_engine(); vlm = load_vlm_engine(vlm_model, vlm_provider, nim_key); cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            st.error(f"Camera failed."); st.session_state.run_live = False
        else:
            try:
                last_cap_t = 0; last_ai_t = 0; last_vlm_t = 0
                results = None; faces = []
                vlm_placeholder = col_stats.empty()
                
                while st.session_state.run_live:
                    ret, frame = cap.read()
                    if not ret: break
                    curr_t = time.time()
                    
                    # 1. YOLO Detection (Eyes)
                    if curr_t - last_ai_t >= ai_interval:
                        results = yolo.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), conf=0.25, verbose=False)
                        faces = face_eng.identify(frame); last_ai_t = curr_t
                        
                        # 2. VLM Trigger (Brain) - Rule: If person or weapon detected and > X s since last report
                        if results and len(results[0].boxes) > 0:
                            dets = [yolo.names[int(b.cls[0])] for b in results[0].boxes]
                            if any(d in ["person", "knife", "gun"] for d in dets) and (curr_t - last_vlm_t > vlm_interval):
                                with vlm_placeholder: st.spinner("🧠 AI Thinking...")
                                nar = vlm.describe_event(frame, identity=", ".join(set([n for _, n in faces])) if faces else "Unknown", 
                                                       detections=dets, 
                                                       last_report=st.session_state.last_vlm_nar)
                                
                                st.session_state.last_vlm_nar = nar.get("summary", "")
                                with vlm_placeholder.container():
                                    st.markdown(f"#### 🎙️ Live Narrative")
                                    if nar.get("alert_level") in ["High", "Critical"]:
                                        st.error(f"**{nar.get('alert_type')}**")
                                    st.write(nar.get("summary"))
                                    
                                    # 3. Watchdog Assessment
                                    watchdog = load_watchdog(nim_key)
                                    risk = watchdog.assess_threat(dets, [n for _, n in faces], nar.get("summary", ""))
                                    
                                    st.divider()
                                    col_risk, col_act = st.columns([1, 2])
                                    col_risk.metric("Threat Score", f"{risk['score']}%", delta=risk['level'], delta_color="inverse")
                                    col_act.warning(f"**Action:** {risk['action']}")
                                    
                                last_vlm_t = curr_t
                    
                    if st.session_state.capturing_remaining > 0 and curr_t - last_cap_t > 0.3:
                        p_dir = os.path.join("data/known_faces", st.session_state.current_train_name)
                        os.makedirs(p_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(p_dir, f"cap_{st.session_state.capturing_remaining}.jpg"), frame)
                        st.session_state.capturing_remaining -= 1; last_cap_t = curr_t
                        if st.session_state.capturing_remaining == 0:
                            st.toast("✅ Done!"); st.cache_resource.clear()

                    res_p = frame.copy()
                    if results:
                        for box in results[0].boxes:
                            b = box.xyxy[0].cpu().numpy(); c = int(box.cls[0])
                            cv2.rectangle(res_p, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
                            cv2.putText(res_p, f"{yolo.names[c]}", (int(b[0]), int(b[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    for loc, name in faces:
                        t, r, b, l = loc
                        cv2.rectangle(res_p, (l, t), (r, b), (139, 0, 255), 3)
                        cv2.putText(res_p, f"ID: {name}", (l, b + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (139, 0, 255), 2)
                    if st.session_state.capturing_remaining > 0:
                        cv2.putText(res_p, f"RECORDING: {st.session_state.capturing_remaining}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    
                    FRAME_WINDOW.image(cv2.cvtColor(res_p, cv2.COLOR_BGR2RGB))
                    with stats_placeholder.container():
                        st.write(f"**Objects:** {len(results[0].boxes) if results else 0}")
                        if faces: st.write(f"**People:** {', '.join(set([n for _, n in faces]))}")
                    time.sleep(0.01)
            finally: cap.release()

# --- PAGE: GENERAL DETECTION ---
elif page == "🔍 General Detection":
    st.title("General Analysis")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file); img_np = np.array(img); img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        col1, col2 = st.columns([2, 1]); col1.image(img, use_container_width=True)
        if st.button("🚀 Run Analysis"):
            model = load_yolo(); res = model.predict(img_np, conf=0.15); res_p = res[0].plot()
            eng = load_face_engine(); faces = eng.identify(img_bgr)
            st.session_state.last_analysis = {"frame": img_bgr, "ids": [n for _, n in faces], "dets": [model.names[int(b.cls[0])] for b in res[0].boxes]}
            for loc, name in faces:
                t, r, b, l = loc
                cv2.rectangle(res_p, (l, t), (r, b), (139, 0, 255), 3)
                cv2.putText(res_p, f"ID: {name}", (l, b + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (139, 0, 255), 2)
            col1.image(cv2.cvtColor(res_p, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                for _, n in faces: st.success(f"ID: {n}")
                for b in res[0].boxes: st.write(f"- {model.names[int(b.cls[0])]} ({float(b.conf[0]):.2f})")
        
        if 'last_analysis' in st.session_state:
            if st.button("🧠 Generate AI Narrative"):
                with st.spinner("Analyzing..."):
                    vlm = load_vlm_engine(vlm_model, vlm_provider, nim_key); d = st.session_state.last_analysis
                    nar = vlm.describe_event(d["frame"], identity=", ".join(set(d["ids"])) if d["ids"] else "Unknown", detections=d["dets"])
                    
                    # --- ALERT UI ---
                    lvl = nar.get("alert_level", "Low").strip().capitalize()
                    if lvl in ["High", "Critical"]:
                        st.error(f"🚨 ALERT: {lvl} Severity Detected!")
                        st.warning(f"**Type:** {nar.get('alert_type', 'General Security Breach')}")
                    elif lvl == "Medium":
                        st.warning(f"⚠️ Warning: {lvl} Severity")
                    else:
                        st.success("✅ Secure: Low Severity")

                    st.info(nar.get("summary", "Done"))
                    
                    # --- WATCHDOG ASSESSMENT ---
                    watchdog = load_watchdog(nim_key)
                    risk = watchdog.assess_threat(d["dets"], d["ids"], nar.get("summary", ""))
                    
                    with st.container(border=True):
                        st.markdown(f"### 🛡️ Watchdog Assessment: {risk['level']}")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Risk Score", f"{risk['score']}%")
                        c2.write(f"**Reason:** {risk['reason']}")
                        c3.write(f"**Recommended Action:** {risk['action']}")

                    with st.expander("View Full Report"):
                        st.json(nar)

# --- PAGE: VIDEO ANALYSIS ---
elif page == "🎬 Video Analysis":
    st.title("Video Investigation")
    u_vid = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    interval = st.slider("Interval (s)", 0.5, 10.0, 2.0)
    if u_vid:
        tfile = "data/videos/temp.mp4"; os.makedirs("data/videos", exist_ok=True)
        with open(tfile, "wb") as f: f.write(u_vid.getbuffer())
        st.video(u_vid)
        if st.button("🚀 Start"):
            cap = cv2.VideoCapture(tfile); fps = cap.get(cv2.CAP_PROP_FPS); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            yolo = load_yolo(); face_eng = load_face_engine(); vlm = load_vlm_engine(vlm_model, vlm_provider, nim_key)
            progress = st.progress(0); results_cont = st.container()
            step = int(fps * interval)
            for fno in range(0, total, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fno); ret, frame = cap.read()
                if not ret: break
                res = yolo.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), conf=0.15, verbose=False)
                if any(yolo.names[int(b.cls[0])] in ["person", "face", "knife", "gun"] for b in res[0].boxes):
                    faces = face_eng.identify(frame); res_p = res[0].plot()
                    ids = [n for _, n in faces]; nar = vlm.describe_event(frame, identity=", ".join(set(ids)) if ids else "Unknown", detections=[yolo.names[int(b.cls[0])] for b in res[0].boxes])
                    for loc, name in faces:
                        t, r, b, l = loc
                        cv2.rectangle(res_p, (l, t), (r, b), (139, 0, 255), 3)
                        cv2.putText(res_p, f"ID: {name}", (l, b + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (139, 0, 255), 2)
                    with results_cont:
                        with st.container():
                            st.markdown(f"#### 🚩 {fno/fps:.1f}s - {nar.get('action', 'Activity')}")
                            c1, c2 = st.columns([2, 1])
                            c1.image(cv2.cvtColor(res_p, cv2.COLOR_BGR2RGB), use_container_width=True)
                            with c2: 
                                st.write(f"**AI Insight:**")
                                st.info(nar.get('summary', ''))
                progress.progress(fno / total)
            cap.release(); st.success("Done!")

# --- PAGE: BIOMETRICS ---
elif page == "👤 Biometrics":
    st.title("Face ID")
    u_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if u_file:
        img = Image.open(u_file); img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR); st.image(img, use_container_width=True)
        if st.button("👤 Identify"):
            eng = load_face_engine(); faces = eng.identify(img_np)
            for loc, name in faces:
                t, r, b, l = loc; cv2.rectangle(img_np, (l, t), (r, b), (0, 255, 0), 2)
                cv2.putText(img_np, name, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            st.image(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB), use_container_width=True)

# --- PAGE: KNOWN FACES ---
elif page == "📁 Known Faces":
    st.title("Known Faces")
    with st.expander("Manage Profiles"):
        existing = [d for d in os.listdir("data/known_faces") if os.path.isdir(os.path.join("data/known_faces", d))] if os.path.exists("data/known_faces") else []
        c_n, _ = st.columns([2, 1]); choice = c_n.selectbox("Person", ["-- New Person --"] + existing)
        p_name = st.text_input("New Name").strip() if choice == "-- New Person --" else choice
        u_files = st.file_uploader("Photos", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
        if st.button("💾 Save") and p_name and u_files:
            p_dir = os.path.join("data/known_faces", p_name); os.makedirs(p_dir, exist_ok=True)
            for f in u_files:
                with open(os.path.join(p_dir, f.name), "wb") as o: o.write(f.getbuffer())
            st.success("Updated!"); st.cache_resource.clear(); st.rerun()
    if os.path.exists("data/known_faces"):
        for p in [d for d in os.listdir("data/known_faces") if os.path.isdir(os.path.join("data/known_faces", d))]:
            with st.container(border=True):
                c1, c2 = st.columns([5, 1]); c1.markdown(f"#### 👤 {p}")
                if c2.button(f"🗑️", key=f"df_{p}"):
                    shutil.rmtree(os.path.join("data/known_faces", p)); st.cache_resource.clear(); st.rerun()
                p_dir = os.path.join("data/known_faces", p); imgs = [f for f in os.listdir(p_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if imgs:
                    cols = st.columns(6)
                    for i, f in enumerate(imgs):
                        with cols[i % 6]:
                            st.image(os.path.join(p_dir, f), use_container_width=True)
                            if st.button("x", key=f"di_{p}_{f}"): os.remove(os.path.join(p_dir, f)); st.cache_resource.clear(); st.rerun()
                st.divider()
