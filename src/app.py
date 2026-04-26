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

# Page configuration
st.set_page_config(
    page_title="CCTV Smart Investigator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f9fafb; }
    .stButton>button {
        background-color: #111827;
        color: white;
        border-radius: 8px;
    }
    h1, h2, h3 { color: #111827; }
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
def load_vlm_engine():
    return VLMEngine(model="moondream")

# Sidebar
with st.sidebar:
    st.title("🛡️ Investigator")
    page = st.radio("Navigation", ["🔍 General Detection", "🎬 Video Analysis", "📡 Live Stream", "👤 Biometrics", "📁 Known Faces"])
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

    col_ctrl, col_idx, col_throttle = st.columns([1, 1, 1])
    cam_idx = col_idx.selectbox("Camera Index", [0, 1, 2], index=0)
    ai_interval = col_throttle.slider("AI Interval (s)", 0.0, 5.0, 0.2, step=0.1)
    
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
        yolo = load_yolo(); face_eng = load_face_engine(); cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            st.error(f"Camera failed."); st.session_state.run_live = False
        else:
            try:
                last_cap_t = 0; last_ai_t = 0; results = None; faces = []
                while st.session_state.run_live:
                    ret, frame = cap.read()
                    if not ret: break
                    curr_t = time.time()
                    
                    if curr_t - last_ai_t >= ai_interval:
                        results = yolo.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), conf=0.25, verbose=False)
                        faces = face_eng.identify(frame); last_ai_t = curr_t
                    
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
                    vlm = load_vlm_engine(); d = st.session_state.last_analysis
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
            yolo = load_yolo(); face_eng = load_face_engine(); vlm = load_vlm_engine()
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
                        st.markdown(f"#### 🚩 {fno/fps:.1f}s - {nar.get('action', 'Activity')}")
                        c1, c2 = st.columns([2, 1])
                        c1.image(cv2.cvtColor(res_p, cv2.COLOR_BGR2RGB), use_container_width=True)
                        with c2: st.write(f"**AI Insight:** {nar.get('summary', '')}")
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
            with st.container():
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
