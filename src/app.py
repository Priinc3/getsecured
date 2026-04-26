import os
# Fix for macOS OpenCV authorization/threading issue
os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLOWorld
from face_engine import FaceEngine
import shutil

# Page configuration
st.set_page_config(
    page_title="CCTV Smart Investigator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Antigravity White Theme)
st.markdown("""
    <style>
    .main { background-color: #f9fafb; }
    .stButton>button {
        background-color: #111827;
        color: white;
        border-radius: 8px;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        background-color: #111827;
        color: white;
    }
    .card {
        background-color: white;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    h1, h2, h3 { color: #111827; font-family: 'Inter', sans-serif; }
    </style>
""", unsafe_allow_html=True)

# Helper: Load Models
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

# Sidebar Navigation
with st.sidebar:
    st.title("🛡️ Investigator")
    page = st.radio("Navigation", ["🔍 General Detection", "🎬 Video Analysis", "📡 Live Stream", "👤 Biometrics", "📁 Known Faces"])
    st.divider()

    if st.button("🔄 Refresh Models"):
        st.cache_resource.clear()
        st.success("Cache cleared!")
        st.rerun()

    st.info("MVP v0.1.0 - Local AI Engine")

# --- PAGE: GENERAL DETECTION ---
if page == "🔍 General Detection":
    st.title("General Object Detection & Identification")
    st.write("Detect objects and identify known faces in a single pass.")
    
    uploaded_file = st.file_uploader("Upload an image or frame...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        img_np_orig = np.array(img)
        img_bgr = cv2.cvtColor(img_np_orig, cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns([2, 1])
        with col1: st.image(img, caption="Original Image", width=None, use_container_width=True)
        
        if st.button("🚀 Run Comprehensive Analysis"):
            model = load_yolo()
            results = model.predict(img_np_orig, conf=0.15)
            res_plotted = results[0].plot()
            
            engine = load_face_engine()
            faces = engine.identify(img_bgr)
            
            for loc, name in faces:
                top, right, bottom, left = loc
                cv2.rectangle(res_plotted, (left, top), (right, bottom), (139, 0, 255), 3)
                cv2.putText(res_plotted, f"ID: {name}", (left, bottom + 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (139, 0, 255), 2)
            
            with col1:
                st.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), caption="Results", width=None, use_container_width=True)
                
            with col2:
                st.subheader("Analysis Summary")
                if faces:
                    st.markdown("### 👤 Identities Found")
                    for _, name in faces:
                        if name == "Unknown": st.warning("Unknown Person Detected")
                        else: st.success(f"Recognized: {name}")
                
                st.markdown("### 📦 Objects Detected")
                boxes = results[0].boxes
                if len(boxes) == 0: st.write("No other objects found.")
                for box in boxes:
                    label = model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    st.markdown(f"- **{label}** ({conf:.2f})")

# --- PAGE: VIDEO ANALYSIS ---
elif page == "🎬 Video Analysis":
    st.title("Automated Video Investigation")
    st.write("Process long footage and extract significant events.")
    uploaded_video = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])
    col_settings, _ = st.columns([1, 2])
    with col_settings:
        interval = st.slider("Analyze every (seconds)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)

    if uploaded_video:
        tfile = "data/videos/temp_upload.mp4"
        os.makedirs("data/videos", exist_ok=True)
        with open(tfile, "wb") as f: f.write(uploaded_video.getbuffer())
        st.video(uploaded_video)

        if st.button("🚀 Start Processing"):
            cap = cv2.VideoCapture(tfile)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            yolo = load_yolo()
            face_eng = load_face_engine()
            
            progress_bar = st.progress(0)
            results_container = st.container()
            frame_step = int(fps * interval)
            
            for fno in range(0, total_frames, frame_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
                ret, frame = cap.read()
                if not ret: break
                
                res = yolo.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), conf=0.15, verbose=False)
                if any(yolo.names[int(b.cls[0])] in ["person", "face", "knife", "gun"] for b in res[0].boxes):
                    faces = face_eng.identify(frame)
                    res_p = res[0].plot()
                    for loc, name in faces:
                        top, right, bottom, left = loc
                        cv2.rectangle(res_p, (left, top), (right, bottom), (139, 0, 255), 3)
                        cv2.putText(res_p, f"ID: {name}", (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (139, 0, 255), 2)
                    
                    with results_container:
                        st.markdown(f"#### 🚩 Event at {fno/fps:.1f}s")
                        c1, c2 = st.columns([2, 1])
                        c1.image(cv2.cvtColor(res_p, cv2.COLOR_BGR2RGB), width=None, use_container_width=True)
                        with c2:
                            if faces: 
                                for _, name in faces: st.success(f"Recognized: {name}")
                            st.write(f"Detected: {', '.join(set([yolo.names[int(b.cls[0])] for b in res[0].boxes]))}")
                progress_bar.progress(min(fno / total_frames, 1.0))
            cap.release()
            st.success("Analysis Complete!")

# --- PAGE: LIVE STREAM ---
elif page == "📡 Live Stream":
    st.title("Live AI Monitoring")
    st.write("Real-time detection from your device camera.")
    run_camera = st.checkbox("Toggle Camera On/Off")
    col_live, col_stats = st.columns([3, 1])
    FRAME_WINDOW = col_live.image([])
    stats_placeholder = col_stats.empty()
    
    if run_camera:
        yolo = load_yolo()
        face_eng = load_face_engine()
        # Explicitly try AVFOUNDATION for macOS
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        
        if not cap.isOpened():
            st.error("Could not open camera. Ensure Terminal has Camera permissions in System Settings > Privacy & Security.")
        
        while run_camera:
            ret, frame = cap.read()
            if not ret: break
            
            res = yolo.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), conf=0.2, verbose=False)
            faces = face_eng.identify(frame)
            res_p = res[0].plot()
            for loc, name in faces:
                t, r, b, l = loc
                cv2.rectangle(res_p, (l, t), (r, b), (139, 0, 255), 3)
                cv2.putText(res_p, f"ID: {name}", (l, b + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (139, 0, 255), 2)
            
            FRAME_WINDOW.image(cv2.cvtColor(res_p, cv2.COLOR_BGR2RGB))
            with stats_placeholder.container():
                st.markdown("### 📊 Live Stats")
                st.write(f"**Objects:** {len(res[0].boxes)}")
                if faces: st.write(f"**People:** {', '.join(set([n for _, n in faces]))}")
        cap.release()
    else:
        st.info("Toggle to start. (Note: On macOS, grant Camera access to your Terminal/IDE)")

# --- PAGE: BIOMETRICS ---
elif page == "👤 Biometrics":
    st.title("Forensic Facial Recognition")
    u_file = st.file_uploader("Upload person frame...", type=["jpg", "png", "jpeg"])
    if u_file:
        img = Image.open(u_file)
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        col1, col2 = st.columns([2, 1])
        col1.image(img, caption="Input Frame", width=None, use_container_width=True)
        if st.button("👤 Identify Person"):
            engine = load_face_engine()
            faces = engine.identify(img_np)
            for loc, name in faces:
                t, r, b, l = loc
                cv2.rectangle(img_np, (l, t), (r, b), (0, 255, 0), 2)
                cv2.putText(img_np, name, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            col1.image(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB), width=None, use_container_width=True)
            with col2:
                st.subheader("Results")
                if not faces: st.warning("No faces detected.")
                for _, name in faces:
                    if name == "Unknown": st.error("⚠️ Unknown person detected!")
                    else: st.success(f"✅ Identity: {name}")

# --- PAGE: KNOWN FACES ---
elif page == "📁 Known Faces":
    st.title("Known Faces Directory")
    with st.expander("👤 Manage Profiles", expanded=True):
        existing = [d for d in os.listdir("data/known_faces") if os.path.isdir(os.path.join("data/known_faces", d))] if os.path.exists("data/known_faces") else []
        c_n, _ = st.columns([2, 1])
        choice = c_n.selectbox("Person", ["-- New Person --"] + existing)
        p_name = st.text_input("New Name").strip() if choice == "-- New Person --" else choice
        u_files = st.file_uploader("Photos", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
        if st.button("💾 Save") and p_name and u_files:
            p_dir = os.path.join("data/known_faces", p_name)
            os.makedirs(p_dir, exist_ok=True)
            for f in u_files:
                with open(os.path.join(p_dir, f.name), "wb") as o: o.write(f.getbuffer())
            st.success("Updated!"); st.cache_resource.clear(); st.rerun()

    st.subheader("Library")
    if os.path.exists("data/known_faces"):
        for p in [d for d in os.listdir("data/known_faces") if os.path.isdir(os.path.join("data/known_faces", d))]:
            with st.container():
                c_t, c_d = st.columns([5, 1])
                c_t.markdown(f"#### 👤 {p}")
                if c_d.button(f"🗑️ Delete", key=f"d_f_{p}"):
                    shutil.rmtree(os.path.join("data/known_faces", p))
                    st.cache_resource.clear(); st.rerun()
                p_dir = os.path.join("data/known_faces", p)
                imgs = [f for f in os.listdir(p_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if imgs:
                    cols = st.columns(6)
                    for i, f in enumerate(imgs):
                        with cols[i % 6]:
                            st.image(os.path.join(p_dir, f), width=None, use_container_width=True)
                            if st.button("x", key=f"d_i_{p}_{f}"):
                                os.remove(os.path.join(p_dir, f))
                                st.cache_resource.clear(); st.rerun()
                st.divider()
