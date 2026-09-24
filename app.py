
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import tempfile
from detector import ObjectDetector

st.set_page_config(page_title="Smart Vision Pro", layout="wide", page_icon="👁️")

# Cache the model loading, using model_name as part of the key so it reloads if changed
@st.cache_resource
def load_detector(model_name):
    return ObjectDetector(model_name=model_name)

st.title("Smart Vision Pro 👁️")
st.write("### Advanced Object Detection, Counting & Analytics using YOLOv8")

# Sidebar Configuration
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.selectbox(
    "Choose YOLO Model",
    options=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
    index=0,
    help="Nano (fastest), Small (balanced), Medium (most accurate)"
)

# Load the AI model dynamically
detector = load_detector(model_choice)

st.sidebar.header("🎯 Inference Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.50, 
    step=0.05,
    help="Filter out predictions below this confidence score."
)

iou_threshold = st.sidebar.slider(
    "IoU (NMS) Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.45, 
    step=0.05,
    help="Intersection over Union threshold for overlapping bounding boxes."
)

st.sidebar.header("🔍 Class & Counting Filters")
person_only_mode = st.sidebar.checkbox(
    "👤 Person-Only Mode (Count & Track People)", 
    value=False,
    help="Restricts AI detection exclusively to human persons."
)

all_classes = list(detector.class_names.values())

if person_only_mode:
    st.sidebar.info("👤 Person-only mode active")
    person_ids = [k for k, v in detector.class_names.items() if v == 'person']
    selected_class_ids = person_ids if person_ids else [0]
else:
    selected_class_names = st.sidebar.multiselect(
        "Filter Classes (Leave empty to detect all)",
        options=all_classes,
        default=[]
    )
    selected_class_ids = [k for k, v in detector.class_names.items() if v in selected_class_names]
    if len(selected_class_ids) == 0:
        selected_class_ids = None  # None means detect all classes

st.sidebar.header("🔒 Visualization & Privacy")
blur_objects = st.sidebar.checkbox(
    "Blur Detected Objects (Privacy Mode)", 
    value=False,
    help="Applies Gaussian blur on detected bounding boxes."
)

# Main UI Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🖼️ Image File & Snapshot", 
    "📹 Real-Time Live Webcam", 
    "🎥 Video File Detection", 
    "ℹ️ Features & Architecture"
])

# Session State Initialization
if "detection_data" not in st.session_state:
    st.session_state.detection_data = None
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

# TAB 1: IMAGE FILE & SNAPSHOT
with tab1:
    input_mode = st.radio(
        "Select Input Source",
        options=["📁 Upload Image File", "📷 Live Webcam Snapshot"],
        horizontal=True
    )
    
    if input_mode == "📁 Upload Image File":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("Capture a photo with your webcam")
    
    if uploaded_file is not None:
        file_identifier = getattr(uploaded_file, "name", "webcam_capture") + str(getattr(uploaded_file, "size", ""))
        
        # Reset cached detection if a different input is provided
        if st.session_state.last_uploaded_file != file_identifier:
            st.session_state.last_uploaded_file = file_identifier
            st.session_state.detection_data = None

        try:
            # Explicitly convert to 3-channel RGB to handle RGBA and grayscale formats
            image = Image.open(uploaded_file).convert("RGB")
            image_array = np.array(image)
            
            st.write("---")
            # For webcam snapshot, auto-detect immediately without requiring button click
            # For file upload, allow triggering detection via button
            should_detect = False
            if input_mode == "📷 Live Webcam Snapshot":
                should_detect = True
            else:
                if st.button("Detect Objects", type="primary"):
                    should_detect = True
            
            if should_detect:
                with st.spinner(f"Analyzing with {model_choice}..."):
                    annotated_img, detections, total_objects, class_counts = detector.detect_objects(
                        image_array, 
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        classes=selected_class_ids,
                        blur=blur_objects
                    )
                    st.session_state.detection_data = {
                        "annotated_img": annotated_img,
                        "detections": detections,
                        "total_objects": total_objects,
                        "class_counts": class_counts
                    }
                    
            if st.session_state.detection_data is not None:
                data = st.session_state.detection_data
                annotated_img = data["annotated_img"]
                detections = data["detections"]
                total_objects = data["total_objects"]
                class_counts = data["class_counts"]

                st.subheader("Detection Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Original / Input Image**")
                    st.image(image, use_container_width=True)
                    
                with col2:
                    st.write("**Detection Output**")
                    st.image(annotated_img, use_container_width=True)
                
                st.write("---")
                st.write(f"### Total Objects Detected: {total_objects}")
                if class_counts:
                    summary_str = " | ".join([f"{k.capitalize()}: {v}" for k, v in class_counts.items()])
                    st.write(f"**Summary Breakdown:** {summary_str}")
                
                if total_objects > 0:
                    df = pd.DataFrame(detections)
                    df_ui = df.copy()
                    df_ui['confidence_pct'] = (df_ui['confidence'] * 100).round(1).astype(str) + "%"
                    df_ui['bounding_box'] = df_ui.apply(lambda row: f"({row['x1']}, {row['y1']}, {row['x2']}, {row['y2']})", axis=1)
                    display_df = df_ui[['class', 'confidence_pct', 'bounding_box']].rename(
                        columns={'class': 'Object', 'confidence_pct': 'Confidence', 'bounding_box': 'Bounding Box (x1, y1, x2, y2)'}
                    )
                    st.table(display_df)
                    
                    csv_data = df[['class', 'confidence', 'x1', 'y1', 'x2', 'y2']].to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_data,
                        file_name="detections.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No objects detected with the current settings.")
            else:
                st.write("**Input Preview**")
                st.image(image, use_container_width=True)
        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")
    else:
        st.session_state.last_uploaded_file = None
        st.session_state.detection_data = None

# TAB 2: REAL-TIME LIVE WEBCAM
with tab2:
    st.subheader("📹 Real-Time Live Webcam Detection")
    st.write("Turn on your webcam to detect objects continuously in real time—no photo capture required!")
    
    run_webcam = st.checkbox("🔴 Start Live Webcam Feed", key="webcam_toggle")
    
    if run_webcam:
        # Use DirectShow backend on Windows to ensure reliable webcam frame reading
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
            
        if not cap.isOpened():
            st.error("⚠️ Unable to access webcam. Please ensure your camera is connected and permissions are granted.")
        else:
            frame_window = st.image([])
            metrics_placeholder = st.empty()
            
            while run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Unable to read frame from webcam. Please verify camera permissions or select a different camera.")
                    break
                
                # Convert BGR (OpenCV format) to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Perform continuous object detection
                annotated_img, detections, total_objects, class_counts = detector.detect_objects(
                    frame_rgb,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    classes=selected_class_ids,
                    blur=blur_objects
                )
                
                # Display real-time annotated frame
                frame_window.image(annotated_img, use_container_width=True)
                
                # Update real-time detection counts
                if class_counts:
                    summary_str = " | ".join([f"{k.capitalize()}: {v}" for k, v in class_counts.items()])
                    metrics_placeholder.markdown(f"### 📊 Live Count: **{total_objects}** objects detected ({summary_str})")
                else:
                    metrics_placeholder.markdown(f"### 📊 Live Count: **{total_objects}** objects detected")
            
            cap.release()

# TAB 3: VIDEO DETECTION
with tab3:
    st.subheader("🎥 Video Object Detection")
    st.write("Upload a video clip to run deep learning detection frame-by-frame.")
    video_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov", "mkv"], key="video_uploader")
    
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile.close()
        
        st.video(tfile.name)
        
        if st.button("🚀 Start Video Analysis", type="primary", key="process_video_btn"):
            detector.reset_tracker()
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                st.error("Could not read frames from the video.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                frame_placeholder = st.empty()
                
                all_video_detections = []
                frame_idx = 0
                total_raw_detections = 0
                unique_tracked_ids = set()
                unique_class_counts = {}
                
                # Sample frames for smooth processing
                step = max(1, total_frames // 120) if total_frames > 120 else 1
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_idx += 1
                    if frame_idx % step != 0:
                        continue
                    
                    annotated_frame, detections, count, counts = detector.process_video_frame(
                        frame,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        classes=selected_class_ids,
                        blur=blur_objects
                    )
                    
                    total_raw_detections += count
                    
                    for d in detections:
                        d_copy = d.copy()
                        d_copy["frame"] = frame_idx
                        all_video_detections.append(d_copy)
                        
                        t_id = d.get("track_id")
                        c_name = d.get("class")
                        if t_id is not None:
                            unique_tracked_ids.add(t_id)
                            if c_name not in unique_class_counts:
                                unique_class_counts[c_name] = set()
                            unique_class_counts[c_name].add(t_id)
                    
                    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(
                        annotated_rgb, 
                        caption=f"Processing Frame {frame_idx}/{total_frames} | Unique Entities So Far: {len(unique_tracked_ids) if unique_tracked_ids else count}", 
                        use_container_width=True
                    )
                    
                    progress = min(1.0, frame_idx / total_frames)
                    progress_bar.progress(progress)
                    status_text.text(f"Tracking frame {frame_idx} of {total_frames} ({int(progress * 100)}%)...")
                    
                cap.release()
                progress_bar.progress(1.0)
                status_text.success("✅ Video processing & tracking complete!")
                
                total_unique = len(unique_tracked_ids) if unique_tracked_ids else total_raw_detections
                
                st.write("---")
                st.write(f"### 📊 Unique Object Tracking Summary")
                st.write(f"👤 **Total Unique Entities Tracked (Counted Once):** `{total_unique}`")
                
                if unique_class_counts:
                    u_summary = " | ".join([f"{k.capitalize()}: {len(v_set)} unique" for k, v_set in unique_class_counts.items()])
                    st.write(f"**Unique Breakdown:** {u_summary}")
                else:
                    st.write(f"**Aggregated Raw Detections:** {total_raw_detections}")
                
                st.caption(f"ℹ️ Total raw detection instances across all frames: {total_raw_detections}")
                
                if all_video_detections:
                    v_df = pd.DataFrame(all_video_detections)
                    st.dataframe(v_df)
                    v_csv = v_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Tracked Video Detections CSV",
                        data=v_csv,
                        file_name="tracked_video_detections.csv",
                        mime="text/csv",
                        key="download_video_csv"
                    )

# TAB 4: FEATURES & ARCHITECTURE
with tab4:
    st.write("### 🚀 All Features & Capabilities")
    st.markdown("""
    - **📹 Real-Time Live Webcam**: Continuous 30 FPS webcam detection feed with zero button clicks required.
    - **📷 Snapshot & File Upload**: Real-time snapshot capture from webcam or file upload.
    - **🎥 Full Video Detection**: Process `.mp4`, `.avi`, `.mov` video files frame-by-frame with visual playback and progress tracking.
    - **👤 Person-Only Counting Mode**: One-click toggle in sidebar to isolate, count, and track people.
    - **🎛️ Dynamic Model Switching**: Switch between Nano (`yolov8n`), Small (`yolov8s`), and Medium (`yolov8m`).
    - **🎯 Threshold Filtering**: Interactive sliders for confidence score and IoU overlap (NMS).
    - **🔍 Granular Class Filtering**: Multi-select target classes (e.g. *Cars, Dogs, Backpacks*).
    - **🔒 Privacy Mode (Dynamic Blurring)**: Automatically blur detected objects with adaptive Gaussian blurring.
    - **📊 CSV Report Export**: Export structured detection records with coordinates `(x1, y1, x2, y2)` and confidence percentages.
    """)
