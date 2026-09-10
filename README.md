# Smart Vision 👁️

An intelligent, interactive computer vision web application for real-time object detection, video analysis, classification, bounding-box analytics, and privacy blurring powered by **Ultralytics YOLOv8** and **Streamlit**.

---

## 📌 Project Overview
**Smart Vision** is developed for the **First-Day Challenge: Smart Vision**. It enables users to upload images, record live photos via webcam, or analyze complete video files through deep learning object detection models, filter predictions dynamically, visualize bounding boxes, and export structured detection datasets.

---

## 🚀 Key Features

### Core Requirements
- **Multi-Source Image Input:** Accepts standard images (JPG, PNG, JPEG) and live webcam snapshots.
- **Deep Learning Object Detection:** Identifies 80 COCO object classes (including people, vehicles, animals, appliances, and everyday items).
- **Visual Bounding Boxes:** Highlights detected objects with class labels and confidence percentages.
- **Metric Aggregations:** Displays the total number of detected entities and a per-class summary breakdown.
- **Detailed Analytics Table:** Presents bounding box coordinates `(x1, y1, x2, y2)`, class names, and confidence scores.

### 🌟 Bonus & Advanced Features
- **📷 Live Webcam Snapshot:** Capture real-time photos directly from your browser webcam (`st.camera_input`).
- **🎥 Full Video Detection:** Upload `.mp4`, `.avi`, `.mov` video files to run frame-by-frame deep learning analysis with live progress playback and summary exports.
- **👤 Dedicated Person-Only Counting Mode:** Sidebar toggle to filter and count only human entities in images and videos.
- **🎛️ Dynamic Model Selection:** Switch between YOLOv8 Nano (`yolov8n.pt`), Small (`yolov8s.pt`), and Medium (`yolov8m.pt`).
- **📊 CSV Report Export:** One-click download of all detection metrics and bounding box coordinates for external analysis.
- **🎯 Interactive Confidence & IoU Filtering:** Real-time sliders for confidence thresholds and Non-Maximum Suppression (IoU) overlap control.
- **🔍 Granular Class Filtering:** Multiselect widget to isolate specific objects (e.g. *Car & Bus only*).
- **🔒 Privacy Mode (Dynamic Blurring):** Replaces bounding boxes with adaptive Gaussian blurring over detected objects to protect identities.

---

## 🛠️ Technologies Used

| Technology | Purpose |
| :--- | :--- |
| **Python 3.10+** | Core programming language |
| **Ultralytics YOLOv8** | State-of-the-art Deep Learning object detection and classification |
| **Streamlit** | Modern, reactive web application framework |
| **OpenCV (`opencv-python-headless`)** | Video decoding, image transformations, bounding box rendering, and Gaussian blurring |
| **NumPy & Pillow (PIL)** | Numerical arrays, color channel normalization (RGB/RGBA), and image I/O |
| **Pandas** | Tabular data manipulation and CSV report generation |

---

## 🧠 Approach

1. **Modular Architecture:** Separated core computer vision and inference logic (`detector.py`) from user interaction and UI presentation (`app.py`).
2. **Robust Input Pipeline:** Sanitized and converted all incoming image buffers (uploaded files or webcam feeds) into 3-channel RGB NumPy arrays to prevent channel mismatch errors with transparent PNGs (RGBA) or grayscale images.
3. **Video Processing Loop:** Utilized OpenCV frame decoders with adaptive sampling to provide responsive real-time visual feedback on video uploads.
4. **Optimized Model Caching:** Leveraged `@st.cache_resource` to load YOLO weights once per model selection, eliminating redundant downloads and reducing latency.
5. **State Persistence:** Preserved detection states, tables, and rendered outputs in `st.session_state` to prevent UI resets when downloading reports or interacting with widgets.
6. **Adaptive Visual Feedback:** Implemented dynamic odd kernel sizing bounded by bounding box dimensions for Gaussian blurring to guarantee smooth privacy filters without OpenCV dimensional errors.

---

## 🤖 AI & Tools Used

- **Ultralytics YOLOv8 Pre-trained Models:** Lightweight, high-accuracy neural network trained on the COCO dataset.
- **Antigravity / AI Coding Assistant:** Used for rapid prototyping, architecture design, edge-case debugging (session state persistence, dynamic kernel sizing, RGBA handling, video stream handling), and code review.

---

## 💻 Implementation Details

### File Structure
```text
smart-vision/
├── app.py              # Streamlit frontend, tabs, video processor, state management
├── detector.py         # YOLO ObjectDetector class, image & video inference engine
├── requirements.txt    # Production dependencies
├── README.md           # Project documentation and submission report
└── .gitignore          # Environment and model weight exclusions
```

---

## ⚠️ Limitations & Future Work

- **Live Streaming WebRTC:** Currently supports static image uploads, webcam snapshots, and video file processing. A future iteration can add WebRTC streaming for continuous live video feeds.
- **Custom Domain Scope:** Currently leverages the 80 COCO classes. Specialized objects (e.g. industrial defects, medical imaging) would require fine-tuning custom YOLO weights.
- **Hardware Acceleration:** Runs on CPU by default; GPU acceleration with CUDA/TensorRT can be enabled for ultra-high FPS processing.

---

## ⚙️ Installation & Running

### 1. Clone & Set Up Environment
```bash
# Navigate to project directory
cd Task_Qloxa/smart-vision

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

---

## 🎤 Quick Demo / Viva Talking Points
- **Architecture:** "We decoupled the YOLO backend in `detector.py` from the Streamlit UI in `app.py` for reusability."
- **Reliability:** "We handled RGBA channel normalizations and adaptive blurring kernels so the app never crashes on unusual images or tiny bounding boxes."
- **Comprehensive Bonus Features:** "Beyond core detection, we included video file detection, live webcam capture, dedicated person-only counting mode, CSV exports, dynamic model switching, IoU/confidence tuning, and privacy blurring."
