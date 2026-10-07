# YOLO + OpenCV Real-Time Object Detection

A Windows-friendly computer vision project using **Python 3.10, OpenCV and Ultralytics YOLO11**.

## Features

- Object detection on images
- Real-time webcam detection
- Video-file detection
- Bounding boxes, class labels and confidence scores
- Confidence threshold control
- FPS monitoring for real-time inference
- Annotated output images/videos

## Tech Stack

- Python 3.10
- OpenCV
- Ultralytics YOLO11
- NumPy
- PyTorch (installed as an Ultralytics dependency)

## Windows + VS Code Setup

Open the project folder in VS Code.

### 1. Create virtual environment

In the VS Code PowerShell terminal:

```powershell
py -3.10 -m venv .venv
```

### 2. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run image detection

```powershell
python detect_image.py --source sample.jpg
```

The first run downloads the YOLO11n model automatically.

### 5. Run webcam detection

```powershell
python detect_webcam.py
```

Press **Q** to stop.

### 6. Run video detection

```powershell
python detect_video.py --source input.mp4
```

The annotated video is saved under `outputs/`.

## Project Flow

```text
Image / Webcam / Video
        ↓
    OpenCV
        ↓
   YOLO11n Model
        ↓
Object Detection
        ↓
Bounding Boxes + Labels
        ↓
Confidence Filtering
        ↓
Annotated Output + FPS
```

## Important Interview Points

### What does OpenCV do here?

OpenCV handles image and video input/output, webcam capture, frame processing, drawing/visualization and video writing.

### What does YOLO do?

YOLO performs object detection and returns detected objects, their bounding boxes, class labels and confidence scores.

### Why use a small YOLO model?

`yolo11n.pt` is a lightweight model that is useful for faster inference and real-time experiments.

### What is the confidence threshold?

It controls the minimum confidence required for a detection to be displayed. This project uses `0.40` by default.

### How is FPS calculated?

The project measures the time between processed frames and calculates approximately:

`FPS = 1 / frame_processing_time`

### Did this project train YOLO?

No. This version uses a pretrained YOLO11n detection model. It focuses on building the inference and video-processing pipeline.

## GitHub

Do not upload the `.pt` model file, generated videos or virtual environment. They are excluded through `.gitignore`.

## Future Improvements

- Object tracking
- Custom dataset training
- Precision/recall and mAP evaluation
- REST API deployment with FastAPI
- ONNX export and optimized inference
- Edge-device deployment
