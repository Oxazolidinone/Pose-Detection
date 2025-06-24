# Pose Classification System

AI-powered pose classification system using MoveNet and LightGBM to detect and classify yoga/exercise poses from images, videos, and webcam.

![Pose Classification Demo](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19+-orange.svg)

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Frontend  │ -> │   FastAPI Server │ -> │  MoveNet Model  │
│   (HTML/JS)     │    │    (Python)      │    │   (TensorFlow)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ LightGBM Model  │
                       │ (Classification)│
                       └─────────────────┘
```

## System Requirements

- **Python**: 3.11+
- **RAM**: Minimum 4GB
- **GPU**: Optional (CPU works fine)
- **Webcam**: For real-time features

## Installation

### 1. Clone repository
```bash
git clone <repository-url>
cd Pose-Yolo
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare dataset
Organize data folders as follows:
```
DATASET/
├── TRAIN/
   ├── goddess/          # Goddess pose images
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   └── plank/           # Plank pose images
       ├── image1.jpg
       ├── image2.jpg
       └── ...
```
### 4. 
```bash
python train_ml_model.py
```
Training process creates 3 files:
- `pose_classifier.pkl`: Trained LightGBM model
- `pose_scaler.pkl`: Data normalization scaler
- `class_mapping.pkl`: Class name mappings

```bash
python main.py
```
Server runs at: `http://127.0.0.1:8001`

## Usage

### 1. Image Upload
- Select "Upload Image" tab
- Choose image file from computer
- Click "Analyze Pose"
- View results with keypoints drawn on image

### 2. URL Analysis
- Select "Image URL" tab
- Enter direct image link
- Click "Analyze Pose from URL"

### 3. Video Analysis
- Select "Upload Video" tab
- Upload video file (MP4, AVI, MOV, WMV)
- Choose max frames to analyze (5-100)
- View frame-by-frame results with statistics

### 4. Real-time Webcam
- Select "Live Camera" tab
- Click "Start Camera" to enable webcam
- System analyzes in real-time every second
- Click "Stop Camera" to stop

### Port and Host
```python
uvicorn.run(app, host="127.0.0.1", port=8001)
```
## 📝 License

This project is released under MIT License.

## Credits
- **MoveNet**: Google Research
- **LightGBM**: Microsoft
- **FastAPI**: Sebastián Ramirez
- **Bootstrap**: Twitter

