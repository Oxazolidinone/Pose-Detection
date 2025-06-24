from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import os
import base64
import io
from PIL import Image
from movenet_detector import MoveNetDetector
import pickle
import requests
import tempfile
from urllib.parse import urlparse
import time

app = FastAPI(title="Pose Classification API")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class PoseClassifier:
    def __init__(self, model_path="pose_classifier.pkl", scaler_path="pose_scaler.pkl", mapping_path="class_mapping.pkl"):
        self.detector = MoveNetDetector()
        self.model = None
        self.scaler = None
        self.class_mapping = None
        
        if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(mapping_path):
            self.load_model(model_path, scaler_path, mapping_path)
        else:
            print("Model files not found. Please train the model first using train_ml_model.py")

    def load_model(self, model_path, scaler_path, mapping_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(mapping_path, 'rb') as f:
            self.class_mapping = pickle.load(f)
        print("Model loaded successfully!")

    def predict_pose_from_array(self, image_array, confidence_threshold=0.8):
        if self.model is None:
            return None, None, None
        
        keypoints_with_scores = self.detector.detect_pose(image_array)
        keypoints = keypoints_with_scores[0, 0, :, :]
        features = keypoints.flatten()
        
        features_scaled = self.scaler.transform([features])
        
        # For LightGBM Booster object, use predict() method which returns probabilities
        probabilities = self.model.predict(features_scaled, num_iteration=self.model.best_iteration)
        
        # Get the predicted class (index of highest probability)
        prediction = np.argmax(probabilities[0])
        confidence = max(probabilities[0])
        
        # Kiểm tra confidence threshold
        if confidence < confidence_threshold:
            class_name = "Unknown/Low Confidence"
        else:
            class_name = self.class_mapping['class_names'][prediction]
        
        return class_name, confidence, keypoints_with_scores

    def predict_pose_from_url(self, image_url):
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            # Chuyển đổi thành numpy array
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return None, None, None, "Could not decode image from URL"
            
            return self.predict_pose_from_array(image)
        except requests.exceptions.RequestException as e:
            return None, None, None, f"Error downloading image: {str(e)}"
        except Exception as e:
            return None, None, None, f"Error processing image: {str(e)}"
    
    def predict_pose_from_video(self, video_path, max_frames=30):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None, "Could not open video file"
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
            
            results = []
            processed_frames = []
            
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                class_name, confidence, keypoints_with_scores = self.predict_pose_from_array(frame)
                
                if class_name is not None:
                    pose_frame = self.detector.draw_keypoints(frame, keypoints_with_scores)
                    text = f"Frame {i+1}: {class_name} ({confidence:.2f})"
                    cv2.putText(pose_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    timestamp = frame_idx / fps if fps > 0 else 0
                    time_text = f"Time: {timestamp:.1f}s"
                    cv2.putText(pose_frame, time_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    results.append({
                        "frame_number": i + 1,
                        "timestamp": timestamp,
                        "class_name": class_name,
                        "confidence": float(confidence),
                        "frame_image": image_to_base64(pose_frame)
                    })
                    
                    processed_frames.append(pose_frame)
            
            cap.release()
            
            if not results:
                return None, "No poses detected in video"
            summary = {
                "total_frames_analyzed": len(results),
                "video_duration": duration,
                "most_common_pose": max(set([r["class_name"] for r in results]), 
                                      key=[r["class_name"] for r in results].count),
                "average_confidence": np.mean([r["confidence"] for r in results])
            }
            
            return results, summary
            
        except Exception as e:
            return None, f"Error processing video: {str(e)}"

classifier = PoseClassifier()

def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_pose(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse({"error": "Could not decode image"}, status_code=400)
        
        class_name, confidence, keypoints_with_scores = classifier.predict_pose_from_array(image)
        
        if class_name is None:
            return JSONResponse({"error": "Model not loaded"}, status_code=500)
        
        pose_image = classifier.detector.draw_keypoints(image, keypoints_with_scores)
        
        text = f"Pose: {class_name} ({confidence:.2f})"
        cv2.putText(pose_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        original_img_base64 = image_to_base64(image)
        result_img_base64 = image_to_base64(pose_image)
        
        return JSONResponse({
            "success": True,
            "class_name": class_name,
            "confidence": float(confidence),
            "original_image": original_img_base64,
            "result_image": result_img_base64
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/predict_webcam")
async def predict_webcam(image_data: str = Form(...)):
    try:
        header, encoded = image_data.split(',', 1)
        data = base64.b64decode(encoded)
        nparr = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse({"error": "Could not decode image"}, status_code=400)
        
        class_name, confidence, keypoints_with_scores = classifier.predict_pose_from_array(image)
        
        if class_name is None:
            return JSONResponse({"error": "Model not loaded"}, status_code=500)
        
        pose_image = classifier.detector.draw_keypoints(image, keypoints_with_scores)
        
        text = f"Pose: {class_name} ({confidence:.2f})"
        cv2.putText(pose_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        result_img_base64 = image_to_base64(pose_image)
        
        return JSONResponse({
            "success": True,
            "class_name": class_name,
            "confidence": float(confidence),
            "result_image": result_img_base64
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/predict_url")
async def predict_from_url(url: str = Form(...)):
    """Dự đoán tư thế từ URL ảnh"""
    try:
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return JSONResponse({"error": "Invalid URL format"}, status_code=400)
        
        # Dự đoán từ URL
        result = classifier.predict_pose_from_url(url)
        
        if len(result) == 4:  # Có lỗi
            _, _, _, error_msg = result
            return JSONResponse({"error": error_msg}, status_code=400)
        
        class_name, confidence, keypoints_with_scores = result
        
        if class_name is None:
            return JSONResponse({"error": "Model not loaded"}, status_code=500)
        
        response = requests.get(url, timeout=10)
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        pose_image = classifier.detector.draw_keypoints(image, keypoints_with_scores)
        
        text = f"Pose: {class_name} ({confidence:.2f})"
        cv2.putText(pose_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        original_img_base64 = image_to_base64(image)
        result_img_base64 = image_to_base64(pose_image)
        
        return JSONResponse({
            "success": True,
            "class_name": class_name,
            "confidence": float(confidence),
            "original_image": original_img_base64,
            "result_image": result_img_base64,
            "source_url": url
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/predict_video")
async def predict_from_video(file: UploadFile = File(...), max_frames: int = Form(30)):
    try:
        if not file.content_type.startswith('video/'):
            return JSONResponse({"error": "Please upload a video file"}, status_code=400)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_video_path = tmp_file.name
        
        try:
            results, summary = classifier.predict_pose_from_video(tmp_video_path, max_frames)
            if results is None:
                return JSONResponse({"error": summary}, status_code=400)
            
            return JSONResponse({
                "success": True,
                "filename": file.filename,
                "summary": summary,
                "results": results,
                "total_frames": len(results)
            })
            
        finally:
            if os.path.exists(tmp_video_path):
                os.unlink(tmp_video_path)
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/health")
async def health_check():
    model_status = "loaded" if classifier.model is not None else "not_loaded"
    return JSONResponse({
        "status": "healthy",
        "model_status": model_status,
        "classes": classifier.class_mapping['class_names'] if classifier.class_mapping else []
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
