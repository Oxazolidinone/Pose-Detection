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
import pickle
import requests
import tempfile
from urllib.parse import urlparse
import time
import mediapipe as mp

# Import custom exercise detection modules
from exercise_detection import ExerciseDetector

app = FastAPI(title="Exercise Form Correction API")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# MediaPipe pose detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class ExerciseFormAnalyzer:
    def __init__(self):
        self.detector = ExerciseDetector()
        self.confidence_threshold = 0.8
        
    def analyze_image(self, image, exercise_type="auto"):
        """Analyze single image for exercise form - returns correct/incorrect"""
        results = {}
        
        with mp_pose.Pose(
            min_detection_confidence=self.confidence_threshold,
            min_tracking_confidence=self.confidence_threshold
        ) as pose:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb_image)
            
            if pose_results.pose_landmarks:
                # Analyze form based on specified exercise type
                if exercise_type == "auto":
                    # For now, we'll just analyze as general form
                    # In future, could add auto-detection logic
                    exercise_type = "general"
                
                form_analysis = self.detector.analyze_exercise_form(pose_results, exercise_type)
                
                # Draw pose landmarks
                annotated_image = image.copy()
                mp_drawing.draw_landmarks(
                    annotated_image,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1),
                )
                
                # Add form feedback text
                status_color = (0, 255, 0) if form_analysis['is_correct'] else (0, 0, 255)
                status_text = "CORRECT FORM" if form_analysis['is_correct'] else "INCORRECT FORM"
                
                cv2.putText(annotated_image, status_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
                
                # Add main feedback
                cv2.putText(annotated_image, form_analysis['feedback'], (10, 70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add first error if any
                if form_analysis['errors']:
                    error_text = form_analysis['errors'][0][:50] + "..." if len(form_analysis['errors'][0]) > 50 else form_analysis['errors'][0]
                    cv2.putText(annotated_image, f"Issue: {error_text}", (10, 100), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                
                results = {
                    "exercise_type": exercise_type,
                    "is_correct": form_analysis['is_correct'],
                    "confidence": form_analysis['confidence'],
                    "form_score": form_analysis['score'],
                    "errors": form_analysis['errors'],
                    "corrections": form_analysis['corrections'],
                    "feedback": form_analysis['feedback'],
                    "pose_detected": True,
                    "annotated_image": annotated_image
                }
            else:
                results = {
                    "exercise_type": exercise_type,
                    "is_correct": False,
                    "confidence": 0.0,
                    "form_score": 0.0,
                    "errors": ["No pose detected"],
                    "corrections": ["Ensure full body is visible", "Check lighting conditions"],
                    "feedback": "Please ensure your full body is visible in the image",
                    "pose_detected": False,
                    "annotated_image": image
                }
                
        return results
    
    def analyze_video(self, video_path, max_frames=30, exercise_type="auto"):
        """Analyze video for exercise form throughout"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Sample frames evenly
        frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
        
        results = []
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            timestamp = frame_idx / fps if fps > 0 else 0
            
            # Analyze frame
            analysis = self.analyze_image(frame, exercise_type)
            analysis["frame_number"] = i + 1
            analysis["timestamp"] = float(timestamp)
            analysis["frame_image"] = self.image_to_base64(analysis["annotated_image"])
            
            # Remove annotated_image to avoid serialization issues
            del analysis["annotated_image"]
            
            results.append(analysis)
        
        cap.release()
        
        if not results:
            return None, "No valid frames found in video"
        
        # Generate summary
        correct_frames = [r for r in results if r["is_correct"]]
        form_scores = [r["form_score"] for r in results if r["pose_detected"]]
        all_errors = []
        all_corrections = []
        for r in results:
            all_errors.extend(r["errors"])
            all_corrections.extend(r["corrections"])
        
        summary = {
            "total_frames_analyzed": len(results),
            "correct_frames": len(correct_frames),
            "accuracy_percentage": float((len(correct_frames) / len(results)) * 100) if results else 0.0,
            "video_duration": float(total_frames / fps) if fps > 0 else 0.0,
            "average_form_score": float(np.mean(form_scores)) if form_scores else 0.0,
            "common_errors": list(set(all_errors)),
            "recommended_corrections": list(set(all_corrections)),
            "overall_feedback": self._generate_overall_feedback(correct_frames, results)
        }
        
        return results, summary
    
    def _generate_overall_feedback(self, correct_frames, all_results):
        """Generate overall feedback based on analysis"""
        if not all_results:
            return "Unable to analyze exercise form. Please ensure proper lighting and full body visibility."
        
        accuracy = len(correct_frames) / len(all_results)
        
        if accuracy >= 0.8:
            return "Excellent form consistency! Keep up the great work."
        elif accuracy >= 0.6:
            return "Good form overall, but watch out for some inconsistencies."
        elif accuracy >= 0.4:
            return "Form needs improvement. Focus on correcting the identified errors."
        else:
            return "Poor form detected consistently. Please review proper technique."
    
    def image_to_base64(self, image):
        """Convert OpenCV image to base64 string"""
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64}"

# Initialize analyzer
analyzer = ExerciseFormAnalyzer()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("form_checker.html", {"request": request})

@app.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...), exercise_type: str = Form("auto")):
    """Analyze uploaded image for exercise form - returns correct/incorrect"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse({"error": "Could not decode image"}, status_code=400)
        
        # Analyze image
        analysis = analyzer.analyze_image(image, exercise_type)
        
        # Convert images to base64
        original_img_base64 = analyzer.image_to_base64(image)
        result_img_base64 = analyzer.image_to_base64(analysis["annotated_image"])
        
        return JSONResponse({
            "success": True,
            "exercise_type": analysis["exercise_type"],
            "is_correct": analysis["is_correct"],
            "confidence": float(analysis["confidence"]),
            "form_score": float(analysis["form_score"]),
            "errors": analysis["errors"],
            "corrections": analysis["corrections"],
            "feedback": analysis["feedback"],
            "pose_detected": analysis["pose_detected"],
            "original_image": original_img_base64,
            "result_image": result_img_base64
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...), max_frames: int = Form(30), exercise_type: str = Form("auto")):
    """Analyze uploaded video for exercise form"""
    try:
        if not file.content_type.startswith('video/'):
            return JSONResponse({"error": "Please upload a video file"}, status_code=400)
        
        # Save video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_video_path = tmp_file.name
        
        try:
            # Analyze video
            results, summary = analyzer.analyze_video(tmp_video_path, max_frames, exercise_type)
            
            if results is None:
                return JSONResponse({"error": summary}, status_code=400)
            
            return JSONResponse({
                "success": True,
                "filename": file.filename,
                "exercise_type": exercise_type,
                "summary": summary,
                "results": results,
                "total_frames": len(results)
            })
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_video_path):
                os.unlink(tmp_video_path)
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/analyze_webcam")
async def analyze_webcam(image_data: str = Form(...), exercise_type: str = Form("auto")):
    """Analyze webcam image for exercise form"""
    try:
        header, encoded = image_data.split(',', 1)
        data = base64.b64decode(encoded)
        nparr = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse({"error": "Could not decode image"}, status_code=400)
        
        # Analyze image
        analysis = analyzer.analyze_image(image, exercise_type)
        
        result_img_base64 = analyzer.image_to_base64(analysis["annotated_image"])
        
        return JSONResponse({
            "success": True,
            "exercise_type": analysis["exercise_type"],
            "is_correct": analysis["is_correct"],
            "confidence": float(analysis["confidence"]),
            "form_score": float(analysis["form_score"]),
            "errors": analysis["errors"],
            "corrections": analysis["corrections"],
            "feedback": analysis["feedback"],
            "pose_detected": analysis["pose_detected"],
            "result_image": result_img_base64
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/health")
async def health_check():
    return JSONResponse({
        "status": "healthy",
        "service": "Exercise Form Correction API",
        "supported_exercises": ["plank", "squat", "push_up", "bicep_curl", "lunge"],
        "analysis_type": "correct/incorrect form detection"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
