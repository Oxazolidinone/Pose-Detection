import numpy as np
import pickle
import os
from typing import Dict, List, Tuple
import mediapipe as mp

mp_pose = mp.solutions.pose

class ExerciseDetector:
    """Exercise form analysis using MediaPipe pose landmarks - focuses on correct/incorrect form detection"""
    
    def __init__(self):
        self.pose_landmarks = mp_pose.PoseLandmark
        self.supported_exercises = ["plank", "squat", "push_up", "lunge", "bicep_curl"]
        
    def analyze_exercise_form(self, pose_results, exercise_type: str) -> Dict:
        """Main method to analyze exercise form - returns correct/incorrect with feedback"""
        if not pose_results.pose_landmarks:
            return {
                "is_correct": False,
                "confidence": 0.0,
                "score": 0.0,
                "errors": ["No pose detected"],
                "feedback": "Please ensure your full body is visible in the image",
                "corrections": ["Position yourself fully in frame", "Ensure good lighting"]
            }
        
        # Analyze based on exercise type
        if exercise_type == "plank":
            return self.analyze_plank_form(pose_results)
        elif exercise_type == "squat":
            return self.analyze_squat_form(pose_results)
        elif exercise_type == "push_up":
            return self.analyze_pushup_form(pose_results)
        elif exercise_type == "lunge":
            return self.analyze_lunge_form(pose_results)
        elif exercise_type == "bicep_curl":
            return self.analyze_bicep_curl_form(pose_results)
        else:
            return self.analyze_general_form(pose_results)
    
    def analyze_general_form(self, pose_results) -> Dict:
        """General pose analysis for unknown exercises"""
        landmarks = pose_results.pose_landmarks.landmark
        
        # Basic posture checks
        errors = []
        corrections = []
        score = 1.0
        
        # Check if person is standing upright
        left_shoulder = landmarks[self.pose_landmarks.LEFT_SHOULDER]
        right_shoulder = landmarks[self.pose_landmarks.RIGHT_SHOULDER]
        left_hip = landmarks[self.pose_landmarks.LEFT_HIP]
        right_hip = landmarks[self.pose_landmarks.RIGHT_HIP]
        
        # Check shoulder alignment
        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        if shoulder_diff > 0.1:
            errors.append("Shoulders not aligned")
            corrections.append("Keep shoulders level and aligned")
            score -= 0.2
        
        # Check overall posture
        avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        avg_hip_y = (left_hip.y + right_hip.y) / 2
        
        if avg_shoulder_y > avg_hip_y:
            errors.append("Poor posture detected")
            corrections.append("Stand up straight, shoulders back")
            score -= 0.3
        
        is_correct = score >= 0.7
        feedback = "Good posture!" if is_correct else "Focus on proper alignment and posture"
        
        return {
            "is_correct": is_correct,
            "confidence": 0.8 if is_correct else 0.6,
            "score": max(0.0, score),
            "errors": errors,
            "feedback": feedback,
            "corrections": corrections
        }
    
    def analyze_plank_form(self, pose_results) -> Dict:
        """Analyze plank form - return correct/incorrect with specific feedback"""
        landmarks = pose_results.pose_landmarks.landmark
        
        # Get key landmarks
        nose = landmarks[self.pose_landmarks.NOSE]
        left_shoulder = landmarks[self.pose_landmarks.LEFT_SHOULDER]
        right_shoulder = landmarks[self.pose_landmarks.RIGHT_SHOULDER]
        left_hip = landmarks[self.pose_landmarks.LEFT_HIP]
        right_hip = landmarks[self.pose_landmarks.RIGHT_HIP]
        left_ankle = landmarks[self.pose_landmarks.LEFT_ANKLE]
        right_ankle = landmarks[self.pose_landmarks.RIGHT_ANKLE]
        
        errors = []
        corrections = []
        score = 1.0
        
        # Check spine alignment (nose to hip should be roughly straight)
        spine_angle = self.calculate_angle(
            [nose.x, nose.y],
            [(left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2],
            [(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2]
        )
        
        if spine_angle < 160 or spine_angle > 200:
            errors.append("Spine not aligned")
            corrections.append("Keep your back straight, head in neutral position")
            score -= 0.3
        
        # Check hip position
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_y = (left_hip.y + right_hip.y) / 2
        
        if hip_y > shoulder_y + 0.1:
            errors.append("Hips too high")
            corrections.append("Lower your hips to align with shoulders")
            score -= 0.3
        elif hip_y < shoulder_y - 0.1:
            errors.append("Hips too low")
            corrections.append("Raise your hips to align with shoulders")
            score -= 0.3
        
        # Check body straightness
        body_angle = self.calculate_angle(
            [(left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2],
            [(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2],
            [(left_ankle.x + right_ankle.x) / 2, (left_ankle.y + right_ankle.y) / 2]
        )
        
        if body_angle < 160 or body_angle > 200:
            errors.append("Body not straight")
            corrections.append("Keep your body in a straight line from head to heels")
            score -= 0.2
        
        is_correct = score >= 0.7
        feedback = "Perfect plank form!" if is_correct else "Plank form needs improvement"
        
        return {
            "is_correct": is_correct,
            "confidence": 0.9,
            "score": max(0.0, score),
            "errors": errors,
            "feedback": feedback,
            "corrections": corrections
        }
    
    def analyze_squat_form(self, pose_results) -> Dict:
        """Analyze squat form - return correct/incorrect with specific feedback"""
        landmarks = pose_results.pose_landmarks.landmark
        
        left_hip = landmarks[self.pose_landmarks.LEFT_HIP]
        right_hip = landmarks[self.pose_landmarks.RIGHT_HIP]
        left_knee = landmarks[self.pose_landmarks.LEFT_KNEE]
        right_knee = landmarks[self.pose_landmarks.RIGHT_KNEE]
        left_ankle = landmarks[self.pose_landmarks.LEFT_ANKLE]
        right_ankle = landmarks[self.pose_landmarks.RIGHT_ANKLE]
        
        errors = []
        corrections = []
        score = 1.0
        
        # Check knee-hip alignment
        left_knee_angle = self.calculate_angle(
            [left_hip.x, left_hip.y],
            [left_knee.x, left_knee.y],
            [left_ankle.x, left_ankle.y]
        )
        
        if left_knee_angle > 160:  # Not deep enough
            errors.append("Squat not deep enough")
            corrections.append("Squat deeper - aim for thighs parallel to ground")
            score -= 0.4
        elif left_knee_angle < 70:  # Too deep
            errors.append("Squatting too deep")
            corrections.append("Stop at 90 degrees, don't go too low")
            score -= 0.2
        
        # Check knee tracking (knees shouldn't cave inward)
        knee_distance = abs(left_knee.x - right_knee.x)
        ankle_distance = abs(left_ankle.x - right_ankle.x)
        
        if knee_distance < ankle_distance * 0.7:
            errors.append("Knees caving inward")
            corrections.append("Push knees out, keep them aligned with toes")
            score -= 0.4
        
        # Check weight distribution
        if abs(left_ankle.x - right_ankle.x) > 0.4:
            errors.append("Feet too wide apart")
            corrections.append("Keep feet shoulder-width apart")
            score -= 0.2
        
        is_correct = score >= 0.7
        feedback = "Excellent squat form!" if is_correct else "Squat form needs improvement"
        
        return {
            "is_correct": is_correct,
            "confidence": 0.85,
            "score": max(0.0, score),
            "errors": errors,
            "feedback": feedback,
            "corrections": corrections
        }
    
    def analyze_pushup_form(self, pose_results) -> Dict:
        """Analyze push-up form - return correct/incorrect with specific feedback"""
        landmarks = pose_results.pose_landmarks.landmark
        
        left_shoulder = landmarks[self.pose_landmarks.LEFT_SHOULDER]
        right_shoulder = landmarks[self.pose_landmarks.RIGHT_SHOULDER]
        left_elbow = landmarks[self.pose_landmarks.LEFT_ELBOW]
        right_elbow = landmarks[self.pose_landmarks.RIGHT_ELBOW]
        left_wrist = landmarks[self.pose_landmarks.LEFT_WRIST]
        right_wrist = landmarks[self.pose_landmarks.RIGHT_WRIST]
        left_hip = landmarks[self.pose_landmarks.LEFT_HIP]
        right_hip = landmarks[self.pose_landmarks.RIGHT_HIP]
        
        errors = []
        corrections = []
        score = 1.0
        
        # Check arm angle
        left_arm_angle = self.calculate_angle(
            [left_shoulder.x, left_shoulder.y],
            [left_elbow.x, left_elbow.y],
            [left_wrist.x, left_wrist.y]
        )
        
        if left_arm_angle > 150:
            errors.append("Arms not bent enough")
            corrections.append("Lower your body more, bend arms to 90 degrees")
            score -= 0.3
        elif left_arm_angle < 70:
            errors.append("Going too low")
            corrections.append("Don't go too low, maintain control")
            score -= 0.2
        
        # Check body alignment
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_y = (left_hip.y + right_hip.y) / 2
        
        if abs(shoulder_y - hip_y) > 0.2:
            errors.append("Body not straight")
            corrections.append("Keep body in straight line from head to feet")
            score -= 0.3
        
        # Check hand position
        hand_distance = abs(left_wrist.x - right_wrist.x)
        shoulder_distance = abs(left_shoulder.x - right_shoulder.x)
        
        if hand_distance > shoulder_distance * 1.5:
            errors.append("Hands too wide")
            corrections.append("Place hands shoulder-width apart")
            score -= 0.2
        elif hand_distance < shoulder_distance * 0.8:
            errors.append("Hands too narrow")
            corrections.append("Widen hand placement to shoulder-width")
            score -= 0.2
        
        is_correct = score >= 0.7
        feedback = "Perfect push-up form!" if is_correct else "Push-up form needs improvement"
        
        return {
            "is_correct": is_correct,
            "confidence": 0.8,
            "score": max(0.0, score),
            "errors": errors,
            "feedback": feedback,
            "corrections": corrections
        }
    
    def analyze_bicep_curl_form(self, pose_results) -> Dict:
        """Analyze bicep curl form - return correct/incorrect with specific feedback"""
        landmarks = pose_results.pose_landmarks.landmark
        
        left_shoulder = landmarks[self.pose_landmarks.LEFT_SHOULDER]
        left_elbow = landmarks[self.pose_landmarks.LEFT_ELBOW]
        left_wrist = landmarks[self.pose_landmarks.LEFT_WRIST]
        right_shoulder = landmarks[self.pose_landmarks.RIGHT_SHOULDER]
        right_elbow = landmarks[self.pose_landmarks.RIGHT_ELBOW]
        right_wrist = landmarks[self.pose_landmarks.RIGHT_WRIST]
        
        errors = []
        corrections = []
        score = 1.0
        
        # Check elbow position (should stay close to body)
        left_elbow_shoulder_distance = abs(left_elbow.x - left_shoulder.x)
        right_elbow_shoulder_distance = abs(right_elbow.x - right_shoulder.x)
        
        if left_elbow_shoulder_distance > 0.2 or right_elbow_shoulder_distance > 0.2:
            errors.append("Elbows moving away from body")
            corrections.append("Keep elbows tucked close to your sides")
            score -= 0.4
        
        # Check arm angle
        left_arm_angle = self.calculate_angle(
            [left_shoulder.x, left_shoulder.y],
            [left_elbow.x, left_elbow.y],
            [left_wrist.x, left_wrist.y]
        )
        
        if left_arm_angle < 30:
            errors.append("Not lowering weight fully")
            corrections.append("Lower the weight completely, full range of motion")
            score -= 0.3
        elif left_arm_angle > 150:
            errors.append("Not curling high enough")
            corrections.append("Curl the weight higher, squeeze biceps at top")
            score -= 0.3
        
        # Check for swinging motion
        if abs(left_shoulder.y - right_shoulder.y) > 0.1:
            errors.append("Using momentum/swinging")
            corrections.append("Use controlled movement, no swinging")
            score -= 0.3
        
        is_correct = score >= 0.7
        feedback = "Great bicep curl form!" if is_correct else "Bicep curl form needs improvement"
        
        return {
            "is_correct": is_correct,
            "confidence": 0.75,
            "score": max(0.0, score),
            "errors": errors,
            "feedback": feedback,
            "corrections": corrections
        }
    
    def analyze_lunge_form(self, pose_results) -> Dict:
        """Analyze lunge form - return correct/incorrect with specific feedback"""
        landmarks = pose_results.pose_landmarks.landmark
        
        left_hip = landmarks[self.pose_landmarks.LEFT_HIP]
        right_hip = landmarks[self.pose_landmarks.RIGHT_HIP]
        left_knee = landmarks[self.pose_landmarks.LEFT_KNEE]
        right_knee = landmarks[self.pose_landmarks.RIGHT_KNEE]
        left_ankle = landmarks[self.pose_landmarks.LEFT_ANKLE]
        right_ankle = landmarks[self.pose_landmarks.RIGHT_ANKLE]
        
        errors = []
        corrections = []
        score = 1.0
        
        # Determine which leg is forward
        front_leg = "left" if left_knee.y > right_knee.y else "right"
        
        if front_leg == "left":
            front_knee = left_knee
            front_hip = left_hip
            front_ankle = left_ankle
            back_knee = right_knee
        else:
            front_knee = right_knee
            front_hip = right_hip
            front_ankle = right_ankle
            back_knee = left_knee
        
        # Check front knee angle
        front_knee_angle = self.calculate_angle(
            [front_hip.x, front_hip.y],
            [front_knee.x, front_knee.y],
            [front_ankle.x, front_ankle.y]
        )
        
        if front_knee_angle > 120:
            errors.append("Not lunging deep enough")
            corrections.append("Lunge deeper - aim for 90 degrees at front knee")
            score -= 0.4
        elif front_knee_angle < 70:
            errors.append("Lunging too deep")
            corrections.append("Don't go too low, maintain balance")
            score -= 0.2
        
        # Check if front knee is over ankle
        if front_knee.x > front_ankle.x + 0.1:
            errors.append("Front knee over toes")
            corrections.append("Keep front knee behind toes, shift weight back")
            score -= 0.4
        
        # Check back knee position
        if back_knee.y < front_knee.y - 0.3:
            errors.append("Back knee not low enough")
            corrections.append("Lower your back knee closer to the ground")
            score -= 0.3
        
        # Check torso alignment
        avg_hip_y = (left_hip.y + right_hip.y) / 2
        if abs(left_hip.y - right_hip.y) > 0.15:
            errors.append("Torso leaning too much")
            corrections.append("Keep torso upright, don't lean forward")
            score -= 0.2
        
        is_correct = score >= 0.7
        feedback = "Excellent lunge form!" if is_correct else "Lunge form needs improvement"
        
        return {
            "is_correct": is_correct,
            "confidence": 0.8,
            "score": max(0.0, score),
            "errors": errors,
            "feedback": feedback,
            "corrections": corrections
        }
    
    def calculate_angle(self, point1: List[float], point2: List[float], point3: List[float]) -> float:
        """Calculate angle between three points"""
        # Convert to numpy arrays
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate angle
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return float(np.degrees(angle))  # Convert to Python float
    
    def calculate_distance(self, point1: List[float], point2: List[float]) -> float:
        """Calculate distance between two points"""
        return float(np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2))  # Convert to Python float
