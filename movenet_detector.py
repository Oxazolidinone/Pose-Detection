import tensorflow as tf
import numpy as np
import cv2
import os
import urllib.request

KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

MIN_CROP_KEYPOINT_SCORE = 0.2

class MoveNetDetector:
    def __init__(self, model_path="model.tflite"):
        self.model_path = model_path
        self.interpreter = None
        self.input_size = 256
        self._download_model()
        self._load_model()

    def _download_model(self):
        if not os.path.exists(self.model_path):
            print("Downloading MoveNet model...")
            model_url = "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite"
            urllib.request.urlretrieve(model_url, self.model_path)
            print("Model downloaded successfully!")

    def _load_model(self):
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

    def _movenet(self, input_image):
        input_image = tf.cast(input_image, dtype=tf.uint8)
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
        self.interpreter.invoke()
        keypoints_with_scores = self.interpreter.get_tensor(output_details[0]['index'])
        return keypoints_with_scores

    def _init_crop_region(self, image_height, image_width):
        if image_width > image_height:
            box_height = image_width / image_height
            box_width = 1.0
            y_min = (image_height / 2 - image_width / 2) / image_height
            x_min = 0.0
        else:
            box_height = 1.0
            box_width = image_height / image_width
            y_min = 0.0
            x_min = (image_width / 2 - image_height / 2) / image_width

        return {
            'y_min': y_min,
            'x_min': x_min,
            'y_max': y_min + box_height,
            'x_max': x_min + box_width,
            'height': box_height,
            'width': box_width
        }

    def _torso_visible(self, keypoints):
        return ((keypoints[0, 0, KEYPOINT_DICT['left_hip'], 2] > MIN_CROP_KEYPOINT_SCORE or
                 keypoints[0, 0, KEYPOINT_DICT['right_hip'], 2] > MIN_CROP_KEYPOINT_SCORE) and
                (keypoints[0, 0, KEYPOINT_DICT['left_shoulder'], 2] > MIN_CROP_KEYPOINT_SCORE or
                 keypoints[0, 0, KEYPOINT_DICT['right_shoulder'], 2] > MIN_CROP_KEYPOINT_SCORE))

    def _determine_torso_and_body_range(self, keypoints, target_keypoints, center_y, center_x):
        torso_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        max_torso_yrange = 0.0
        max_torso_xrange = 0.0
        
        for joint in torso_joints:
            dist_y = abs(center_y - target_keypoints[joint][0])
            dist_x = abs(center_x - target_keypoints[joint][1])
            if dist_y > max_torso_yrange:
                max_torso_yrange = dist_y
            if dist_x > max_torso_xrange:
                max_torso_xrange = dist_x

        max_body_yrange = 0.0
        max_body_xrange = 0.0
        
        for joint in KEYPOINT_DICT.keys():
            if keypoints[0, 0, KEYPOINT_DICT[joint], 2] < MIN_CROP_KEYPOINT_SCORE:
                continue
            dist_y = abs(center_y - target_keypoints[joint][0])
            dist_x = abs(center_x - target_keypoints[joint][1])
            if dist_y > max_body_yrange:
                max_body_yrange = dist_y
            if dist_x > max_body_xrange:
                max_body_xrange = dist_x

        return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]

    def _determine_crop_region(self, keypoints, image_height, image_width):
        target_keypoints = {}
        for joint in KEYPOINT_DICT.keys():
            target_keypoints[joint] = [
                keypoints[0, 0, KEYPOINT_DICT[joint], 0] * image_height,
                keypoints[0, 0, KEYPOINT_DICT[joint], 1] * image_width
            ]

        if self._torso_visible(keypoints):
            center_y = (target_keypoints['left_hip'][0] + target_keypoints['right_hip'][0]) / 2
            center_x = (target_keypoints['left_hip'][1] + target_keypoints['right_hip'][1]) / 2

            (max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange) = \
                self._determine_torso_and_body_range(keypoints, target_keypoints, center_y, center_x)

            crop_length_half = np.amax([
                max_torso_xrange * 1.9, max_torso_yrange * 1.9,
                max_body_yrange * 1.2, max_body_xrange * 1.2
            ])

            tmp = np.array([center_x, image_width - center_x, center_y, image_height - center_y])
            crop_length_half = np.amin([crop_length_half, np.amax(tmp)])

            crop_corner = [center_y - crop_length_half, center_x - crop_length_half]

            if crop_length_half > max(image_width, image_height) / 2:
                return self._init_crop_region(image_height, image_width)
            else:
                crop_length = crop_length_half * 2
                return {
                    'y_min': crop_corner[0] / image_height,
                    'x_min': crop_corner[1] / image_width,
                    'y_max': (crop_corner[0] + crop_length) / image_height,
                    'x_max': (crop_corner[1] + crop_length) / image_width,
                    'height': crop_length / image_height,
                    'width': crop_length / image_width
                }
        else:
            return self._init_crop_region(image_height, image_width)

    def _crop_and_resize(self, image, crop_region, crop_size):
        boxes = [[crop_region['y_min'], crop_region['x_min'],
                  crop_region['y_max'], crop_region['x_max']]]
        output_image = tf.image.crop_and_resize(
            image, box_indices=[0], boxes=boxes, crop_size=crop_size)
        return output_image

    def _run_inference(self, image, crop_region):
        image_height, image_width, _ = image.shape
        input_image = self._crop_and_resize(
            tf.expand_dims(image, axis=0), crop_region, crop_size=[self.input_size, self.input_size])
        keypoints_with_scores = self._movenet(input_image)
        
        for idx in range(17):
            keypoints_with_scores[0, 0, idx, 0] = (
                crop_region['y_min'] * image_height +
                crop_region['height'] * image_height *
                keypoints_with_scores[0, 0, idx, 0]) / image_height
            keypoints_with_scores[0, 0, idx, 1] = (
                crop_region['x_min'] * image_width +
                crop_region['width'] * image_width *
                keypoints_with_scores[0, 0, idx, 1]) / image_width
        return keypoints_with_scores

    def detect_pose(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not load image: {image}")
        
        height, width = image.shape[:2]
        crop_region = self._init_crop_region(height, width)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = tf.convert_to_tensor(image_rgb, dtype=tf.uint8)
        
        keypoints_with_scores = self._run_inference(image_tensor, crop_region)
        return keypoints_with_scores

    def extract_features(self, image):
        keypoints_with_scores = self.detect_pose(image)
        keypoints = keypoints_with_scores[0, 0, :, :]
        return keypoints.flatten()

    def draw_keypoints(self, image, keypoints_with_scores, confidence_threshold=0.1):
        keypoints = keypoints_with_scores[0, 0, :, :]
        image = image.copy()
        
        y, x, c = image.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
        
        for kp in shaped:
            ky, kx, kp_conf = kp
            if kp_conf > confidence_threshold:
                cv2.circle(image, (int(kx), int(ky)), 4, (0, 255, 0), -1)
        
        # Draw connections
        for edge_pair in SKELETON_CONNECTIONS:
            kp_a, kp_b = edge_pair
            if (shaped[kp_a][2] > confidence_threshold and 
                shaped[kp_b][2] > confidence_threshold):
                cv2.line(image, 
                        (int(shaped[kp_a][1]), int(shaped[kp_a][0])),
                        (int(shaped[kp_b][1]), int(shaped[kp_b][0])),
                        (0, 0, 255), 2)
        
        return image
