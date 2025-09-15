import numpy as np
import cv2
# from mediapipe.solutions.pose import PoseLandmark
from mediapipe import solutions
PoseLandmark = solutions.pose.PoseLandmark #(Newly added)

class PostureClassifier:
    def __init__(self):
        self.prev_state = {}

    def classify(self, landmarks, visibility_threshold=0.5):
        if landmarks is None:
            return "Unknown"

        def get_point(lm):
            return np.array([lm.x, lm.y]) if lm.visibility > visibility_threshold else None

        keypoints = {
            "left_shoulder": get_point(landmarks.landmark[PoseLandmark.LEFT_SHOULDER]),
            "right_shoulder": get_point(landmarks.landmark[PoseLandmark.RIGHT_SHOULDER]),
            "left_hip": get_point(landmarks.landmark[PoseLandmark.LEFT_HIP]),
            "right_hip": get_point(landmarks.landmark[PoseLandmark.RIGHT_HIP]),
            "left_knee": get_point(landmarks.landmark[PoseLandmark.LEFT_KNEE]),
            "right_knee": get_point(landmarks.landmark[PoseLandmark.RIGHT_KNEE]),
        }

        if any(v is None for v in keypoints.values()):
            return "Uncertain"

        shoulders_y = np.mean([keypoints["left_shoulder"], keypoints["right_shoulder"]])
        hips_y = np.mean([keypoints["left_hip"], keypoints["right_hip"]])
        knees_y = np.mean([keypoints["left_knee"], keypoints["right_knee"]])

        shoulder_hip_dist = hips_y - shoulders_y
        hip_knee_dist = knees_y - hips_y
        total_height = knees_y - shoulders_y

        if total_height < 0.15:
            return "Lying"
        elif shoulder_hip_dist < 0.1 and hip_knee_dist > 0.1:
            return "Sitting"
        elif shoulder_hip_dist > 0.1 and hip_knee_dist > 0.1:
            return "Standing"
        else:
            return "Unknown"

class DemographicsDetector:
    def __init__(self):
        # Load pre-trained models
        self.age_net = cv2.dnn.readNetFromCaffe('models/age_deploy.prototxt', 'models/age_net.caffemodel')
        self.gender_net = cv2.dnn.readNetFromCaffe('models/gender_deploy.prototxt', 'models/gender_net.caffemodel')

        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.gender_list = ['Male', 'Female']

    def detect_age_gender(self, frame):
        if frame is None or frame.size == 0:
            return "N/A", "N/A"
        # Prepare input for the model
        blob = cv2.dnn.blobFromImage(frame, 1, (227, 227), (78.0, 87.0, 114.0), swapRB=False)
        
        # Predict Gender
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = self.gender_list[gender_preds.argmax()]

        # Predict Age
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age = self.age_list[age_preds.argmax()]

        return age, gender
