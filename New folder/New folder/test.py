import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

class ExerciseTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
            model_complexity=2,
            smooth_landmarks=True
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.counter = 0
        self.stage = None
        self.exercise_state = "READY"
        self.exercise_type = "SQUAT"
        self.last_hip_y = 0
        self.movement_threshold = 0.05

        self.colors = {
            'CORRECT': (0, 255, 0),
            'INCORRECT': (0, 0, 255),
            'READY': (255, 255, 0),
            'TEXT': (0, 0, 0),
            'SKELETON': (0, 255, 0)
        }

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle

    def analyze_situp(self, coords):
        torso_angle = self.calculate_angle(coords['right_shoulder'], 
                                         coords['right_hip'],
                                         coords['right_knee'])
        
        leg_angle = self.calculate_angle(coords['right_knee'],
                                       coords['right_hip'],
                                       coords['right_ankle'])
        
        vertical_angle = self.calculate_angle(coords['right_shoulder'],
                                            coords['right_hip'],
                                            (coords['right_hip'][0], coords['right_hip'][1] + 100))
        
        current_hip_y = coords['right_hip'][1]
        hip_movement = abs(current_hip_y - self.last_hip_y)
        self.last_hip_y = current_hip_y

        if torso_angle < 60 and vertical_angle < 60:  # ผ่อนปรนให้ลำตัวไม่ต้องตรงมาก
            if self.stage == 'down' and hip_movement > self.movement_threshold:
                if 60 <= leg_angle <= 120:  # ขาอยู่ในตำแหน่งที่ถูกต้อง
                    self.counter += 1  # เพิ่มตัวนับเมื่อท่าถูกต้อง
                    self.exercise_state = 'CORRECT'
                    self.stage = 'up'
                else:
                    self.exercise_state = 'INCORRECT'
        elif torso_angle > 90 and vertical_angle > 90:  # กลับไปที่พื้น
            if self.stage == 'up':
                self.stage = 'down'
            self.exercise_state = 'READY'
        else:
            self.exercise_state = 'INCORRECT'
        
        return {
            'torso_angle': torso_angle,
            'leg_angle': leg_angle,
            'vertical_angle': vertical_angle,
            'movement': hip_movement
        }

    def process_frame(self, frame):
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=self.colors['SKELETON'], thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=self.colors['SKELETON'], thickness=2)
            )
            
            coords = self.get_coordinates(results.pose_landmarks.landmark, width, height)
            
            if self.exercise_type == "SQUAT":
                angles = self.analyze_squat(coords)
            else:  # SITUP
                angles = self.analyze_situp(coords)
            
            self.draw_feedback(frame, angles)
        
        return frame
