import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

class ExerciseTracker:
    def __init__(self, dataset_file="exercise_dataset.csv"):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=2,
            smooth_landmarks=True
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # ตัวแปรสำหรับนับจำนวนครั้ง
        self.counter = 0
        self.stage = None  # up หรือ down
        self.exercise_state = "READY"  # READY, CORRECT, INCORRECT
        self.exercise_mode = "squat"  # squat หรือ sit-up
        
        # สีสำหรับแสดงผล
        self.colors = {
            'CORRECT': (0, 255, 0),    # เขียว
            'INCORRECT': (0, 0, 255),  # แดง
            'READY': (255, 255, 0),    # เหลือง
            'TEXT': (0, 0, 0),   # ขาว
            'SKELETON': (0, 255, 0)    # เขียว
        }

    def calculate_angle(self, a, b, c):
        """คำนวณมุมระหว่างจุด 3 จุด"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle

    def get_coordinates(self, landmarks, width, height):
        """แปลงค่า landmarks เป็นพิกัด x, y"""
        return {
            'nose': (int(landmarks[self.mp_pose.PoseLandmark.NOSE.value].x * width),
                    int(landmarks[self.mp_pose.PoseLandmark.NOSE.value].y * height)),
            'left_shoulder': (int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width),
                            int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height)),
            'right_shoulder': (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width),
                             int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height)),
            'left_hip': (int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * width),
                        int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * height)),
            'right_hip': (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * width),
                         int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * height)),
            'left_knee': (int(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x * width),
                         int(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y * height)),
            'right_knee': (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x * width),
                          int(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y * height))
        }

    def analyze_squat(self, coords):
        """วิเคราะห์ท่า squat"""
        left_knee_angle = self.calculate_angle(coords['left_hip'], coords['left_knee'], 
                                             (coords['left_knee'][0], coords['left_knee'][1] + 100))
        right_knee_angle = self.calculate_angle(coords['right_hip'], coords['right_knee'],
                                              (coords['right_knee'][0], coords['right_knee'][1] + 100))
        
        hip_angle = self.calculate_angle(coords['left_shoulder'], coords['left_hip'], coords['left_knee'])
        
        if left_knee_angle < 90 and right_knee_angle < 90:
            if hip_angle > 45:
                if self.stage == 'up':
                    self.counter += 1
                    self.stage = 'down'
                self.exercise_state = 'CORRECT'
            else:
                self.exercise_state = 'INCORRECT'
        elif left_knee_angle > 160 and right_knee_angle > 160:
            self.stage = 'up'
            self.exercise_state = 'READY'
        
        return {
            'left_knee_angle': left_knee_angle,
            'right_knee_angle': right_knee_angle,
            'hip_angle': hip_angle
        }

    def analyze_situp(self, coords):
        """วิเคราะห์ท่า sit-up"""
        shoulder_angle = self.calculate_angle(coords['left_hip'], coords['left_shoulder'], coords['nose'])
        
        if shoulder_angle < 50:  # ท่าก้มต่ำสุด
            if self.stage == 'up':
                self.counter += 1
                self.stage = 'down'
            self.exercise_state = 'CORRECT'
        elif shoulder_angle > 160:  # ท่าเริ่มต้น
            self.stage = 'up'
            self.exercise_state = 'READY'
        else:
            self.exercise_state = 'INCORRECT'
        
        return {
            'shoulder_angle': shoulder_angle
        }

    def draw_feedback(self, frame, angles):
        """แสดงผลข้อมูลและคำแนะนำ"""
        cv2.putText(frame, f'Reps: {self.counter}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['TEXT'], 2)
        cv2.putText(frame, f'State: {self.exercise_state}', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors[self.exercise_state], 2)
        y_pos = 110
        for angle_name, angle_value in angles.items():
            cv2.putText(frame, f'{angle_name}: {angle_value:.1f}', (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['TEXT'], 2)
            y_pos += 30

        # แสดงเมนูการเปลี่ยนท่า
        cv2.putText(frame, 'Press S for Squat, U for Sit-Up', (10, y_pos + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['TEXT'], 2)

    def process_frame(self, frame):
        """ประมวลผลเฟรมและวิเคราะห์ท่าทาง"""
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
            if self.exercise_mode == "squat":
                angles = self.analyze_squat(coords)
            elif self.exercise_mode == "situp":
                angles = self.analyze_situp(coords)
            
            self.draw_feedback(frame, angles)
        
        return frame

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    tracker = ExerciseTracker()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("ไม่สามารถอ่านภาพจากกล้องได้")
            break
        
        frame = tracker.process_frame(frame)
        cv2.imshow('Exercise Tracking', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            tracker.exercise_mode = "squat"
            tracker.counter = 0
        elif key == ord('u'):
            tracker.exercise_mode = "situp"
            tracker.counter = 0
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
