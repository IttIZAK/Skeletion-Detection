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

    def get_coordinates(self, landmarks, width, height):
        return {
            'nose': (int(landmarks[self.mp_pose.PoseLandmark.NOSE.value].x * width),
                    int(landmarks[self.mp_pose.PoseLandmark.NOSE.value].y * height)),
            'left_shoulder': (int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width),
                            int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height)),
            'right_shoulder': (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width),
                             int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height)),
            'left_elbow': (int(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width),
                          int(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height)),
            'right_elbow': (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * width),
                           int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * height)),
            'left_wrist': (int(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x * width),
                          int(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y * height)),
            'right_wrist': (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x * width),
                           int(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y * height)),
            'left_hip': (int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * width),
                        int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * height)),
            'right_hip': (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * width),
                         int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * height)),
            'left_knee': (int(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x * width),
                         int(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y * height)),
            'right_knee': (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x * width),
                          int(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y * height)),
            'left_ankle': (int(landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width),
                          int(landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height)),
            'right_ankle': (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * width),
                           int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * height))
        }

    def analyze_situp_all_directions(self, coords):
        torso_angle_front = self.calculate_angle(coords['left_shoulder'], coords['left_hip'], coords['left_knee'])
        torso_angle_side = self.calculate_angle(coords['right_shoulder'], coords['right_hip'], coords['right_knee'])
        leg_angle = self.calculate_angle(coords['right_knee'], coords['right_hip'], coords['right_ankle'])
        vertical_angle = self.calculate_angle(coords['right_shoulder'], coords['right_hip'], 
                                              (coords['right_hip'][0], coords['right_hip'][1] + 100))

        # ตรวจสอบการเคลื่อนไหวของสะโพก
        current_hip_y = coords['right_hip'][1]

        if self.last_hip_y is None:  # กำหนดค่าเริ่มต้น
            self.last_hip_y = current_hip_y

        hip_movement = abs(current_hip_y - self.last_hip_y)
        self.last_hip_y = current_hip_y  # อัปเดตค่า

        if (torso_angle_front < 45 or torso_angle_side < 45) and vertical_angle < 45:
            if self.stage == 'down' and hip_movement > self.movement_threshold:
                if 70 <= leg_angle <= 110:
                    self.counter += 1
                    self.exercise_state = 'CORRECT'
                    self.stage = 'up'
                else:
                    self.exercise_state = 'INCORRECT'

        elif (torso_angle_front > 80 or torso_angle_side > 80) and vertical_angle > 80:
            if self.stage == 'up':
                self.stage = 'down'
            self.exercise_state = 'READY'

        else:
            self.exercise_state = 'INCORRECT'

        return {
            'torso_angle_front': torso_angle_front,
            'torso_angle_side': torso_angle_side,
            'leg_angle': leg_angle,
            'vertical_angle': vertical_angle,
            'movement': hip_movement
        }

    def analyze_squat(self, coords):
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
        # คำนวณมุมระหว่างไหล่-สะโพก-หัวเข่า (ลำตัว)
        torso_angle = self.calculate_angle(coords['right_shoulder'], 
                                         coords['right_hip'],
                                         coords['right_knee'])
        
        # คำนวณมุมระหว่างหัวเข่า-สะโพก-ข้อเท้า (ขา)
        leg_angle = self.calculate_angle(coords['right_knee'],
                                       coords['right_hip'],
                                       coords['right_ankle'])
        
        # คำนวณมุมระหว่างไหล่-สะโพก-แนวดิ่ง
        vertical_angle = self.calculate_angle(coords['right_shoulder'],
                                            coords['right_hip'],
                                            (coords['right_hip'][0], coords['right_hip'][1] + 100))
        
        # ติดตามการเคลื่อนที่ของสะโพก
        current_hip_y = coords['right_hip'][1]
        hip_movement = abs(current_hip_y - self.last_hip_y)
        self.last_hip_y = current_hip_y

        # ตรวจสอบท่า sit-up
        if torso_angle < 45 and vertical_angle < 45:  # ลำตัวยกขึ้น
            if self.stage == 'down' and hip_movement > self.movement_threshold:
                if 70 <= leg_angle <= 110:  # ขาอยู่ในตำแหน่งที่ถูกต้อง
                    self.counter += 1
                    self.exercise_state = 'CORRECT'
                    self.stage = 'up'
                else:
                    self.exercise_state = 'INCORRECT'
        elif torso_angle > 80 and vertical_angle > 80:  # ลำตัวนอนราบ
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

    def analyze_plank(self, coords):
        shoulder_angle = self.calculate_angle(coords['left_shoulder'], coords['left_hip'], coords['left_ankle'])
        hip_angle = self.calculate_angle(coords['right_shoulder'], coords['right_hip'], coords['right_ankle'])
        
        if 160 <= shoulder_angle <= 180 and 160 <= hip_angle <= 180:
            self.exercise_state = 'CORRECT'
        else:
            self.exercise_state = 'INCORRECT'
        
        return {'shoulder_angle': shoulder_angle, 'hip_angle': hip_angle}

    def draw_feedback(self, frame, angles):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        cv2.putText(frame, "Exercise Tracking", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.putText(frame, f'Exercise: {self.exercise_type}', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(frame, f'Reps: {self.counter}', (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        status_color = self.colors[self.exercise_state]
        cv2.putText(frame, f'Status: {self.exercise_state}', (10, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        x_pos = frame.shape[1] - 200
        y_pos = 30
        cv2.rectangle(overlay, (x_pos - 10, 0), (frame.shape[1], 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        for angle_name, angle_value in angles.items():
            cv2.putText(frame, f'{angle_name}: {angle_value:.1f}°', (x_pos, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos += 30
        
        menu_y = frame.shape[0] - 120
        cv2.rectangle(overlay, (0, menu_y), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        cv2.putText(frame, "Controls:", (10, menu_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "'1' - Switch to Squat | '2' - Switch to Situp | '3' - Switch to Plank |'q' - Quit", 
                    (10, menu_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.exercise_state == 'INCORRECT':
            error_y = menu_y - 40
            error_box_start = error_y - 30
            
            cv2.rectangle(overlay, (0, error_box_start), (frame.shape[1], error_y + 10), 
                         self.colors['INCORRECT'], -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            if self.exercise_type == "SQUAT":
                cv2.putText(frame, "INCORRECT FORM! Keep your back straight and knees aligned!", 
                           (10, error_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            elif self.exercise_type == "SITUP":
                cv2.putText(frame, "INCORRECT FORM! Keep your back straight and feet on the ground!", 
                           (10, error_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

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
        elif key == ord('1'):
            tracker.exercise_type = "SQUAT"
            tracker.counter = 0
        elif key == ord('2'):
            tracker.exercise_type = "SITUP"
            tracker.counter = 0
        elif key == ord('3'):
            tracker.exercise_type = "PLANK"
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()