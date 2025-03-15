import cv2
import mediapipe as mp
import numpy as np

class ExerciseTracker:
    def __init__(self, exercise_type="SQUAT"):
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
        self.exercise_type = exercise_type.upper()

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
            angle = 360 - angle
            
        return angle

    def analyze_squat(self, coords):
        knee_angle = self.calculate_angle(coords['right_hip'], coords['right_knee'], coords['right_ankle'])
        
        if knee_angle < 90:
            if self.stage == 'up':
                self.counter += 1
                self.stage = 'down'
                self.exercise_state = 'CORRECT'
        elif knee_angle > 140:
            self.stage = 'up'
            self.exercise_state = 'READY'
        else:
            self.exercise_state = 'INCORRECT'
        
        return {'knee_angle': knee_angle}

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
        
        if torso_angle < 60 and vertical_angle < 60:
            if self.stage == 'down':
                if 60 <= leg_angle <= 120:
                    self.counter += 1
                    self.exercise_state = 'CORRECT'
                    self.stage = 'up'
                else:
                    self.exercise_state = 'INCORRECT'
        elif torso_angle > 90 and vertical_angle > 90:
            if self.stage == 'up':
                self.stage = 'down'
            self.exercise_state = 'READY'
        else:
            self.exercise_state = 'INCORRECT'
        
        return {
            'torso_angle': torso_angle,
            'leg_angle': leg_angle,
            'vertical_angle': vertical_angle
        }

    def analyze_plank(self, coords):
        torso_angle = self.calculate_angle(coords['right_shoulder'], coords['right_hip'], coords['right_knee'])
        
        if torso_angle < 160:
            if self.stage == 'down':
                self.counter += 1
                self.stage = 'up'
                self.exercise_state = 'CORRECT'
        else:
            self.exercise_state = 'INCORRECT'
        
        return {'torso_angle': torso_angle}

    def analyze_lunge(self, coords):
        knee_angle = self.calculate_angle(coords['right_hip'], coords['right_knee'], coords['right_ankle'])
        
        if knee_angle < 90:
            if self.stage == 'up':
                self.counter += 1
                self.stage = 'down'
                self.exercise_state = 'CORRECT'
        elif knee_angle > 140:
            self.stage = 'up'
            self.exercise_state = 'READY'
        else:
            self.exercise_state = 'INCORRECT'
        
        return {'knee_angle': knee_angle}

    def get_coordinates(self, landmarks, width, height):
        try:
            return {
                'right_shoulder': (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width), 
                                   int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)),
                'right_hip': (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x * width), 
                              int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y * height)),
                'right_knee': (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].x * width), 
                               int(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].y * height)),
                'right_ankle': (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].x * width), 
                                int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y * height))
            }
        except:
            return {}

    def draw_feedback(self, frame):
        # No resolution scaling here
        cv2.putText(frame, f'Count: {self.counter}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['TEXT'], 2)
        cv2.putText(frame, f'State: {self.exercise_state}', (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors[self.exercise_state], 2)
        cv2.putText(frame, f'Exercise: {self.exercise_type}', (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['TEXT'], 2)
        cv2.putText(frame, f'Instructions: 1=Squat | 2=Situp | 3=Plank | 4=Lunge', (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['TEXT'], 2)

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
                self.analyze_squat(coords)
            elif self.exercise_type == "SITUP":
                self.analyze_situp(coords)
            elif self.exercise_type == "PLANK":
                self.analyze_plank(coords)
            elif self.exercise_type == "LUNGES":
                self.analyze_lunge(coords)
            
            self.draw_feedback(frame)
        
        return frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    # Set default resolution for the webcam (no scaling)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1270)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    tracker = ExerciseTracker("SQUAT")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("ไม่สามารถอ่านภาพจากกล้องได้")
            break

        frame = tracker.process_frame(frame)
        cv2.imshow("Exercise Tracker", frame)

        # Switch exercises based on key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            tracker.exercise_type = "SQUAT"
            tracker.counter = 0
        elif key == ord('2'):
            tracker.exercise_type = "SITUP"
            tracker.counter = 0
        elif key == ord('3'):
            tracker.exercise_type = "PLANK"
            tracker.counter = 0
        elif key == ord('4'):
            tracker.exercise_type = "LUNGES"
            tracker.counter = 0
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
