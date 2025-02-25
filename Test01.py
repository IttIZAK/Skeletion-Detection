import cv2
import mediapipe as mp
import numpy as np
import speech_recognition as sr
import threading

class ExerciseTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose()

        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()

        self.exercise_type = None  
        self.counter = 0  
        self.stage = None  

        self.command = None
        self.listening = False

    def get_coordinates(self, landmarks, width, height):
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

    def process_frame(self, frame):
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
            
            coords = self.get_coordinates(results.pose_landmarks.landmark, width, height)

            if self.exercise_type == "SQUAT":
                self.analyze_squat(coords)
            elif self.exercise_type == "SITUP":
                self.analyze_situp(coords)
            elif self.exercise_type == "PLANK":
                self.analyze_plank(coords)

        if self.exercise_type:
            cv2.putText(frame, f'Exercise: {self.exercise_type}', (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Count: {self.counter}', (30, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return frame

    def recognize_speech(self):
        """ การรับคำสั่งเสียงในแยก thread """
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = self.recognizer.listen(source, timeout=5)
                self.command = self.recognizer.recognize_google(audio).lower()
                print(f"Recognized command: {self.command}")
                self.process_command(self.command)
            except sr.UnknownValueError:
                print("Could not understand the command.")
            except sr.RequestError:
                print("Could not request results, please check your internet connection.")
            except sr.WaitTimeoutError:
                print("Listening timeout reached.")

    def start_listening(self):
        """ ฟังก์ชันในการเริ่มฟังคำสั่งเสียงใน thread แยก """
        while True:
            self.recognize_speech()

    def process_command(self, command):
        if "squat" in command or "squats" in command or "squad" in command:
            self.exercise_type = "SQUAT"
            self.counter = 0
            print("Switched to SQUAT mode")
        elif "sit-up" in command or "situp" in command or "sit up" in command or "sit down" in command or "set up" in command :
            self.exercise_type = "SITUP"
            self.counter = 0
            print("Switched to SITUP mode")
        elif "plank" in command or "play" in command or "playing" in command:
            self.exercise_type = "PLANK"
            print("Switched to PLANK mode")
        else:
            print("Unknown exercise command.")

    def analyze_squat(self, coords):
        hip_y = coords['left_hip'][1]
        knee_y = coords['left_knee'][1]

        if hip_y > knee_y and self.stage != "down":
            self.stage = "down"
        elif hip_y < knee_y and self.stage == "down":
            self.stage = "up"
            self.counter += 1
            print(f"Squat Count: {self.counter}")

    def analyze_situp(self, coords):
        shoulder_y = coords['left_shoulder'][1]
        hip_y = coords['left_hip'][1]

        if shoulder_y > hip_y and self.stage != "down":
            self.stage = "down"
        elif shoulder_y < hip_y and self.stage == "down":
            self.stage = "up"
            self.counter += 1
            print(f"Sit-up Count: {self.counter}")

    def analyze_plank(self, coords):
        print("Holding plank...")

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    tracker = ExerciseTracker()

    # เริ่มรับคำสั่งเสียงใน thread แยก
    speech_thread = threading.Thread(target=tracker.start_listening, daemon=True)
    speech_thread.start()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("ไม่สามารถอ่านภาพจากกล้องได้")
            break
        
        frame = tracker.process_frame(frame)

        if tracker.command:
            cv2.putText(frame, f"Command: {tracker.command}", (30, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Exercise Tracking', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('v'):  # กด 'v' เพื่อเปิดการรับคำสั่งเสียง
            tracker.command = None  # ล้างคำสั่งเสียงที่ได้รับ
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
