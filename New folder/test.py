import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
from datetime import datetime

class ExercisePoseTrainer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=2
        )
        
        # กำหนดโฟลเดอร์สำหรับเก็บข้อมูล
        self.data_dir = 'exercise_data'
        self.model_dir = 'trained_models'
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def extract_pose_features(self, landmarks):
        """สกัดคุณลักษณะจาก landmarks"""
        features = []
        for landmark in landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        return features

    def collect_training_data(self, exercise_name, num_frames=100):
        """เก็บข้อมูลสำหรับการเทรน"""
        cap = cv2.VideoCapture(0)
        collected_data = []
        frame_count = 0
        
        print(f"เริ่มเก็บข้อมูลท่า {exercise_name}")
        print("กด 'c' เพื่อเริ่มเก็บข้อมูล, 'q' เพื่อออก")
        
        collecting = False
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            # แสดงสถานะการเก็บข้อมูล
            status_text = "กำลังเก็บข้อมูล..." if collecting else "พร้อมเก็บข้อมูล (กด 'c')"
            cv2.putText(frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Frames: {frame_count}/{num_frames}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # ประมวลผลภาพ
            if collecting:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame_rgb)
                
                if results.pose_landmarks:
                    # สกัดคุณลักษณะและเก็บข้อมูล
                    features = self.extract_pose_features(results.pose_landmarks)
                    collected_data.append({
                        'exercise': exercise_name,
                        'features': features,
                        'timestamp': datetime.now().isoformat()
                    })
                    frame_count += 1
                    
                    # วาดโครงกระดูก
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            cv2.imshow('Data Collection', frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                collecting = True
            elif key == ord('q') or frame_count >= num_frames:
                break

        cap.release()
        cv2.destroyAllWindows()
        
        # บันทึกข้อมูล
        filename = f"{self.data_dir}/{exercise_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        np.savez(filename, data=collected_data)
        print(f"บันทึกข้อมูล {len(collected_data)} เฟรมไปที่ {filename}")
        
        return collected_data

    def prepare_dataset(self, data_files):
        """เตรียมข้อมูลสำหรับการเทรน"""
        X = []  # features
        y = []  # labels
        
        for file in data_files:
            data = np.load(file, allow_pickle=True)['data']
            for sample in data:
                X.append(sample['features'])
                y.append(sample['exercise'])
        
        return np.array(X), np.array(y)

    def train_model(self, X, y):
        """เทรนโมเดลสำหรับการจำแนกท่าทาง"""
        # แบ่งข้อมูลสำหรับเทรนและทดสอบ
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # สร้างและเทรนโมเดล
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # ประเมินผลโมเดล
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"Train accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        
        # บันทึกโมเดล
        model_path = f"{self.model_dir}/exercise_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        dump(model, model_path)
        print(f"บันทึกโมเดลไปที่ {model_path}")
        
        return model

    def validate_poses(self, model, exercise_name):
        """ทดสอบโมเดลแบบ real-time"""
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # สกัดคุณลักษณะและทำนาย
                features = self.extract_pose_features(results.pose_landmarks)
                prediction = model.predict([features])[0]
                confidence = model.predict_proba([features]).max()
                
                # วาดโครงกระดูกและแสดงผลการทำนาย
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                color = (0, 255, 0) if prediction == exercise_name else (0, 0, 255)
                cv2.putText(frame, f"Pose: {prediction}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow('Pose Validation', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    trainer = ExercisePoseTrainer()
    
    # ตัวอย่างการใช้งาน
    exercises = ['squat_correct', 'squat_incorrect', 'standing']
    
    # เก็บข้อมูลสำหรับแต่ละท่า
    for exercise in exercises:
        print(f"\nเก็บข้อมูลสำหรับท่า {exercise}")
        trainer.collect_training_data(exercise, num_frames=100)
    
    # เตรียมข้อมูลและเทรนโมเดล
    data_files = [f for f in os.listdir(trainer.data_dir) if f.endswith('.npz')]
    data_files = [os.path.join(trainer.data_dir, f) for f in data_files]
    
    X, y = trainer.prepare_dataset(data_files)
    model = trainer.train_model(X, y)
    
    # ทดสอบโมเดล
    print("\nทดสอบการตรวจจับท่า squat_correct")
    trainer.validate_poses(model, 'squat_correct')

if __name__ == "__main__":
    main()