import cv2
import mediapipe as mp
import csv
import os

class DatasetCreator:
    def __init__(self, output_file="exercise_dataset.csv"):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
            model_complexity=2
        )
        self.output_file = output_file
        self.fields = ["label"] + [f"landmark_{i}_{axis}" for i in range(33) for axis in ['x', 'y', 'z', 'visibility']]
        
        # สร้างไฟล์ CSV และเพิ่ม header ถ้ายังไม่มีไฟล์
        if not os.path.exists(self.output_file):
            with open(self.output_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(self.fields)

    def collect_data(self, frame, label):
        """เก็บข้อมูลตำแหน่ง Landmark พร้อม Label"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            row = [label]
            for lm in results.pose_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z, lm.visibility])
            
            with open(self.output_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(row)

    def __del__(self):
        self.pose.close()

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    dataset_creator = DatasetCreator()

    print("Press 's' to save 'Squat', 'u' for 'Sit-up', 'i' for 'Idle', and 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame")
            break

        label = None
        cv2.putText(frame, "Press 's' for Squat, 'u' for Sit-up, 'i' for Idle", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Dataset Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            label = "Squat"
        elif key == ord('u'):
            label = "Sit-up"
        elif key == ord('i'):
            label = "Idle"

        if label:
            dataset_creator.collect_data(frame, label)
            print(f"Saved data for label: {label}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
