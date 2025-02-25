import cv2
import threading
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com/",
    api_key="8BTxtDLw6Mr2YDRirXZr"
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้")
    exit()

result = None

def infer_image(frame):
    global result
    cv2.imwrite("temp.jpg", frame)  # บันทึกเฟรมเป็นไฟล์ชั่วคราว
    result = CLIENT.infer("temp.jpg", model_id="sticker-rs1ct/1")

frame_count = 0  # ตัวนับเฟรม

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ไม่สามารถอ่านเฟรมได้")
            break

        frame_count += 1

        # ส่งภาพไปยัง API ทุก 5 เฟรม
        if frame_count % 5 == 0:
            threading.Thread(target=infer_image, args=(frame.copy(),)).start()

        # วาดกรอบรอบ sticker ถ้ามีผลลัพธ์
        if result and 'predictions' in result:
            for pred in result['predictions']:
                x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
                confidence = pred.get('confidence', 0)
                label = pred.get('class', 'sticker')

                # คำนวณพิกัดกรอบ
                x1, y1 = x - w // 2, y - h // 2
                x2, y2 = x + w // 2, y + h // 2

                # วาดกรอบและป้ายชื่อ
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # แสดงเฟรม
        cv2.imshow('Webcam', frame)

        # กด 'q' เพื่อออก
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # ปิดกล้องและหน้าต่าง
    cap.release()
    cv2.destroyAllWindows()