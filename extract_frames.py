import cv2
import os
import time

video_path = 'เต้ว่ายน้ำ3.mp4' # ชื่อนี้น่าจะผิด ให้แก้เป็นชื่อวิดีโอของคุณ
output_folder = 'Correction_Images'
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    cv2.imshow("Press 's' to save frame, 'q' to quit", frame)

    key = cv2.waitKey(30) & 0xFF # waitKey(30) ทำให้วิดีโอเล่นช้าลงนิดหน่อย

    if key == ord('s'):
        # สร้างชื่อไฟล์ที่ไม่ซ้ำกัน
        timestamp = int(time.time() * 1000)
        image_name = f'problem_frame_{timestamp}.jpg'
        save_path = os.path.join(output_folder, image_name)
        cv2.imwrite(save_path, frame)
        print(f"Saved: {save_path}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()