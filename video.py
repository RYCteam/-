import cv2
from ultralytics import YOLO
import os
import numpy as np
import requests
import time

# --- 1. โหลดโมเดลและวิดีโอ ---
model = YOLO('best.pt') 
# ระบุชื่อไฟล์วิดีโอของคุณที่นี่
video_path = 'เต้ลอย.mp4' 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: ไม่สามารถเปิดไฟล์วิดีโอ '{video_path}' ได้")
    exit()

# --- 2. เตรียมการสำหรับบันทึกวิดีโอ ---
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_folder = 'output_videos'
os.makedirs(output_folder, exist_ok=True)
base_name, ext = os.path.splitext(os.path.basename(video_path))
output_path = os.path.join(output_folder, f'{base_name}_annotated_final.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print(f"เริ่มประมวลผลวิดีโอ: {video_path}")

# --- 3. วนลูปประมวลผล ---
# กำหนดค่าพิกัด (ตัวอย่าง: กรุงเทพฯ)
latitude = 13.7563
longitude = 100.5018
# Discord Webhook URL
DISCORD_WEBHOOK_URL = 'https://discord.com/api/webhooks/1389905612409802843/1NSMixTKW7N6HMaugh-sopI8gy42RUGDJFLayX_BEzcciVzJgGQfEi_NUwa7J1w1tW7T'

def send_discord_alert(image_path):
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
        data = {
            'content': f'🚨 พบคนจมน้ำ!'
        }
        try:
            response = requests.post(DISCORD_WEBHOOK_URL, data=data, files=files)
            if response.status_code == 204 or response.status_code == 200:
                print('แจ้งเตือน Discord สำเร็จ')
            else:
                print(f'แจ้งเตือน Discord ไม่สำเร็จ: {response.status_code} {response.text}')
        except Exception as e:
            print(f'เกิดข้อผิดพลาดในการแจ้งเตือน Discord: {e}')

# --- ตัวแปรนับจำนวนเฟรม drowning ---
drowning_frame_count = 0
DROWNING_ALERT_THRESHOLD = 10
ESP_IP = "192.168.1.167"
alert_sent = False

def trigger_relay(ESP_IP):
    """
    ส่งคำสั่ง HTTP GET ไปยัง ESP8266 เพื่อเปิดรีเลย์
    """
    try:
        url = f"http://{ESP_IP}/relay/on"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print("คำสั่งเปิดรีเลย์ถูกส่งไปยังบอร์ดสำเร็จ!")
            time.sleep(1)  # หน่วงเวลา 10 วินาที
        else:
            print(f"ส่งคำสั่งไม่สำเร็จ, บอร์ดตอบกลับมาด้วย Status Code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"เกิดข้อผิดพลาด: ไม่สามารถเชื่อมต่อกับบอร์ดที่ IP {ESP_IP} ได้")
        print(f"รายละเอียด: {e}")

def trigger_relay_off(ESP_IP):
    """
    ส่งคำสั่ง HTTP GET ไปยัง ESP8266 เพื่อปิดรีเลย์
    """
    try:
        url = f"http://{ESP_IP}/relay/off"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print("สั่งปิดรีเลย์สำเร็จ (เริ่มต้น)!")
        else:
            print(f"สั่งปิดรีเลย์ไม่สำเร็จ, Status Code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"เกิดข้อผิดพลาด: ไม่สามารถเชื่อมต่อกับบอร์ดที่ IP {ESP_IP} ได้")
        print(f"รายละเอียด: {e}")

# เรียกใช้ก่อนเข้า while loop
trigger_relay_off(ESP_IP)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # --- 4. ให้โมเดลทำการตรวจจับด้วย iou สูงก่อน เพื่อให้ได้กรอบดิบทั้งหมด ---
    # เราจะไม่กรอง conf หรือ iou ในขั้นตอนนี้
    results = model(frame, verbose=False)

    # --- 5. สร้าง "เฟรมใหม่" และใช้ Non-Maximum Suppression (NMS) ---
    annotated_frame = frame.copy() 
    
    # ดึงข้อมูล boxes, scores, และ class_ids จาก results
    # แล้วใช้ฟังก์ชัน NMS ของ OpenCV เพื่อกรองกรอบที่ซ้อนกัน
    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

    # ใช้ Non-Maximum Suppression เพื่อเลือกกรอบที่ดีที่สุด
    # score_threshold คือค่าความมั่นใจขั้นต่ำที่จะพิจารณา
    # nms_threshold คือค่า iou ที่เราคุยกัน
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.3, nms_threshold=0.5)

    if len(indices) > 0:
        found_drowning = False
        for i in indices.flatten():
            x1, y1, x2, y2 = map(int, boxes[i])
            confidence = scores[i]
            class_name = model.names[class_ids[i]]
            
            # กำหนดสีตามคลาส
            color = (0, 255, 0) if class_name == 'swimming' else (0, 0, 255)
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, f'{class_name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            # --- แจ้งเตือน Discord ถ้าเจอ drowning ---
            if class_name == 'drowning':
                found_drowning = True

    if found_drowning:
        drowning_frame_count += 1
    else:
        drowning_frame_count = 0
        alert_sent = False

    if drowning_frame_count >= DROWNING_ALERT_THRESHOLD and not alert_sent:
        # แจ้งเตือนและสั่งเปิดรีเลย์ที่นี่เท่านั้น!
        alert_img_path = os.path.join(output_folder, f'drowning_alert_{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f}.jpg')
        cv2.imwrite(alert_img_path, annotated_frame)
        send_discord_alert(alert_img_path)
        trigger_relay(ESP_IP)
        alert_sent = True

    # บันทึกเฟรมที่วาดแล้ว
    out.write(annotated_frame)

    # แสดงผลสด
    h, w, _ = annotated_frame.shape
    scale = 960 / w
    small_frame = cv2.resize(annotated_frame, (int(w*scale), int(h*scale)))
    cv2.imshow("AI Demo (Final Version) - Press 'q' to stop & save", small_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- 6. สิ้นสุดการทำงาน ---
cap.release()
out.release()
cv2.destroyAllWindows()
    
print(f"🎉 วิดีโอ Demo บันทึกสำเร็จแล้วที่ไฟล์: {output_path}")

