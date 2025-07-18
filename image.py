import cv2
from ultralytics import YOLO
import os
import requests

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

# --- 1. โหลดโมเดล 'best.pt' ของเรา ---
model = YOLO('bestt.pt') 

# --- 2. ระบุไฟล์ภาพนิ่งทดสอบ ---
image_path = 'คลอง2.jpg' # ผมเปลี่ยนชื่อไฟล์ให้ตรงกับที่คุณอัปโหลดมาล่าสุด
try:
    # อ่านไฟล์ภาพนิ่ง
    img = cv2.imread(image_path)
    
    # --- 3. ให้โมเดลทำการตรวจจับในภาพ (พร้อมปรับค่าความมั่นใจ) ---
    # การเพิ่ม conf=0.1 คือการบอกโมเดลว่า "ถึงแม้จะมั่นใจแค่ 10% ก็แสดงผลออกมา"
    results = model(img, conf=0.1)

    # --- 4. นำผลลัพธ์มาวาดลงบนภาพ ---
    annotated_img = results[0].plot()

    # --- ตรวจสอบว่ามี class 'drowning' หรือไม่ ---
    found_drowning = False
    for box in results[0].boxes:
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        if class_name == 'drowning':
            found_drowning = True
            break

    # --- ถ้าพบ drowning ให้บันทึกภาพและแจ้งเตือน Discord ---
    if found_drowning:
        output_folder = 'output_images'
        os.makedirs(output_folder, exist_ok=True)
        alert_img_path = os.path.join(output_folder, f'drowning_alert_{os.path.basename(image_path)}')
        cv2.imwrite(alert_img_path, annotated_img)
        send_discord_alert(alert_img_path)

    # --- 5. แสดงผลลัพธ์ ---
    cv2.imshow("AI Model Test - Press any key to exit", annotated_img)

    # รอให้มีการกดปุ่มใดๆ แล้วจึงปิดหน้าต่าง
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except FileNotFoundError:
    print(f"Error: ไม่พบไฟล์ '{image_path}'")
    print("กรุณาตรวจสอบว่ามีไฟล์ภาพอยู่ในโฟลเดอร์เดียวกับโค้ดหรือไม่")
except Exception as e:
    print(f"เกิดข้อผิดพลาด: {e}")
