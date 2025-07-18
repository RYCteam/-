import cv2
from ultralytics import YOLO

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
