import cv2 as cv
import sys

# ตั้ง encoding ของ stdout เป็น utf-8 เพื่อรองรับภาษาไทย
sys.stdout.reconfigure(encoding='utf-8')

# อ่านภาพจากไฟล์
img = cv.imread('xx.jpg')

# ปรับขนาดภาพให้เหมาะสม (1024x768)
img = cv.resize(img, (1024, 768))

# ปรับความคมชัดและความสว่างของภาพ
alpha = 1.5  # ค่าความคมชัด
beta = 50    # ค่าความสว่าง
img = cv.convertScaleAbs(img, alpha=alpha, beta=beta)

# ใช้ Bilateral Filter เพื่อทำให้ภาพเรียบเนียนขณะรักษาขอบ
gray_scale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_smooth = cv.bilateralFilter(gray_scale, 9, 75, 75)

# โหลดโมเดลตรวจจับใบหน้า (เลือกโมเดลที่ละเอียดขึ้น)
face_model = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

# ตรวจจับใบหน้าในภาพ (ปรับพารามิเตอร์เพื่อความแม่นยำ)
faces = face_model.detectMultiScale(
    img_smooth, 
    scaleFactor=1.02,   # ลดขนาดทีละน้อยเพื่อเพิ่มความละเอียด
    minNeighbors=4,    # เพิ่มความแม่นยำด้วยการเพิ่มจำนวนกล่องที่ต้องตรวจพบ
    minSize=(30, 30)   # ขนาดขั้นต่ำของใบหน้าที่ตรวจจับ
)

# พิมพ์จำนวนใบหน้าที่ตรวจพบ
print(f"ตรวจพบใบหน้าทั้งหมด: {len(faces)} ใบหน้า")

# วาดกรอบรอบใบหน้าที่ตรวจพบ
for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)  # สี BGR: เหลือง

# ทำให้หน้าต่างแสดงผลสามารถปรับขนาดได้
cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.resizeWindow('image', 1024, 768)  # ตั้งขนาดหน้าต่างแสดงภาพ

# แสดงภาพที่ทำให้เรียบเนียนและตรวจจับใบหน้า
cv.imshow('image', img)

# รอการกดปุ่มเพื่อปิดหน้าต่าง
cv.waitKey(0)

# ปิดหน้าต่างทั้งหมด
cv.destroyAllWindows()
