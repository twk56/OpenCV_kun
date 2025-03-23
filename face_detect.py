import cv2 as cv
import sys

sys.stdout.reconfigure(encoding='utf-8')

img = cv.imread('xx.jpg')

img = cv.resize(img, (1024, 768))

alpha = 1.5 
beta = 50
img = cv.convertScaleAbs(img, alpha=alpha, beta=beta)

gray_scale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_smooth = cv.bilateralFilter(gray_scale, 9, 75, 75)

face_model = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

faces = face_model.detectMultiScale(
    img_smooth, 
    scaleFactor=1.02, 
    minNeighbors=4,
    minSize=(30, 30)
)

print(f"ตรวจพบใบหน้าทั้งหมด: {len(faces)} ใบหน้า")

for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.resizeWindow('image', 1024, 768)

cv.imshow('image', img)

cv.waitKey(0)

cv.destroyAllWindows()
