import cv2
from ultralytics import YOLO
import os
import time

# MODEL SETUP
model = YOLO("./yolov8s.pt")

#results = model("./data/frame0.jpg")
#results = model.track(source="<REDACTED>", show=True)

cap = cv2.VideoCapture("<REDACTED>")

try:
    if not os.path.exists('./data'):
        os.makedirs('./data')
except OSError:
    print('Error: Creating directory of data')

fps = int(cap.get(cv2.CAP_PROP_FPS))

frame_count = 0
second = 0

while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, second * fps)
    ret, frame = cap.read()
    if not ret:
        break

    frame_filename = os.path.join(f"frame_{frame_count}.jpg")
    cv2.imwrite(frame_filename, frame)
    frame_count += 1
    second += 1


camerasnap.release()
cv2.destroyAllWindows()
