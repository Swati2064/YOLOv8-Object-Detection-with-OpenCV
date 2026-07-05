import ultralytics
ultralytics.checks()

from ultralytics import YOLO

model = YOLO("yolo11n.pt")
print("YOLO loaded successfully!")