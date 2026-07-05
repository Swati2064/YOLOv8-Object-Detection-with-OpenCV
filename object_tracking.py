import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO(r"D:\AI_NLP_Projects\YOLO_model\yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform object tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )

    # Draw bounding boxes and tracking IDs
    annotated_frame = results[0].plot()

    # Display output
    cv2.imshow("YOLOv8 Object Tracking", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()