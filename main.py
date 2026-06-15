import cv2
from ultralytics import YOLO

# Load YOLOv8 pretrained model
model = YOLO("yolov8n.pt")  # you can use 'yolov8s.pt' for more accuracy

# Get the class index for 'cow'
COW_IDX = [i for i, n in model.names.items() if n == "cow"][0]

# Open laptop webcam
cap = cv2.VideoCapture(0)  # 0 = default camera

if not cap.isOpened():
    print("❌ Error: Could not open camera.")
    exit()

print("✅ Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection only for 'cow'
    results = model.predict(frame, classes=[COW_IDX], conf=0.4, verbose=False)

    # Draw results
    annotated_frame = results[0].plot()
    cv2.imshow("Cow Detector", annotated_frame)

    if len(results[0].boxes) > 0:
        print("✅ Cow detected in camera!")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
