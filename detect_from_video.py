import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort  # Make sure sort.py is present in the same folder

# Load YOLOv8 model
model = YOLO("C:/Users/SHREE/Downloads/best.pt")  # Replace with your model if needed

# Initialize SORT
tracker = Sort()

# Class names (adjust if needed)
CLASS_NAMES = ["ball", "player", "referee"]

# Colors for drawing
COLORS = {
    "ball": (0, 255, 255),
    "player": (0, 255, 0),
    "referee": (255, 0, 0)
}

# Open video file
cap = cv2.VideoCapture("C:/Users/SHREE/Downloads/15sec_input_720p.mp4")

print("🚀 Starting object detection and tracking... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = model(frame)[0]

    # Prepare detections for SORT
    dets_for_sort = []
    ball_added = False  # Only allow one ball

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())

        # Avoid out-of-range class IDs
        if cls_id >= len(CLASS_NAMES):
            continue

        class_name = CLASS_NAMES[cls_id]

        if class_name == "ball":
            if not ball_added:
                dets_for_sort.append([x1, y1, x2, y2, conf, cls_id])
                ball_added = True
        else:
            dets_for_sort.append([x1, y1, x2, y2, conf, cls_id])

    # Remove class info for SORT input
    if dets_for_sort:
        dets_np = np.array(dets_for_sort)
        sort_input = dets_np[:, :5]  # Only x1, y1, x2, y2, conf
    else:
        sort_input = np.empty((0, 5))

    # Update SORT tracker
    tracks = tracker.update(sort_input)

    # Draw tracked objects
    for track in tracks:
        x1, y1, x2, y2, track_id = track
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Match to original class ID (rough match)
        matched_cls = "unknown"
        for det in dets_for_sort:
            if np.allclose([x1, y1, x2, y2], det[:4], atol=5):
                matched_cls = CLASS_NAMES[int(det[5])]
                break

        color = COLORS.get(matched_cls, (255, 255, 255))
        label = f"{matched_cls} ID:{int(track_id)}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
