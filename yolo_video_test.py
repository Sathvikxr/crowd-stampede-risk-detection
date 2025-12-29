import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open video
video_path = "videos/sample.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read first frame
ret, prev_frame = cap.read()
if not ret:
    print("Error reading first frame")
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Frame info
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_area = frame_width * frame_height

print("Running stampede risk analysis. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    people_count = 0

    # -------- YOLO detection --------
    results = model(frame, stream=True)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if cls_id == 0 and confidence > 0.4:
                people_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)

    # -------- Density --------
    density = people_count / frame_area

    if density < 0.00005:
        density_level = "LOW"
    elif density < 0.00015:
        density_level = "MEDIUM"
    else:
        density_level = "HIGH"

    # -------- Motion --------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray,
        None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    motion_score = np.mean(magnitude)

    if motion_score < 1.0:
        motion_state = "CALM"
    elif motion_score < 3.0:
        motion_state = "MOVING"
    else:
        motion_state = "CHAOTIC"

    prev_gray = gray

    # -------- Stampede Risk Logic --------
    if density_level == "HIGH" and motion_state == "CHAOTIC":
        risk_level = "HIGH RISK"
        risk_color = (0, 0, 255)
    elif density_level in ["MEDIUM", "HIGH"] or motion_state == "MOVING":
        risk_level = "WARNING"
        risk_color = (0, 255, 255)
    else:
        risk_level = "NORMAL"
        risk_color = (0, 255, 0)

    # -------- Display --------
    cv2.putText(frame, f"People: {people_count}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)

    cv2.putText(frame, f"Density: {density_level}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)

    cv2.putText(frame, f"Motion: {motion_state}",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)

    cv2.putText(frame, f"Risk: {risk_level}",
                (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, risk_color, 3)

    cv2.imshow("Stampede Risk Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
