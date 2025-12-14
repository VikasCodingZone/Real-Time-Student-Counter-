import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import argparse
import time

# ------------------ Args ------------------
ap = argparse.ArgumentParser()
ap.add_argument("--source", default="0", help="0=webcam, or path/rtsp/url")
ap.add_argument("--model", default="yolov8n.pt", help="yolov8n/s/m/l/x .pt")
ap.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
ap.add_argument("--imgsz", type=int, default=640, help="inference size")
ap.add_argument("--device", default="", help="cuda / 0 / cpu (auto if empty)")
ap.add_argument("--smooth_k", type=int, default=8, help="smoothing window")
args = ap.parse_args()

# --------------- Load Model ---------------
# persist=True keeps model warm for higher FPS
model = YOLO(args.model)
names = model.names  # class id -> label map

# --------------- Video Source --------------
src = 0 if args.source == "0" else args.source
cap = cv2.VideoCapture(src)
if not cap.isOpened():
    raise SystemExit("❌ Video source open nahi ho raha. --source check karein.")

# Optional: set a lower resolution for speed (uncomment if needed)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ------------- Smoothing Helper ------------
count_hist = deque(maxlen=max(3, args.smooth_k))
prev_smooth = 0
font = cv2.FONT_HERSHEY_SIMPLEX
t0 = time.time()
frames = 0

# ------------- Inference Loop --------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not received. Stream ended?")
        break
    frames += 1

    # Ultralytics YOLO predict (stream=False returns list)
    results = model.predict(
        source=frame,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device if args.device else None,
        verbose=False
    )

    # Count only 'person'
    persons = 0
    r = results[0]
    if r and r.boxes is not None:
        for b in r.boxes:
            cls = int(b.cls[0])
            if names.get(cls, "") == "person":
                persons += 1
                # Draw bbox
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(frame, "Student", (x1, max(20, y1-8)), font, 0.55, (0, 200, 0), 2)

    # Smooth the count (reduces flicker)
    count_hist.append(persons)
    smooth = int(round(np.mean(count_hist)))

    # Change indicator
    if smooth > prev_smooth:
        arrow, color = " ↑", (0, 200, 0)
    elif smooth < prev_smooth:
        arrow, color = " ↓", (0, 0, 200)
    else:
        arrow, color = "", (0, 170, 255)
    prev_smooth = smooth

    # FPS
    dt = time.time() - t0
    fps = frames / dt if dt > 0 else 0.0

    # Header banner
    cv2.rectangle(frame, (10, 10), (460, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Students: {smooth}{arrow}", (20, 55), font, 1.2, color, 3)
    cv2.putText(frame, f"FPS: {fps:.1f}", (330, 55), font, 0.7, (255, 255, 255), 2)

    cv2.imshow("YOLOv8 Student Counter (Fast)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
