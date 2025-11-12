# webcam_lpr.py
import time
import os
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO


WEIGHTS = "license_plate_detector.pt"   
assert os.path.isfile(WEIGHTS), f"Missing weights: {WEIGHTS} in {os.getcwd()}"

# --- load YOLO plate detector from local .pt file ---
plate_model = YOLO(WEIGHTS)

# --- OCR reader (first run downloads small models to ~/.EasyOCR) ---
reader = easyocr.Reader(['en'], gpu=False)

# --- open webcam ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # try 1 or 2 if needed
assert cap.isOpened(), "No webcam found."

last_text, last_t = "", 0.0

def prep_for_ocr(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    g = cv2.bilateralFilter(g, 7, 75, 75)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # detect plates
    res = plate_model(frame, verbose=False, imgsz=640)[0]

    for bx in res.boxes:
        x1, y1, x2, y2 = bx.xyxy[0].cpu().numpy().astype(int)
        conf = float(bx.conf[0])
        if conf < 0.35:
            continue

        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # OCR ~3 times/sec
        now = time.time()
        if now - last_t > 0.33:
            th = prep_for_ocr(crop)
            texts = reader.readtext(th, detail=0, paragraph=True)
            raw = " ".join(texts)
            text = "".join(ch for ch in raw if ch.isalnum())  # keep A–Z0–9
            if text:
                last_text = text
                last_t = now

        # draw detection
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, f"Plate {conf:.2f}", (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    if last_text:
        cv2.putText(
            frame, f"READ: {last_text}", (15, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2
        )

    cv2.imshow("ALPR (ESC to exit)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
