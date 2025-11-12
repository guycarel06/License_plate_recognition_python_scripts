# webcam_lpr_rc832s.py
import os
import time
import argparse
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

def parse_args():
    ap = argparse.ArgumentParser(description="Analog FPV (RC832S) → USB capture → YOLO + EasyOCR LPR")
    ap.add_argument("--weights", default="license_plate_detector.pt",
                    help="Path to YOLO weights (.pt)")
    ap.add_argument("--device", type=str, default=None,
                    help="Camera index or V4L2 path (e.g., 1 or /dev/video2). If omitted, auto-discover.")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    ap.add_argument("--conf", type=float, default=0.35, help="Min confidence for detections")
    ap.add_argument("--mirror", action="store_true", help="Mirror the preview horizontally")
    ap.add_argument("--no-deinterlace", action="store_true", help="Disable quick deinterlacing")
    ap.add_argument("--fps", type=int, default=30, help="Target capture FPS")
    ap.add_argument("--width", type=int, default=720, help="Capture width (analog: 720)")
    ap.add_argument("--height", type=int, default=480, help="Capture height (NTSC: 480; PAL: 576)")
    return ap.parse_args()

def quick_deinterlace(bgr):
    # Simple deinterlace: keep even lines and scale back
    even = bgr[0::2, :, :]
    return cv2.resize(even, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

def open_capture(device_hint, width, height, fps):
    # Try a direct path or index first if provided
    backends = []
    if os.name == "nt":
        # Windows: try both DirectShow and Media Foundation
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    else:
        # Linux/macOS: use default (V4L2 on Linux)
        backends = [cv2.CAP_ANY]

    if device_hint is not None:
        # device_hint can be "1" or "/dev/video2"
        dev = device_hint
        # Convert to int if it's a pure number
        try:
            dev = int(device_hint)
        except ValueError:
            pass
        for be in backends:
            cap = cv2.VideoCapture(dev, be)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS,          fps)
                return cap
        raise RuntimeError(f"Could not open capture device: {device_hint}")

    # Auto-discover: try common indexes
    candidate_indexes = [1, 2, 0, 3, 4]
    for idx in candidate_indexes:
        for be in backends:
            cap = cv2.VideoCapture(idx, be)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS,          fps)
                print(f"[INFO] Using capture device index {idx} (backend {be})")
                return cap
    raise RuntimeError("No capture device found. Is the USB CVBS dongle connected?")

def prep_for_ocr(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Slight sharpen before scaling helps analog feeds
    k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
    g = cv2.filter2D(g, -1, k)
    g = cv2.resize(g, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    g = cv2.bilateralFilter(g, 7, 75, 75)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def main():
    args = parse_args()

    # --- Load YOLO ---
    assert os.path.isfile(args.weights), f"Missing weights: {args.weights} in {os.getcwd()}"
    plate_model = YOLO(args.weights)

    # --- OCR reader ---
    reader = easyocr.Reader(['en'], gpu=False)

    # --- Open capture ---
    cap = open_capture(args.device, args.width, args.height, args.fps)

    last_text, last_t = "", 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Capture read failed; exiting.")
            break

        if not args.no_deinterlace:
            frame = quick_deinterlace(frame)

        if args.mirror:
            frame = cv2.flip(frame, 1)

        # Detect plates
        res = plate_model(frame, verbose=False, imgsz=args.imgsz)[0]

        for bx in res.boxes:
            x1, y1, x2, y2 = bx.xyxy[0].cpu().numpy().astype(int)
            conf = float(bx.conf[0])
            if conf < args.conf:
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
                text = "".join(ch for ch in raw if ch.isalnum())  # A–Z0–9
                if text:
                    last_text = text
                    last_t = now

            # Draw detection box/label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Plate {conf:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if last_text:
            cv2.putText(frame, f"READ: {last_text}", (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        cv2.imshow("ALPR (ESC to exit)", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
