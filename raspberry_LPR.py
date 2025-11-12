# webcam_lpr.py
import time, os, re
import cv2
import numpy as np
import easyocr
from collections import Counter, deque
from ultralytics import YOLO

# --- settings ---
WEIGHTS = "license_plate_detector.pt"   # your trained detector
OCR_EVERY_SEC = 0.33                    # OCR rate
PLATE_RX = re.compile(r'^[A-Z]{3}\d{3}$')  # AAA999

# position-aware fixes: left 3 are letters, right 3 are digits
LETTER_FIX = str.maketrans({'0':'O','1':'I','2':'Z','5':'S','8':'B'})
DIGIT_FIX  = str.maketrans({'O':'0','I':'1','Z':'2','S':'5','B':'8'})

def clean_candidate(raw: str) -> str | None:
    """
    Uppercase, strip non-alnum, scan 6-char windows, and coerce
    letter slots vs digit slots to fix common OCR confusions.
    Return AAA999 if found, else None.
    """
    s = ''.join(ch for ch in raw.upper() if ch.isalnum())
    if len(s) < 6:
        return None
    for i in range(0, len(s) - 5):
        w = s[i:i+6]
        w = w[:3].translate(LETTER_FIX) + w[3:].translate(DIGIT_FIX)
        if PLATE_RX.match(w):
            return w
    return None

class PlateStabilizer:
    """
    Keep last N candidates; emit when the same plate appears K times
    with sufficient average OCR confidence, and apply a cooldown.
    """
    def __init__(self, N=8, K=5, min_conf=0.55, cooldown=3.0):
        self.buf = deque(maxlen=N)
        self.K = K
        self.min_conf = min_conf
        self.cooldown = cooldown
        self.last_emit_t = 0.0

    def push(self, plate: str, conf: float):
        if not plate or conf < self.min_conf:
            return None
        self.buf.append((plate, conf, time.time()))
        counts = Counter([p for p, _, _ in self.buf])
        candidate, hits = counts.most_common(1)[0]
        if hits >= self.K:
            avg_conf = np.mean([c for p, c, _ in self.buf if p == candidate])
            if time.time() - self.last_emit_t > self.cooldown:
                self.last_emit_t = time.time()
                return candidate, float(avg_conf)
        return None

def prep_for_ocr(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    g = cv2.bilateralFilter(g, 7, 75, 75)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def ocr_text_and_conf(reader, roi):
    """
    easyocr.readtext returns [ [bbox, text, conf], ... ]
    We join texts and average conf for a coarse score.
    """
    res = reader.readtext(roi, detail=1, paragraph=True)
    if not res:
        return "", 0.0
    texts, confs = [], []
    for _, txt, conf in res:
        if txt:
            texts.append(txt)
            confs.append(conf if conf is not None else 0.0)
    raw = ''.join(texts)
    avg_conf = float(np.mean(confs)) if confs else 0.0
    return raw, avg_conf

# --- load models ---
assert os.path.isfile(WEIGHTS), f"Missing weights: {WEIGHTS} in {os.getcwd()}"
plate_model = YOLO(WEIGHTS)
reader = easyocr.Reader(['en'], gpu=False)

# --- webcam ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # change index if needed
assert cap.isOpened(), "No webcam found."

stb = PlateStabilizer(N=8, K=5, min_conf=0.55, cooldown=3.0)
last_ocr_t = 0.0
last_raw = ""
locked_display = ""   # last emitted plate for HUD

while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]

    # 1) detect plates
    det = plate_model(frame, verbose=False, imgsz=640)[0]

    # choose the highest-confidence box (optional: handle many)
    boxes = det.boxes
    if boxes is not None and len(boxes) > 0:
        i = int(np.argmax(boxes.conf.cpu().numpy()))
        bx = boxes[i]
        x1, y1, x2, y2 = bx.xyxy[0].cpu().numpy().astype(int)
        conf = float(bx.conf[0])

        if conf >= 0.35:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            crop = frame[y1:y2, x1:x2]

            # 2) OCR ~3x/sec on the cropped plate
            now = time.time()
            if crop.size > 0 and (now - last_ocr_t) > OCR_EVERY_SEC:
                roi = prep_for_ocr(crop)
                raw, avg_conf = ocr_text_and_conf(reader, roi)
                last_ocr_t = now
                last_raw = raw

                # 3) Clean to AAA999 & 4) Stabilize
                cand = clean_candidate(raw)
                out = stb.push(cand, avg_conf) if cand else None
                if out:
                    plate, conf_avg = out
                    locked_display = f"{plate} ({conf_avg:.2f})"
                    # 5) Do your one-shot actions here:
                    #    - save snapshot
                    #    - send via UART/HTTP/MQTT, etc.
                    ts = int(time.time())
                    cv2.imwrite(f"plate_{plate}_{ts}.jpg", frame)
                    print("STABLE:", locked_display)

            # draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 40), 2)
            cv2.putText(frame, f"plate {conf:.2f}", (x1, max(0, y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 220, 40), 2)

    # HUD
    if last_raw:
        cv2.putText(frame, f"OCR raw: {last_raw[:24]}", (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 120, 0), 2)
    if locked_display:
        cv2.putText(frame, f"STABLE: {locked_display}", (12, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 255), 2)

    cv2.imshow("ALPR (ESC to exit)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
