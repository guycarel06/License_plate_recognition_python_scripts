import os
import time
import cv2
from ultralytics import YOLO
import easyocr

# ================== CONFIG ==================

# Folder where VLC saves snapshots

SNAP_DIR = "/home/drone/lpr/Vlc_snapshots"

# YOLO weights
WEIGHTS = "license_plate_detector.pt"

# ============================================

# ---------- LPR setup ----------
print("[LPR] Loading YOLO model...")
plate_model = YOLO(WEIGHTS)

print("[LPR] Initializing EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)


def run_lpr_on_file(path: str):
    """Running YOLO + OCR on a single image file"""
    print(f"\n New snapshot: {path}")
    img = cv2.imread(path)
    if img is None:
        print("[LPR] Could not read image.")
        return

    # 1) YOLO plate detection
    results = plate_model(img)[0]

    best_text = None

    for box in results.boxes:
        # bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]

        # prepare crop for OCR
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # OCR
        ocr_results = reader.readtext(gray)

        for (_bbox, text, conf) in ocr_results:
            print(f"[LPR] OCR candidate: {text} (conf={conf:.2f})")
            best_text = text

    if best_text:
        print("Plate detected:", best_text)
    else:
        print("No plate detected")


def watcher_loop():
    """Watch SNAP_DIR and run LPR on every new image file."""
    os.makedirs(SNAP_DIR, exist_ok=True)

    # start with existing files as "known"
    known = {
        f for f in os.listdir(SNAP_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    }

    print(f"[WATCHER] Watching {SNAP_DIR} for new snapshots...")
    print("[WATCHER] Make sure VLC is saving snapshots there.")
    print("[WATCHER] Press Ctrl+C to stop.\n")

    while True:
        # list current image files
        current = [
            f for f in os.listdir(SNAP_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        # new = in current but not in known
        new_files = [f for f in current if f not in known]

        for fname in sorted(new_files):
            path = os.path.join(SNAP_DIR, fname)
            # small delay to be safe that VLC finished writing file
            time.sleep(0.3)
            run_lpr_on_file(path)

        # update known set
        known.update(new_files)

        # don't hammer the disk
        time.sleep(0.5)


def main():
    try:
        watcher_loop()
    except KeyboardInterrupt:
        print("\n[WATCHER] Stopping due to Ctrl+C...")


if __name__ == "__main__":
    main()
