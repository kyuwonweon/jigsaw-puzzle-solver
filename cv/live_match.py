import cv2
import numpy as np
import pyrealsense2 as rs
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent / "image/templates_crab/live_templates"
MIN_AREA = 1500
MIN_MATCHES = 6
RATIO = 0.75

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

for _ in range(30):
    pipeline.wait_for_frames()

sift = cv2.SIFT_create()
bf   = cv2.BFMatcher(cv2.NORM_L2)

tmpl_features = []
for tf in sorted(TEMPLATES_DIR.glob("piece_*.png")):
    idx   = int(tf.stem.split("_")[1])
    tmpl  = cv2.imread(str(tf), cv2.IMREAD_UNCHANGED)
    gray  = cv2.cvtColor(tmpl[:, :, :3], cv2.COLOR_BGR2GRAY)
    mask  = (tmpl[:, :, 3] > 128).astype(np.uint8)
    kp, des = sift.detectAndCompute(gray, mask)
    tmpl_features.append((idx, kp, des))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

try:
    while True:
        frames = pipeline.wait_for_frames()
        while True:
            fresh = pipeline.poll_for_frames()
            if fresh:
                frames = fresh
            else:
                break
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        color = np.asanyarray(color_frame.get_data())

        gray_frame = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        hsv_frame  = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        piece_mask = (hsv_frame[:, :, 1] > 40).astype(np.uint8) * 255
        piece_mask = cv2.morphologyEx(piece_mask, cv2.MORPH_OPEN,  kernel)
        piece_mask = cv2.morphologyEx(piece_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(piece_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(cnt)

            roi_gray = gray_frame[y:y+h, x:x+w]
            roi_mask = piece_mask[y:y+h, x:x+w]
            kp_roi, des_roi = sift.detectAndCompute(roi_gray, roi_mask)

            if des_roi is None or len(des_roi) < 2:
                continue

            best_matches, best_idx = -1, -1
            for idx, kp_t, des_t in tmpl_features:
                if des_t is None or len(des_t) < 2:
                    continue
                raw = bf.knnMatch(des_roi, des_t, k=2)
                good = [m for m, n in raw if m.distance < RATIO * n.distance]
                if len(good) > best_matches:
                    best_matches, best_idx = len(good), idx

            if best_matches >= MIN_MATCHES:
                cv2.rectangle(color, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(color, str(best_idx), (x + 3, y + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("live match", color)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
