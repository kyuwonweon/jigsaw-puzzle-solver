import cv2
import numpy as np
import pyrealsense2 as rs
from pathlib import Path

OUT_DIR = Path(__file__).parent / "image/templates_crab/live_templates"
OUT_DIR.mkdir(exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

for _ in range(30):
    pipeline.wait_for_frames()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

try:
    while True:
        frames = pipeline.wait_for_frames()
        color = np.asanyarray(frames.get_color_frame().get_data())

        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        piece_mask = (hsv[:, :, 1] > 40).astype(np.uint8) * 255
        piece_mask = cv2.morphologyEx(piece_mask, cv2.MORPH_OPEN,  kernel)
        piece_mask = cv2.morphologyEx(piece_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(piece_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        display = color.copy()

        best_cnt = None
        if contours:
            best_cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(best_cnt) > 1500:
                x, y, w, h = cv2.boundingRect(best_cnt)
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(display, "press piece number to save",
                            (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("capture", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        key_map = {ord(str(i)): i for i in range(10)}
        key_map[ord('a')] = 10
        key_map[ord('b')] = 11

        if key in key_map and best_cnt is not None:
            piece_id = key_map[key]
            x, y, w, h = cv2.boundingRect(best_cnt)

            crop = color[y:y+h, x:x+w]
            lmask = piece_mask[y:y+h, x:x+w]
            bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = lmask

            out = OUT_DIR / f"piece_{piece_id:02d}.png"
            cv2.imwrite(str(out), bgra)
            print(f"Saved piece_{piece_id:02d}.png ({w}x{h})", flush=True)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
