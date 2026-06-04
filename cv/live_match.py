"""Live puzzle piece matching using RealSense camera and template matching.

This module captures video from a RealSense camera, detects puzzle pieces,
and matches them against a library of template pieces using color histograms
and shape descriptors.
"""
from pathlib import Path

import cv2

import numpy as np

import pyrealsense2 as rs

TEMPLATES_DIR = Path('image/templates_crab/live_templates')
MIN_AREA = 2000
TOP_N = 3

# Load templates
templates = []
for tf in sorted(TEMPLATES_DIR.glob('piece_*.png')):
    idx = tf.stem.split('_')[1]
    img = cv2.imread(str(tf), cv2.IMREAD_UNCHANGED)
    if img is None:
        continue
    bgr = img[:, :, :3]
    mask = (img[:, :, 3] > 128).astype(np.uint8)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], mask, [36], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], mask, [32], [0, 256])
    cv2.normalize(h_hist, h_hist)
    cv2.normalize(s_hist, s_hist)

    moments = cv2.moments(mask)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(cnts[0])
    aspect = w / h if h > 0 else 1.0

    templates.append({
        'idx': idx,
        'h_hist': h_hist,
        's_hist': s_hist,
        'hu': hu,
        'aspect': aspect,
        'area': mask.sum(),
        'bgr': bgr,
        'mask': mask,
    })

print(f'Loaded {len(templates)} templates')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
for _ in range(30):
    pipeline.wait_for_frames()


def match_piece(roi_bgr, roi_mask):
    """Match a puzzle piece ROI against template library.

    Args:
        roi_bgr: BGR image of the puzzle piece region.
        roi_mask: Binary mask of the puzzle piece region.

    Returns:
        Tuple of (best_idx, best_score, top3_matches) or (None, 999) if no contours found.
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], roi_mask, [36], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], roi_mask, [32], [0, 256])
    cv2.normalize(h_hist, h_hist)
    cv2.normalize(s_hist, s_hist)

    moments = cv2.moments(roi_mask)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    cnts, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 999
    x, y, w, h = cv2.boundingRect(cnts[0])
    aspect = w / h if h > 0 else 1.0

    scores = []
    for t in templates:
        h_score = cv2.compareHist(h_hist, t['h_hist'], cv2.HISTCMP_BHATTACHARYYA)
        s_score = cv2.compareHist(s_hist, t['s_hist'], cv2.HISTCMP_BHATTACHARYYA)
        hu_score = min(np.linalg.norm(hu - t['hu']) * 0.01, 1.0)
        asp_score = min(abs(aspect - t['aspect']) * 0.5, 1.0)
        total = 0.5 * h_score + 0.2 * s_score + 0.2 * hu_score + 0.1 * asp_score
        scores.append((total, t['idx']))

    scores.sort()
    return scores[0][1], scores[0][0], scores[:TOP_N]


print("Press 's' to save screenshot, 'q' to quit")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color = np.asanyarray(frames.get_color_frame().get_data())
        display = color.copy()

        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        piece_mask = (hsv[:, :, 1] > 40).astype(np.uint8) * 255
        piece_mask = cv2.morphologyEx(piece_mask, cv2.MORPH_OPEN, kernel)
        piece_mask = cv2.morphologyEx(piece_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            piece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(cnt)

            if x <= 5 or y <= 5 or x+w >= 635 or y+h >= 475:
                continue

            roi_bgr = color[y:y+h, x:x+w]
            roi_mask = piece_mask[y:y+h, x:x+w]

            result = match_piece(roi_bgr, roi_mask)
            if result[0] is None:
                continue
            best_idx, best_score, top3 = result

            color_box = (0, 255, 0) if best_score < 0.35 else (0, 200, 255)
            cv2.rectangle(display, (x, y), (x+w, y+h), color_box, 2)

            label = f'{best_idx} ({best_score:.2f})'
            cv2.putText(display, label, (x+3, y+16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            for i, (sc, idx) in enumerate(top3[:2]):
                cv2.putText(display, f'  #{i+1}: {idx} {sc:.2f}',
                            (x+3, y+h+14+i*14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        cv2.imshow('live match', display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('match_screenshot.png', display)
            print('Saved match_screenshot.png')

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
