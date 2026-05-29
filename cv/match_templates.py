import cv2
import numpy as np
from pathlib import Path

base_dir      = Path(__file__).parent / "image/templates_crab"
templates_dir = base_dir / "templates"

bg      = cv2.imread(str(base_dir / "background.png"))
bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
bf   = cv2.BFMatcher(cv2.NORM_L2)

# Pre-compute background keypoints once
kp_bg, des_bg = sift.detectAndCompute(bg_gray, None)
print(f"Background keypoints: {len(kp_bg)}\n")

results = []   # (idx, x0, y0, w, h, n_inliers)

for tf in sorted(templates_dir.glob("piece_*.png")):
    idx  = int(tf.stem.split("_")[1])
    tmpl = cv2.imread(str(tf))
    gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)

    # Mask out dark table background so SIFT only picks up piece features
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kp_t, des_t = sift.detectAndCompute(gray, mask)
    if des_t is None or len(kp_t) < 4:
        print(f"piece_{idx:02d}: only {len(kp_t) if kp_t else 0} keypoints — skip")
        continue

    # Lowe's ratio test to keep only reliable matches
    raw = bf.knnMatch(des_t, des_bg, k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]

    if len(good) < 4:
        print(f"piece_{idx:02d}: {len(good)} good matches — not enough for homography")
        continue

    src = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_bg[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    if H is None:
        print(f"piece_{idx:02d}: homography failed")
        continue

    # Project the template corners into background coordinates
    h, w = tmpl.shape[:2]
    corners   = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    x0, y0 = projected.min(axis=0).astype(int)
    x1, y1 = projected.max(axis=0).astype(int)
    n_in = int(inlier_mask.sum())

    results.append((idx, x0, y0, x1 - x0, y1 - y0, n_in))
    print(f"piece_{idx:02d}: bbox ({x0:4d},{y0:4d}) {x1-x0}×{y1-y0}  inliers={n_in}")

# ── Visualise ─────────────────────────────────────────────────────────────────
vis = bg.copy()
bH, bW = bg.shape[:2]

for idx, x, y, w, h, n_in in results:
    # Clamp to image bounds for drawing
    x1, y1 = min(bW, x+w), min(bH, y+h)
    x0, y0 = max(0, x), max(0, y)
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.putText(vis, f"{idx}({n_in})", (x0+3, y0+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

out = base_dir / "match_result.png"
cv2.imwrite(str(out), vis)
print(f"\nResult saved → {out}")
print(f"Matched {len(results)}/12 pieces")
