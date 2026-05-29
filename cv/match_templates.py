import cv2
import numpy as np
from pathlib import Path

base_dir = Path(__file__).parent / "image/templates_crab"
templates_dir = base_dir / "templates"

bg = cv2.imread(str(base_dir / "background.png"))
bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
bg_hue = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)[:, :, 0]

sift_strict = cv2.SIFT_create()
sift_loose  = cv2.SIFT_create(contrastThreshold=0.01)
bf = cv2.BFMatcher(cv2.NORM_L2)

kp_bg_strict, des_bg_strict = sift_strict.detectAndCompute(bg_gray, None)
kp_bg_hue_strict, des_bg_hue_strict = sift_strict.detectAndCompute(bg_hue,None)
kp_bg_loose, des_bg_loose = sift_loose.detectAndCompute(bg_gray, None)
kp_bg_hue_loose, des_bg_hue_loose = sift_loose.detectAndCompute(bg_hue, None)

def try_match(sift, kp_bg, des_bg, img, mask, idx, label, ratio=0.75):
    kp_t, des_t = sift.detectAndCompute(img, mask)
    if des_t is None or len(kp_t) < 4:
        return None, None, f"piece_{idx:02d} [{label}]: only {len(kp_t) if kp_t else 0} keypoints"

    raw  = bf.knnMatch(des_t, des_bg, k=2)
    good = [m for m, n in raw if m.distance < ratio * n.distance]
    if len(good) < 4:
        return None, None, f"piece_{idx:02d} [{label}]: {len(good)} good matches"

    src = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_bg[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return None, None, f"piece_{idx:02d} [{label}]: homography failed"
    return H, int(inlier_mask.sum()), None


def project_bbox(H, mask, tmpl_wh, bg_shape, idx, label):
    tw, th = tmpl_wh
    bH, bW = bg_shape[:2]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, f"piece_{idx:02d} [{label}]: no contour in mask"
    pts       = cnts[0].astype(np.float32)
    projected = cv2.perspectiveTransform(pts, H)
    x0, y0    = projected.min(axis=0)[0].astype(int)
    x1, y1    = projected.max(axis=0)[0].astype(int)
    pw, ph    = x1 - x0, y1 - y0

    if pw <= 0 or ph <= 0:
        return None, f"piece_{idx:02d} [{label}]: degenerate bbox ({pw}×{ph})"
    if x1 < 0 or y1 < 0 or x0 > bW or y0 > bH:
        return None, f"piece_{idx:02d} [{label}]: projected outside background"
    if not (0.3 < (pw * ph) / (tw * th) < 2.0):
        return None, f"piece_{idx:02d} [{label}]: size {pw}×{ph} unreasonable vs {tw}×{th}"

    return (x0, y0, pw, ph), None


def ncc_match(bgr, alpha, bg_bgr, idx):
    """Multi-scale BGR template matching for pieces where SIFT fails."""
    bH, bW = bg_bgr.shape[:2]
    th, tw = bgr.shape[:2]
    best_val, best_box = -np.inf, None

    for scale in np.arange(0.50, 1.05, 0.05):
        rw, rh = int(tw * scale), int(th * scale)
        if rw < 10 or rh < 10 or rw >= bW or rh >= bH:
            continue
        t_resized = cv2.resize(bgr,   (rw, rh))
        m_resized = cv2.resize(alpha, (rw, rh))
        mask3     = cv2.merge([m_resized] * 3)
        result    = cv2.matchTemplate(bg_bgr, t_resized, cv2.TM_CCORR_NORMED, mask=mask3)
        _, val, _, loc = cv2.minMaxLoc(result)
        if val > best_val:
            best_val, best_box = val, (loc[0], loc[1], rw, rh)

    if best_box is None or best_val < 0.35:
        return None, f"piece_{idx:02d} [ncc]: best={best_val:.3f} below threshold"
    x0, y0, pw, ph = best_box
    return (x0, y0, pw, ph), None


results = []
tmpl_size = {}
stages = [
    (sift_strict, kp_bg_strict, des_bg_strict, "gray", "strict gray", 0.75),
    (sift_strict, kp_bg_hue_strict, des_bg_hue_strict, "hue",  "strict hue", 0.75),
    (sift_loose, kp_bg_loose, des_bg_loose, "gray", "loose gray", 0.75),
    (sift_loose, kp_bg_hue_loose, des_bg_hue_loose, "hue", "loose hue", 0.75),
    (sift_loose, kp_bg_loose, des_bg_loose, "gray", "wide", 0.85),
]

for tf in sorted(templates_dir.glob("piece_*.png")):
    idx = int(tf.stem.split("_")[1])
    tmpl = cv2.imread(str(tf), cv2.IMREAD_UNCHANGED)
    mask = tmpl[:, :, 3]
    bgr = tmpl[:, :, :3]
    tmpl_size[idx] = (tmpl.shape[1], tmpl.shape[0])
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hue = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 0]
    channels = {"gray": gray, "hue": hue}

    last_err = f"piece_{idx:02d}: all stages failed"
    matched  = False

    for sift, kp_bg, des_bg, ch_key, label, ratio in stages:
        H, n_in, err = try_match(sift, kp_bg, des_bg, channels[ch_key], mask, idx, label, ratio)
        if H is None:
            last_err = err
            continue
        bbox, err = project_bbox(H, mask, (tmpl.shape[1], tmpl.shape[0]), bg.shape, idx, label)
        if bbox is None:
            last_err = err
            continue
        x0, y0, pw, ph = bbox
        print(f"piece_{idx:02d}: bbox ({x0:4d},{y0:4d}) {pw}×{ph}  inliers={n_in}  [{label}]")
        results.append((idx, x0, y0, pw, ph, n_in))
        matched = True
        break

    if not matched:
        bbox, err = ncc_match(bgr, mask, bg, idx)
        if bbox is not None:
            x0, y0, pw, ph = bbox
            print(f"piece_{idx:02d}: bbox ({x0:4d},{y0:4d}) {pw}×{ph}  [ncc]")
            results.append((idx, x0, y0, pw, ph, 0))
        else:
            print(err)

# Post processing
hi_scales = [
    np.sqrt((pw * ph) / (tmpl_size[idx][0] * tmpl_size[idx][1]))
    for idx, _, _, pw, ph, n_in in results
    if n_in >= 15 and idx in tmpl_size
]
if hi_scales:
    ref_scale = np.median(hi_scales)
    refined = []
    for r in results:
        idx, x0, y0, pw, ph, n_in = r
        if idx in tmpl_size:
            tw, th = tmpl_size[idx]
            scale = np.sqrt((pw * ph) / (tw * th))
            if n_in == 0:
                pass
            elif scale > ref_scale * 1.3:
                # Low-confidence SIFT — re-match entirely with NCC
                tmpl = cv2.imread(str(templates_dir / f"piece_{idx:02d}.png"), cv2.IMREAD_UNCHANGED)
                bbox, _ = ncc_match(tmpl[:, :, :3], tmpl[:, :, 3], bg, idx)
                if bbox is not None:
                    x0, y0, pw, ph = bbox
                    n_in = 0
                    print(f"piece_{idx:02d}: scale {scale:.2f} > {ref_scale*1.3:.2f} — re-matched with NCC → ({x0},{y0}) {pw}×{ph}")
                    r = (idx, x0, y0, pw, ph, n_in)
            elif scale > ref_scale * 1.1 and n_in >= 25:
                # High-confidence position, oversized — keep SIFT center, resize to ref scale
                new_w = int(tw * ref_scale)
                new_h = int(th * ref_scale)
                cx, cy = x0 + pw // 2, y0 + ph // 2
                x0, y0 = cx - new_w // 2, cy - new_h // 2
                pw, ph = new_w, new_h
                print(f"piece_{idx:02d}: scale {scale:.2f} — resized to {pw}×{ph}")
                r = (idx, x0, y0, pw, ph, n_in)
        refined.append(r)
    results = refined

# Visualize
bH, bW = bg.shape[:2]
bbox_vis = bg.copy()
for idx, x, y, w, h, n_in in results:
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(bW, x + w), min(bH, y + h)
    cv2.rectangle(bbox_vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.putText(bbox_vis, f"{idx}({n_in})", (x0 + 3, y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

cv2.imwrite(str(base_dir / "match_result.png"), bbox_vis)

