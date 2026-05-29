import cv2
import numpy as np
from pathlib import Path

base_dir = Path(__file__).parent / "image/templates_crab"
templates_dir = base_dir / "templates"
templates_dir.mkdir(exist_ok=True)
for f in templates_dir.glob("piece_*.png"):
    f.unlink()

## Extract piece templates from splitted.webp 
img = cv2.imread(str(base_dir / "splitted.webp"))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

_, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8), iterations=1)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_area = img.shape[0] * img.shape[1] * 0.01
pieces = []
for c in contours:
    if cv2.contourArea(c) > min_area:
        x, y, w, h = cv2.boundingRect(c)
        if max(w, h) / min(w, h) < 3:
            pieces.append((x, y, w, h, c))

if not pieces:
    raise RuntimeError("No puzzle pieces detected. Try changing threshold or min_area.")

pieces.sort(key=lambda piece: piece[1])
rows, cur = [], [pieces[0]]
for p in pieces[1:]:
    if abs(p[1] - cur[0][1]) < cur[0][3] * 0.5:
        cur.append(p)
    else:
        rows.append(sorted(cur, key=lambda piece: piece[0]))
        cur = [p]
rows.append(sorted(cur, key=lambda piece: piece[0]))

print(f"Detected {sum(len(r) for r in rows)} pieces in {len(rows)} rows")

piece_id = 0
for row in rows:
    for x, y, w, h, contour in row:
        crop_img = img[y:y+h, x:x+w]

        local_mask = np.zeros((h, w), dtype=np.uint8)
        shifted_contour = contour - np.array([[[x, y]]])
        cv2.drawContours(local_mask, [shifted_contour], -1, 255, thickness=cv2.FILLED)

        crop_bgra = cv2.cvtColor(crop_img, cv2.COLOR_BGR2BGRA)
        crop_bgra[:, :, 3] = local_mask

        out_path = templates_dir / f"piece_{piece_id:02d}.png"
        cv2.imwrite(str(out_path), crop_bgra)
        print(f"  Saved {out_path.name} ({w}×{h})")
        piece_id += 1


## Extract background from crab.webp
crab = cv2.imread(str(base_dir / "crab.webp"))
hsv = cv2.cvtColor(crab, cv2.COLOR_BGR2HSV)

wood_mask = cv2.inRange(hsv, np.array([8, 20, 140]), np.array([28, 130, 255]))
outside_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))

non_puzzle = cv2.bitwise_or(wood_mask, outside_mask)
puzzle_mask = cv2.bitwise_not(non_puzzle)

puzzle_mask = cv2.morphologyEx(puzzle_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

contours, _ = cv2.findContours(puzzle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(largest)

background = crab[y:y+h, x:x+w]
bg_path = base_dir / "background.png"
cv2.imwrite(str(bg_path), background)
print(f"\nBackground saved: {bg_path.name}  ({w}×{h})")
