import cv2
import numpy as np
import pyrealsense2 as rs
import serial
import time
import json
from pathlib import Path
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup

# ── Config ──────────────────────────────────────────────────────────────────
TEMPLATES_DIR = Path("image/templates_crab/live_templates")
POSES_FILE    = Path("poses.json")
PLACE_POSES_FILE = Path("place_poses.json")
MIN_AREA      = 1500
MIN_MATCHES   = 6
RATIO         = 0.75
PICK_Z_HIGH   = 0.12

# ── Load poses ───────────────────────────────────────────────────────────────
with open(POSES_FILE) as f:
    pick_poses = json.load(f)
with open(PLACE_POSES_FILE) as f:
    place_poses = json.load(f)
print(f"Pick poses:  {list(pick_poses.keys())}")
print(f"Place poses: {list(place_poses.keys())}")

# ── Camera ───────────────────────────────────────────────────────────────────
pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
for _ in range(30):
    pipeline.wait_for_frames()

# ── SIFT templates ───────────────────────────────────────────────────────────
sift = cv2.SIFT_create()
bf   = cv2.BFMatcher(cv2.NORM_L2)

tmpl_features = []
for tf in sorted(TEMPLATES_DIR.glob("piece_*.png")):
    idx  = tf.stem.split("_")[1]          # keep as string to match pose keys
    tmpl = cv2.imread(str(tf), cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(tmpl[:, :, :3], cv2.COLOR_BGR2GRAY)
    mask = (tmpl[:, :, 3] > 128).astype(np.uint8)
    kp, des = sift.detectAndCompute(gray, mask)
    tmpl_features.append((idx, kp, des))
    print(f"Loaded template piece_{idx}")

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# ── Pump ─────────────────────────────────────────────────────────────────────
pump = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)

def pump_on():
    pump.write(b'1')
    time.sleep(0.8)

def pump_off():
    pump.write(b'0')
    time.sleep(5.0)

# ── Robot ─────────────────────────────────────────────────────────────────────
robot = InterbotixManipulatorXS("px100", "arm", "gripper")
robot_startup()
robot.arm.go_to_home_pose()
robot.gripper.grasp(2.0)

# ── Robot actions ─────────────────────────────────────────────────────────────
def pick(name):
    if name not in pick_poses:
        print(f"No pick pose for piece {name}")
        return False
    p = pick_poses[name]
    x, y, z = p['x'], p['y'], p['z']
    print(f"Picking piece {name} at ({x}, {y}, {z})")
    if not robot.arm.set_ee_pose_components(x=x, y=y, z=PICK_Z_HIGH)[1]:
        print("Cannot reach above pick position")
        return False
    if not robot.arm.set_ee_pose_components(x=x, y=y, z=z)[1]:
        print("Cannot reach pick contact position")
        return False
    pump_on()
    robot.arm.set_ee_pose_components(x=x, y=y, z=PICK_Z_HIGH)
    return True

def place(name):
    if name not in place_poses:
        print(f"No place pose for piece {name}")
        pump_off()
        return
    p = place_poses[name]
    x, y, z = p['x'], p['y'], p['z']
    print(f"Placing piece {name} at ({x}, {y}, {z})")
    robot.arm.set_ee_pose_components(x=x, y=y, z=PICK_Z_HIGH)
    robot.arm.set_ee_pose_components(x=x, y=y, z=z)
    pump_off()
    robot.arm.set_ee_pose_components(x=x, y=y, z=PICK_Z_HIGH)
    robot.arm.go_to_home_pose()

# ── Detection helper ──────────────────────────────────────────────────────────
def detect_pieces(color):
    hsv        = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    piece_mask = (hsv[:, :, 1] > 40).astype(np.uint8) * 255
    piece_mask = cv2.morphologyEx(piece_mask, cv2.MORPH_OPEN,  kernel)
    piece_mask = cv2.morphologyEx(piece_mask, cv2.MORPH_CLOSE, kernel)
    gray_frame = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(piece_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    found = {}
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_mask = piece_mask[y:y+h, x:x+w]
        kp_roi, des_roi = sift.detectAndCompute(roi_gray, roi_mask)
        if des_roi is None or len(des_roi) < 2:
            continue
        best_matches, best_idx = -1, None
        for idx, kp_t, des_t in tmpl_features:
            if des_t is None or len(des_t) < 2:
                continue
            raw  = bf.knnMatch(des_roi, des_t, k=2)
            good = [m for m, n in raw if m.distance < RATIO * n.distance]
            if len(good) > best_matches:
                best_matches, best_idx = len(good), idx
        if best_matches >= MIN_MATCHES and best_idx is not None:
            found[best_idx] = (x, y, w, h)
    return found

# ── Main loop ─────────────────────────────────────────────────────────────────
solved = set()

print("\n=== Puzzle Solver ===")
print("Press 's' to scan and solve next piece")
print("Press 'r' to reset solved set")
print("Press 'q' to quit")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color  = np.asanyarray(frames.get_color_frame().get_data())
        display = color.copy()

        # Show what is currently visible
        found = detect_pieces(color)
        for idx, (x, y, w, h) in found.items():
            color_box = (0, 180, 0) if idx not in solved else (128, 128, 128)
            cv2.rectangle(display, (x, y), (x+w, y+h), color_box, 2)
            cv2.putText(display, f"piece {idx}", (x+3, y+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(display, f"Solved: {sorted(solved)}", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("Puzzle Solver", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('r'):
            solved.clear()
            print("Reset solved set")

        elif key == ord('s'):
            # Robot goes home so camera can see clearly
            robot.arm.go_to_home_pose()
            time.sleep(0.5)

            # Fresh scan
            frames = pipeline.wait_for_frames()
            color  = np.asanyarray(frames.get_color_frame().get_data())
            found  = detect_pieces(color)

            # Find first unsolved piece that has both pick and place pose
            target = None
            for idx in sorted(found.keys()):
                if idx not in solved and idx in pick_poses and idx in place_poses:
                    target = idx
                    break

            if target is None:
                print("No solvable piece found in view")
            else:
                print(f"Solving piece {target}")
                if pick(target):
                    place(target)
                    solved.add(target)
                    print(f"Piece {target} placed. Solved so far: {sorted(solved)}")
                else:
                    print(f"Failed to pick piece {target}")
                    robot.arm.go_to_home_pose()

finally:
    pump_off()
    pump.close()
    pipeline.stop()
    cv2.destroyAllWindows()
    robot_shutdown()