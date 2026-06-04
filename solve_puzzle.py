import cv2
import numpy as np
import pyrealsense2 as rs
import serial
import time
import json
from pathlib import Path
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup

TEMPLATES_DIR = Path('cv/image/templates_crab/live_templates')
POSES_FILE = Path('poses_interactive.json')
MIN_AREA = 2000
CONFIDENCE_THRESH = 0.35
PICK_Z_HIGH = 0.12
TOP_N = 3

with open(POSES_FILE) as f:
    all_poses = json.load(f)
print(f'Loaded poses for pieces: {sorted(all_poses.keys())}')

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
    if cnts:
        bx, by, bw, bh = cv2.boundingRect(cnts[0])
        aspect = bw / bh if bh > 0 else 1.0
    else:
        aspect = 1.0

    templates.append({
        'idx': idx, 'h_hist': h_hist, 's_hist': s_hist,
        'hu': hu, 'aspect': aspect,
    })

print(f'Loaded {len(templates)} templates')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

pipeline = rs.pipeline()
rs_config = rs.config()
rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(rs_config)
for _ in range(30):
    pipeline.wait_for_frames()

pump = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)
print('Pump connected')

robot = InterbotixManipulatorXS('px100', 'arm', 'gripper')
robot_startup()
robot.arm.go_to_home_pose()
robot.gripper.grasp(2.0)
print('Robot ready')


def match_piece(roi_bgr, roi_mask):
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
        return None, 999, []
    bx, by, bw, bh = cv2.boundingRect(cnts[0])
    aspect = bw / bh if bh > 0 else 1.0

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


def detect_pieces(color):
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    piece_mask = (hsv[:, :, 1] > 40).astype(np.uint8) * 255
    piece_mask = cv2.morphologyEx(piece_mask, cv2.MORPH_OPEN, kernel)
    piece_mask = cv2.morphologyEx(piece_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        piece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    found = {}
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if x <= 5 or y <= 5 or x+w >= 635 or y+h >= 475:
            continue

        roi_bgr = color[y:y+h, x:x+w]
        roi_mask = piece_mask[y:y+h, x:x+w]

        best_idx, best_score, top3 = match_piece(roi_bgr, roi_mask)
        if best_idx is None or best_score > CONFIDENCE_THRESH:
            continue

        if best_idx in found:
            if best_score < found[best_idx]['score']:
                found[best_idx] = {'box': (x, y, w, h), 'score': best_score}
        else:
            found[best_idx] = {'box': (x, y, w, h), 'score': best_score}

    return found


def pick(name):
    if name not in all_poses:
        print(f'No pose for piece {name}')
        return False
    p = all_poses[name]['pick']
    x, y, z = p['x'], p['y'], p['z']
    print(f'  Moving above piece {name}')
    if not robot.arm.set_ee_pose_components(x=x, y=y, z=PICK_Z_HIGH)[1]:
        print('  Cannot reach above position')
        return False
    print(f'  Descending to z={z:.4f}')
    if not robot.arm.set_ee_pose_components(x=x, y=y, z=z)[1]:
        print('  Cannot reach contact position')
        return False
    print('  Pump ON')
    pump.write(b'1')
    time.sleep(1.0)
    print('  Lifting')
    robot.arm.set_ee_pose_components(x=x, y=y, z=PICK_Z_HIGH)
    return True


def place(name):
    p = all_poses[name]['place']
    x, y, z = p['x'], p['y'], p['z']
    print('  Moving to place position')
    robot.arm.set_ee_pose_components(x=x, y=y, z=PICK_Z_HIGH)
    robot.arm.set_ee_pose_components(x=x, y=y, z=z)
    print('  Pump OFF')
    pump.write(b'0')
    time.sleep(10.0)
    robot.arm.set_ee_pose_components(x=x, y=y, z=PICK_Z_HIGH)
    robot.arm.go_to_home_pose()
    print(f'  Piece {name} placed')


solved = set()

print('\n=== Puzzle Solver ===')
print('s : scan and solve next piece')
print('r : reset solved set')
print('q : quit')

try:
    while True:
        frames = pipeline.wait_for_frames()
        color = np.asanyarray(frames.get_color_frame().get_data())
        display = color.copy()

        found = detect_pieces(color)
        for idx, info in found.items():
            x, y, w, h = info['box']
            score = info['score']
            col = (128, 128, 128) if idx in solved else (0, 255, 0)
            cv2.rectangle(display, (x, y), (x+w, y+h), col, 2)
            status = 'done' if idx in solved else f'{score:.2f}'
            cv2.putText(display, f'{idx} ({status})', (x+3, y+16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        remaining = [k for k in all_poses if k not in solved]
        cv2.putText(
            display,
            f'Solved: {len(solved)}/{len(all_poses)}  Remaining: {remaining}',
            (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1
        )
        cv2.imshow('Puzzle Solver', display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('r'):
            solved.clear()
            print('Reset')

        elif key == ord('s'):
            robot.arm.go_to_home_pose()
            time.sleep(0.8)

            detections = {}
            for _ in range(5):
                frames = pipeline.wait_for_frames()
                color = np.asanyarray(frames.get_color_frame().get_data())
                found = detect_pieces(color)
                for idx, info in found.items():
                    if idx not in detections:
                        detections[idx] = []
                    detections[idx].append(info['score'])

            stable = {k: np.mean(v) for k, v in detections.items() if len(v) >= 3}
            print(f'Stable detections: {stable}')

            target = None
            for idx in sorted(stable.keys()):
                if idx not in solved and idx in all_poses:
                    target = idx
                    break

            if target is None:
                print('No solvable piece found')
            else:
                print(f'\nSolving piece {target} (score={stable[target]:.3f})')
                if pick(target):
                    place(target)
                    solved.add(target)
                    print(f'Done. Solved: {sorted(solved)}')
                else:
                    print('Pick failed')
                    pump.write(b'0')
                    robot.arm.go_to_home_pose()

finally:
    pump.write(b'0')
    pump.close()
    pipeline.stop()
    cv2.destroyAllWindows()
    robot_shutdown()
