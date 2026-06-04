# Jigsaw Puzzle Solver

An autonomous robotic system that detects, identifies, and places jigsaw puzzle pieces using computer vision and a 4-DOF robot arm with vacuum suction.

---

## Demo

<p align="center">
  <video src="./media/demo.mp4" width="100%" controls autoplay muted loop>
    Your browser does not support the video tag.
  </video>
</p>

---

## System Overview

```
Intel RealSense D435
        │  (live color feed)
        ▼
  Piece Detection          ← saturation thresholding + contour filtering
        │
  Piece Identification     ← HSV histogram + Hu moments + aspect ratio
        │
  Pick & Place Decision    ← confidence threshold + pose lookup
        │
  PX100 Robot Arm  ────────────────── Arduino (vacuum pump)
        │                                      │
   (servo joints)                       (on/off via serial)
```

### Hardware

| Component | Description |
|---|---|
| Intel RealSense D435 | RGB-D camera, mounted above the workspace |
| Interbotix PX100 | 4-DOF robot arm with vacuum end-effector |
| Vacuum pump + solenoid | Suction gripper, controlled by Arduino via serial |
| Arduino (any) | Receives `'1'`/`'0'` commands to toggle pump pin |
| White tray / table | Uniform background for piece segmentation |

### Software Dependencies

```bash
pip install opencv-python pyrealsense2 numpy scipy Pillow pyserial
```

> Interbotix ROS packages are also required for robot control.
> Follow the [Interbotix setup guide](https://docs.trossenrobotics.com/interbotix_xsarms_docs/).

---

## Repository Structure

```
jigsaw-puzzle-solver/
├── solve_puzzle.py             # Main autonomous solver (vision + robot)
├── interactive_pick_place.py   # Interactive teach-and-execute tool
├── poses_interactive.json      # Saved pick + place poses per piece
├── pump_valve.ino              # Arduino firmware for vacuum pump control
└── cv/
    ├── capture_templates.py    # Live template capture via RealSense
    ├── extract_templates.py    # Extract templates from reference images
    ├── live_match.py           # Live piece detection & matching (debug)
    ├── match_templates.py      # Offline SIFT + NCC template matching
    ├── pick_place.py           # Simple pick/place runner (poses.json)
    ├── teach.py                # Teach mode: record pick positions only
    ├── realsense_capture.py    # RealSense camera stream viewer
    ├── edge_detection.py       # Canny edge detection pipeline (standalone)
    └── image/
        └── templates_crab/
            ├── crab.webp           # Reference image of solved puzzle
            ├── splitted.webp       # Reference image of all pieces separated
            └── live_templates/     # PNG templates captured from camera
```

---

## How to Start

### Step 0 — Flash the Arduino

Upload `pump_valve.ino` to your Arduino. Pin 7 controls the pump relay:
- Serial `'1'` → pump ON
- Serial `'0'` → pump OFF

---

### Step 1 — Prepare Reference Images

Place two images in `cv/image/templates_crab/`:

| File | Content |
|---|---|
| `crab.webp` | Photo of the **completed** puzzle on the tray |
| `splitted.webp` | Photo of all **separated** pieces laid out flat |

Then extract templates and the background crop:

```bash
cd cv
python3 extract_templates.py
```

This saves individual piece PNGs to `cv/image/templates_crab/templates/` and a background crop to `cv/image/templates_crab/background.png`.

---

### Step 2 — Capture Live Templates

Place each piece one at a time in front of the RealSense camera on a white background:

```bash
cd cv
python3 capture_templates.py
```

| Key | Action |
|---|---|
| `0`–`9` | Save current piece as that number |
| `a` / `b` | Save as piece 10 / 11 |
| `q` | Quit |

Templates are saved to `cv/image/templates_crab/live_templates/`.

---

### Step 3 — Verify Detection (Optional)

Run the live matcher to confirm pieces are being recognized correctly:

```bash
cd cv
python3 live_match.py
```

Green box = confident match (score < 0.35). Yellow box = uncertain. Top-3 candidates are shown below each box.

---

### Step 4 — Teach Robot Poses

Use the interactive teach tool to record both pick and place positions for each piece:

```bash
python3 interactive_pick_place.py
```

For each piece:
1. Enter the piece number
2. **Pick position**: press `t` (torque off) → drag arm over the piece → press `e` (lock) → press `r` (record)
3. **Place position**: same process for the destination slot
4. Press `SPACE` twice to execute the pick-and-place and verify
5. Poses are saved automatically to `poses_interactive.json`

---

### Step 5 — Run the Autonomous Solver

```bash
python3 solve_puzzle.py
```

The solver shows a live camera feed with piece detections overlaid.

| Key | Action |
|---|---|
| `s` | Scan the scene and solve the next detected piece |
| `r` | Reset the solved-pieces tracking set |
| `q` | Quit |

The system takes 5 frames when scanning, keeps only pieces detected in 3+ frames (stable detections), then picks and places the first unsolved piece it finds.

---

## Additional Tools

| Script | Purpose |
|---|---|
| `cv/match_templates.py` | Offline SIFT + NCC matching; outputs `match_result.png` showing bounding boxes on the solved puzzle image |
| `cv/teach.py` | Minimal teach mode — records pick positions only, saves to `poses.json` |
| `cv/pick_place.py` | Simple runner that reads `poses.json` and places pieces at a fixed destination |
| `cv/realsense_capture.py` | View aligned color + depth streams from the RealSense |
| `cv/edge_detection.py` | Standalone Canny edge detection pipeline demo (Gaussian → gradient → NMS → hysteresis) |
