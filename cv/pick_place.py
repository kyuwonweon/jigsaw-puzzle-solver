import json
import serial
import time
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup

with open('poses.json', 'r') as f:
    poses = json.load(f)
print(f"Loaded poses: {list(poses.keys())}")

PLACE_X = 0.18
PLACE_Y = 0.15
PLACE_Z_HIGH = 0.12
PLACE_Z_LOW = 0.05
PICK_Z_HIGH = 0.12

pump = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)

robot = InterbotixManipulatorXS("px100", "arm", "gripper")
robot_startup()
robot.arm.go_to_home_pose()
robot.gripper.grasp(2.0)

def pump_on():
    pump.write(b'1')
    time.sleep(0.8)

def pump_off():
    pump.write(b'0')
    time.sleep(5.0)

def pick(name):
    if name not in poses:
        print(f"Unknown piece: {name}")
        print(f"Available: {list(poses.keys())}")
        return False
    p = poses[name]
    x, y, z = p['x'], p['y'], p['z']
    print(f"Picking '{name}' at ({x}, {y}, {z})")
    result = robot.arm.set_ee_pose_components(x=x, y=y, z=PICK_Z_HIGH)
    if not result[1]:
        print("Cannot reach above position")
        return False
    result = robot.arm.set_ee_pose_components(x=x, y=y, z=z)
    if not result[1]:
        print("Cannot reach contact position")
        return False
    pump_on()
    robot.arm.set_ee_pose_components(x=x, y=y, z=PICK_Z_HIGH)
    return True

def place():
    print(f"Placing at ({PLACE_X}, {PLACE_Y})")
    robot.arm.set_ee_pose_components(x=PLACE_X, y=PLACE_Y, z=PLACE_Z_HIGH)
    robot.arm.set_ee_pose_components(x=PLACE_X, y=PLACE_Y, z=PLACE_Z_LOW)
    pump_off()
    robot.arm.set_ee_pose_components(x=PLACE_X, y=PLACE_Y, z=PLACE_Z_HIGH)
    robot.arm.go_to_home_pose()
    print("Done")

try:
    while True:
        print(f"\nAvailable pieces: {list(poses.keys())}")
        name = input("Enter piece name to pick (or 'q' to quit): ").strip()
        if name == 'q':
            break
        if pick(name):
            place()
        else:
            print("Pick failed, going home")
            robot.arm.go_to_home_pose()
finally:
    pump_off()
    pump.close()
    robot_shutdown()