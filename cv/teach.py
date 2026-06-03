import json
import sys
import tty
import termios
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup

robot = InterbotixManipulatorXS("px100", "arm", "gripper")
robot_startup()
robot.arm.go_to_home_pose()
robot.gripper.grasp(2.0)

poses = {}

print("=== Teach Mode ===")
print("t   : torque OFF  (free move - grab and drag arm)")
print("e   : torque ON   (lock in place)")
print("r   : record current pose, enter name")
print("h   : go home (torque must be ON first)")
print("p   : print all recorded poses")
print("q   : save and quit")

def get_pos():
    T = robot.arm.get_ee_pose()
    return round(T[0,3], 4), round(T[1,3], 4), round(T[2,3], 4)

def get_joints():
    return robot.arm.get_joint_commands()

def get_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

torque_on = True
x, y, z = get_pos()
print(f"Start pos: ({x}, {y}, {z})")
print("Ready:")

try:
    while True:
        key = get_key()

        if key == 't':
            # Torque off: arm can be moved by hand
            robot.core.robot_torque_enable("group", "arm", False)
            torque_on = False
            print("Torque OFF - grab and drag arm to position")

        elif key == 'e':
            # Torque on: lock arm in place
            robot.core.robot_torque_enable("group", "arm", True)
            torque_on = True
            x, y, z = get_pos()
            print(f"Torque ON - locked at ({x}, {y}, {z})")

        elif key == 'r':
            if torque_on:
                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
                name = input("\nEnter piece name (e.g. 0, 1, 2...): ").strip()
                x, y, z = get_pos()
                joints = get_joints()
                poses[name] = {
                    'x': x, 'y': y, 'z': z,
                    'joints': list(joints)
                }
                print(f"Recorded '{name}': ({x}, {y}, {z})")
                print(f"All poses: {list(poses.keys())}")
            else:
                print("Turn torque ON first (press e)")

        elif key == 'h':
            if torque_on:
                robot.arm.go_to_home_pose()
                print("home")
            else:
                print("Turn torque ON first (press e)")

        elif key == 'p':
            print(json.dumps(poses, indent=2))

        elif key == 'q':
            break

finally:
    # Make sure torque is on before quitting
    robot.core.robot_torque_enable("group", "arm", True)
    with open('poses.json', 'w') as f:
        json.dump(poses, f, indent=2)
    print(f"\nSaved {len(poses)} poses to poses.json")
    print(json.dumps(poses, indent=2))
    robot_shutdown()