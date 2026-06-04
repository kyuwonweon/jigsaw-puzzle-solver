"""Module for interactively teaching pick and place poses and recording positions."""
import json
import sys
import termios
import time
import tty

from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup

from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

import serial

robot = InterbotixManipulatorXS('px100', 'arm', 'gripper')
robot_startup()
robot.arm.go_to_home_pose()
robot.gripper.grasp(2.0)

pump = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)
print('Pump connected')

PICK_Z_HIGH = 0.12
poses = {}


def get_key():
    """Get a single key press from the user."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def get_pos():
    """Get the current end-effector position."""
    T = robot.arm.get_ee_pose()
    return round(T[0, 3], 4), round(T[1, 3], 4), round(T[2, 3], 4)


def torque_off():
    """Turn off robot torque to allow free movement."""
    robot.core.robot_torque_enable('group', 'arm', False)
    print('Torque OFF - drag arm to position')


def torque_on():
    """Turn on robot torque to lock in place."""
    robot.core.robot_torque_enable('group', 'arm', True)
    x, y, z = get_pos()
    print(f'Torque ON - locked at ({x}, {y}, {z})')


def record_pos():
    """Record the current end-effector position."""
    T = robot.arm.get_ee_pose()
    return {
        'x': round(T[0, 3], 4),
        'y': round(T[1, 3], 4),
        'z': round(T[2, 3], 4)
    }


try:
    while True:
        print('\n' + '=' * 40)
        name = input('Enter piece number (or q to quit): ').strip()
        if name == 'q':
            break

        print(f'\n--- PICK position for piece {name} ---')
        print("Press 't' torque off, drag arm to pick position")
        print("Press 'e' torque on to lock")
        print("Press 'r' to record pick position")

        pick_pos = None
        while pick_pos is None:
            key = get_key()
            if key == 't':
                torque_off()
            elif key == 'e':
                torque_on()
            elif key == 'r':
                pick_pos = record_pos()
                print(
                    f"Pick recorded: ({pick_pos['x']}, {pick_pos['y']}, {pick_pos['z']})"
                )

        print(f'\n--- PLACE position for piece {name} ---')
        print("Press 't' torque off, drag arm to place position")
        print("Press 'e' torque on to lock")
        print("Press 'r' to record place position")

        place_pos = None
        while place_pos is None:
            key = get_key()
            if key == 't':
                torque_off()
            elif key == 'e':
                torque_on()
            elif key == 'r':
                place_pos = record_pos()
                print(
                    f'Place recorded: ({place_pos["x"]}, {place_pos["y"]}, {place_pos["z"]})'
                )

        print(f'\n--- Executing piece {name} ---')
        torque_on()
        robot.arm.go_to_home_pose()

        print('Moving above pick position...')
        result = robot.arm.set_ee_pose_components(
            x=pick_pos['x'], y=pick_pos['y'], z=PICK_Z_HIGH)
        if not result[1]:
            print('Cannot reach above pick, trying direct...')

        print('Moving to pick position...')
        result = robot.arm.set_ee_pose_components(
            x=pick_pos['x'], y=pick_pos['y'], z=pick_pos['z'])
        if not result[1]:
            print('WARNING: could not reach exact pick position')

        print('\nPress SPACE to start suction, press SPACE again to stop and lift')
        while get_key() != ' ':
            pass
        pump.write(b'1')
        print('Suction ON - press SPACE when ready to lift')
        while get_key() != ' ':
            pass

        print('Lifting...')
        robot.arm.set_ee_pose_components(
            x=pick_pos['x'], y=pick_pos['y'], z=PICK_Z_HIGH)

        print('Moving to place position...')
        robot.arm.set_ee_pose_components(
            x=place_pos['x'], y=place_pos['y'], z=PICK_Z_HIGH)
        robot.arm.set_ee_pose_components(
            x=place_pos['x'], y=place_pos['y'], z=place_pos['z'])

        print('\nPress SPACE to release suction')
        while get_key() != ' ':
            pass
        pump.write(b'0')
        print('Suction OFF - press SPACE when piece has released and ready to lift')
        while get_key() != ' ':
            pass

        robot.arm.set_ee_pose_components(
            x=place_pos['x'], y=place_pos['y'], z=PICK_Z_HIGH)
        robot.arm.go_to_home_pose()
        print(f'Piece {name} done!')

        poses[name] = {'pick': pick_pos, 'place': place_pos}
        with open('poses_interactive.json', 'w') as f:
            json.dump(poses, f, indent=2)
        print('Saved to poses_interactive.json')

finally:
    pump.write(b'0')
    pump.close()
    robot_shutdown()
