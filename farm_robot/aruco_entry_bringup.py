# -*- coding: utf-8 -*-
"""Run only the first-stage ArUco entry behaviour.

Usage examples:
    python3 aruco_entry_bringup.py
    python3 aruco_entry_bringup.py --marker 2 --turn right --stop 0.55

This does not run pump/ToF/furrow-following logic.  It is intentionally a
small hardware bring-up program for camera + ArUco + motors + encoders.
"""

import argparse
import math
import sys

from actuators.motor_driver import MotorDriver
from navigation.aruco_entry_fsm import ArucoEntryConfig, ArucoEntryFSM
from sensors.aruco_detector import ArucoDetector
from sensors.camera import Camera
from sensors.odometry import Odometry


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker", type=int, default=1, help="target ArUco marker ID")
    parser.add_argument(
        "--turn",
        choices=("left", "right"),
        default="left",
        help="furrow direction after reaching the marker",
    )
    parser.add_argument(
        "--stop",
        type=float,
        default=0.60,
        help="fixed marker stand-off distance in metres",
    )
    parser.add_argument("--approach-speed", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    camera = Camera()
    odom = Odometry()
    motors = MotorDriver(odometry=odom)
    aruco = ArucoDetector(camera)

    turn_angle = math.pi / 2.0 if args.turn == "left" else -math.pi / 2.0
    cfg = ArucoEntryConfig(
        target_marker_id=args.marker,
        stop_distance_m=args.stop,
        turn_angle_rad=turn_angle,
        approach_speed=args.approach_speed,
    )

    fsm = ArucoEntryFSM(motors=motors, odom=odom, aruco=aruco, config=cfg)

    try:
        ok = fsm.run()
        return 0 if ok else 2
    except KeyboardInterrupt:
        motors.stop()
        return 130
    finally:
        try:
            motors.cleanup()
        finally:
            try:
                odom.cleanup()
            finally:
                camera.close()


if __name__ == "__main__":
    sys.exit(main())
