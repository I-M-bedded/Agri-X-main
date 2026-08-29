# -*- coding: utf-8 -*-
"""Entrypoint for the lightweight non-ROS agricultural mission pipeline.

Recommended Pi 4 deployment:
    python3 tools/export_yoloe_rpi.py --output models/agri_yoloe26n_ncnn_model
    python3 pipeline_main.py --model models/agri_yoloe26n_ncnn_model

For workstation testing, the public checkpoint may be used directly:
    python3 pipeline_main.py --model yoloe-26n-seg.pt
"""

import argparse
import math
import sys

from actuators.motor_driver import MotorDriver
from actuators.pump_controller import PumpController
from config import TOF_LEFT, TOF_RIGHT
from control.furrow_ai_controller import AIFurrowController
from navigation.agri_pipeline_fsm import AgriPipelineFSM, PipelineConfig
from sensors.ai_perception import ZeroShotFieldPerception
from sensors.camera import Camera
from sensors.frame_aruco import FrameArucoDetector
from sensors.odometry import Odometry
from sensors.tof_sensor import ToFPair


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="models/agri_yoloe26n_ncnn_model",
        help="baked YOLOE NCNN directory or yoloe-26n-seg.pt for development",
    )
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--inference-hz", type=float, default=2.0)
    p.add_argument("--turn", choices=("left", "right"), default="left")
    p.add_argument("--marker-stop", type=float, default=0.60)
    p.add_argument("--furrow-speed", type=float, default=0.28)
    p.add_argument("--no-pump", action="store_true")
    p.add_argument(
        "--unsafe-no-ai",
        action="store_true",
        help="bring-up only: permit motion without the always-on AI safety watchdog",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # OpenCV thread oversubscription competes badly with NCNN/PyTorch on a 4-core Pi.
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        pass

    camera = Camera()
    odom = Odometry()
    motors = MotorDriver(odometry=odom)
    tof = ToFPair(TOF_LEFT, TOF_RIGHT)
    aruco = FrameArucoDetector(camera)
    pump = None if args.no_pump else PumpController()
    perception = ZeroShotFieldPerception(
        model_path=args.model,
        imgsz=args.imgsz,
        inference_hz=args.inference_hz,
        cpu_threads=3,
    )
    controller = AIFurrowController(tof_pair=tof, odometry=odom)

    turn_rad = math.pi / 2.0 if args.turn == "left" else -math.pi / 2.0
    cfg = PipelineConfig(
        entry_turn_rad=turn_rad,
        marker_stop_distance_m=args.marker_stop,
        furrow_speed=args.furrow_speed,
        require_ai_safety=not args.unsafe_no_ai,
    )

    fsm = AgriPipelineFSM(
        camera=camera,
        aruco=aruco,
        perception=perception,
        tof=tof,
        odom=odom,
        motors=motors,
        furrow_controller=controller,
        pump=pump,
        config=cfg,
    )

    try:
        ok = fsm.run_forever()
        return 0 if ok else 2
    except KeyboardInterrupt:
        motors.stop()
        return 130
    finally:
        try:
            perception.close()
        finally:
            try:
                if pump is not None:
                    pump.cleanup()
            finally:
                try:
                    tof.close()
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
