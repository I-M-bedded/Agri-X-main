# -*- coding: utf-8 -*-
"""Hardware-free deterministic smoke test for the lightweight mission FSM.

This is not a physics simulator. It verifies the intended high-level sequence:
marker 1 -> enter -> outbound -> U-turn -> return -> exit -> END -> HOME.
Use it before ROS/Gazebo so state-machine regressions are caught cheaply.
"""

from dataclasses import dataclass
from pathlib import Path
import math
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from navigation.agri_pipeline_fsm import AgriPipelineFSM, PipelineConfig, PipelineState
from sensors.ai_perception import FurrowEstimate, PerceptionSnapshot


@dataclass
class Obs:
    marker_id: int
    distance_m: float
    forward_m: float
    lateral_offset_m: float = 0.0
    yaw_error_rad: float = 0.0


class FakeCamera:
    def capture_frame(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)


class FakePerception:
    ready = True
    last_error = ""

    def submit(self, frame):
        pass

    def snapshot(self):
        return PerceptionSnapshot(
            timestamp=time.monotonic(),
            inference_sec=0.01,
            furrow=FurrowEstimate(0.0, 0.0, 0.95, 0.4),
            obstacle_detected=False,
        )

    def age_sec(self):
        return 0.0


class FakeOdom:
    def __init__(self):
        self.theta = 0.0
        self.path_length = 0.0
        self.left_dir = 1
        self.right_dir = 1
        self.total_ticks = 0
        self.motors = None

    def update(self):
        if self.motors is None:
            return
        left, right = self.motors.left, self.motors.right
        dt = 0.05
        v_scale = 0.8
        wheelbase = 0.22
        dl = left * v_scale * dt
        dr = right * v_scale * dt
        self.path_length += abs(0.5 * (dl + dr))
        self.theta += (dr - dl) / wheelbase
        if abs(left) + abs(right) > 1e-6:
            self.total_ticks += 1


class FakeMotors:
    def __init__(self, odom):
        self.left = 0.0
        self.right = 0.0
        odom.motors = self

    def set_speeds(self, left, right):
        self.left, self.right = float(left), float(right)

    def drive(self, base, steer):
        self.set_speeds(base + steer, base - steer)

    def forward(self, speed):
        self.set_speeds(speed, speed)

    def stop(self):
        self.set_speeds(0.0, 0.0)

    def rotate_in_place(self, clockwise=True, speed=0.3):
        if clockwise:
            self.set_speeds(speed, -speed)
        else:
            self.set_speeds(-speed, speed)


class FakeToF:
    def __init__(self):
        self.fsm = None
        self._out_streak = 0
        self._state_start_path = 0.0
        self._last_state = None
        self._walls = False

    def read(self):
        state = self.fsm.state
        if state != self._last_state:
            self._last_state = state
            self._state_start_path = self.fsm.odom.path_length
            self._out_streak = 0

        travelled = self.fsm.odom.path_length - self._state_start_path
        if state == PipelineState.ACQUIRE_FURROW:
            self._walls = True
        elif state in (PipelineState.FOLLOW_OUTBOUND, PipelineState.FOLLOW_RETURN):
            self._walls = travelled < 1.4
        else:
            self._walls = False

        if self._walls:
            self._out_streak = 0
            return 150.0, 150.0
        self._out_streak += 1
        return 800.0, 800.0

    def walls_visible(self):
        return self._walls

    def both_out_of_range(self):
        return self._out_streak >= 3

    def reset_end_detection(self):
        self._out_streak = 0


class FakeFurrowController:
    class Result:
        steer = 0.0
        using_vision = True
        using_tof = True
        furrow_end_detected = False

    def __init__(self, tof):
        self.tof = tof

    def reset(self):
        self.tof.reset_end_detection()

    def step(self, furrow, left, right):
        r = self.Result()
        r.furrow_end_detected = self.tof.both_out_of_range()
        r.using_tof = self.tof.walls_visible()
        return r


class FakePump:
    def tick(self):
        pass

    def set_zone(self, value):
        pass

    def turn_on(self):
        return True

    def turn_off(self):
        pass


class FakeAruco:
    def __init__(self):
        self.phase = "row1"
        self.approach_count = 0
        self.home_count = 0
        self.fsm = None

    def detect_from_frame(self, frame):
        state = self.fsm.state

        if state in (PipelineState.SEARCH_MARKER, PipelineState.APPROACH_MARKER):
            if self.phase != "row1":
                return {}
            self.approach_count += 1
            forward = max(0.55, 1.2 - 0.16 * self.approach_count)
            return {1: Obs(1, forward, forward)}

        if state == PipelineState.SEARCH_NEXT_MARKER:
            self.phase = "end"
            return {249: Obs(249, 1.0, 1.0)}

        if state in (PipelineState.RETURN_HOME, PipelineState.HOME_APPROACH):
            self.phase = "home"
            self.home_count += 1
            forward = max(0.65, 1.4 - 0.16 * self.home_count)
            return {0: Obs(0, forward, forward)}

        return {}


def main():
    camera = FakeCamera()
    perception = FakePerception()
    odom = FakeOdom()
    motors = FakeMotors(odom)
    tof = FakeToF()
    aruco = FakeAruco()
    controller = FakeFurrowController(tof)

    cfg = PipelineConfig(
        loop_dt_sec=0.0,
        marker_scan_every_n_ticks=1,
        marker_cache_sec=1.0,
        marker_search_timeout_sec=999.0,
        marker_approach_timeout_sec=999.0,
        furrow_max_leg_sec=999.0,
        home_return_timeout_sec=999.0,
        require_ai_safety=True,
        entry_turn_rad=math.pi / 2,
    )

    fsm = AgriPipelineFSM(
        camera=camera,
        aruco=aruco,
        perception=perception,
        tof=tof,
        odom=odom,
        motors=motors,
        furrow_controller=controller,
        pump=FakePump(),
        config=cfg,
    )
    tof.fsm = fsm
    aruco.fsm = fsm

    history = [fsm.state.name]
    for _ in range(2000):
        previous = fsm.state
        fsm.step()
        if fsm.state != previous:
            history.append(fsm.state.name)
        if fsm.state in (PipelineState.MISSION_COMPLETE, PipelineState.SAFE_HALT):
            break

    print(" -> ".join(history))
    if fsm.state != PipelineState.MISSION_COMPLETE:
        raise SystemExit("FAILED: %s (%s)" % (fsm.state.name, fsm.halt_reason))
    if fsm.manager.completed != [1]:
        raise SystemExit("FAILED: completed=%r" % (fsm.manager.completed,))
    print("PASS: one-furrow round trip and HOME return")


if __name__ == "__main__":
    main()
