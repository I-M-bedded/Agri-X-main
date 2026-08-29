# -*- coding: utf-8 -*-
"""Minimal ArUco-based furrow entry bring-up FSM.

This module intentionally does only one job:
  1) find a target ArUco marker along the headland,
  2) drive toward it while keeping it near the camera center,
  3) stop at a fixed stand-off distance,
  4) rotate a fixed +/- 90 degrees using encoder odometry,
  5) hand control over to the furrow-centre follower.

The marker is treated as installed navigation infrastructure.  Therefore the
marker position, stop distance and turn angle are part of the field setup
specification rather than quantities inferred online.
"""

import math
import time
from dataclasses import dataclass
from enum import Enum, auto

from logutil import get_logger

log = get_logger("aruco-entry")


@dataclass
class ArucoEntryConfig:
    target_marker_id: int = 1
    stop_distance_m: float = 0.60
    turn_angle_rad: float = math.pi / 2.0   # +CCW(left), -CW(right)

    approach_speed: float = 0.25
    search_speed: float = 0.20
    turn_speed: float = 0.30

    bearing_kp: float = 0.8
    max_steer: float = 0.20
    stop_lateral_tolerance_m: float = 0.08

    marker_lost_timeout_sec: float = 1.0
    search_timeout_sec: float = 20.0
    approach_timeout_sec: float = 20.0
    loop_dt_sec: float = 0.05


class EntryState(Enum):
    SEARCH = auto()
    APPROACH = auto()
    STOP = auto()
    TURN_90 = auto()
    DONE = auto()
    SAFE_HALT = auto()


class ArucoEntryFSM:
    """Small bring-up state machine for marker -> stop -> 90deg turn."""

    def __init__(self, motors, odom, aruco, config=None):
        self.motors = motors
        self.odom = odom
        self.aruco = aruco
        self.cfg = config or ArucoEntryConfig()

        self.state = EntryState.SEARCH
        self._state_enter_time = time.monotonic()
        self._last_seen_time = None
        self._last_obs = None
        self.halt_reason = ""

    def _goto(self, state: EntryState):
        if state != self.state:
            log.info("state: %s -> %s", self.state.name, state.name)
        self.state = state
        self._state_enter_time = time.monotonic()

    def _elapsed(self):
        return time.monotonic() - self._state_enter_time

    def _halt(self, reason: str):
        self.motors.stop()
        self.halt_reason = reason
        log.error("SAFE_HALT: %s", reason)
        self._goto(EntryState.SAFE_HALT)

    def step(self):
        self.odom.update()

        if self.state == EntryState.SEARCH:
            return self._search()
        if self.state == EntryState.APPROACH:
            return self._approach()
        if self.state == EntryState.STOP:
            self.motors.stop()
            self._goto(EntryState.TURN_90)
            return
        if self.state == EntryState.TURN_90:
            return self._turn_90()

    def _target_observation(self):
        observed = self.aruco.detect()
        obs = observed.get(self.cfg.target_marker_id)
        if obs is not None:
            self._last_obs = obs
            self._last_seen_time = time.monotonic()
        return obs

    def _search(self):
        obs = self._target_observation()
        if obs is not None:
            self.motors.stop()
            log.info(
                "marker %d acquired: forward=%.2fm lateral=%+.2fm",
                obs.marker_id, obs.forward_m, obs.lateral_offset_m,
            )
            self._goto(EntryState.APPROACH)
            return

        if self._elapsed() > self.cfg.search_timeout_sec:
            return self._halt("target marker search timeout")

        # Slow in-place scan. The camera only has to acquire the marker once;
        # after that the approach controller keeps it near image centre.
        self.motors.rotate_in_place(clockwise=True, speed=self.cfg.search_speed)

    def _approach(self):
        obs = self._target_observation()

        if obs is None:
            self.motors.stop()
            if (
                self._last_seen_time is None
                or time.monotonic() - self._last_seen_time > self.cfg.marker_lost_timeout_sec
            ):
                return self._halt("marker lost during approach")
            return

        if self._elapsed() > self.cfg.approach_timeout_sec:
            return self._halt("marker approach timeout")

        # Stop using only explicit installation geometry: fixed forward distance
        # plus a loose lateral tolerance. Remaining small error is intentionally
        # delegated to the furrow-centre follower after the 90deg turn.
        if (
            obs.forward_m <= self.cfg.stop_distance_m
            and abs(obs.lateral_offset_m) <= self.cfg.stop_lateral_tolerance_m
        ):
            self.motors.stop()
            log.info(
                "stand-off reached: forward=%.2fm lateral=%+.2fm",
                obs.forward_m, obs.lateral_offset_m,
            )
            self._goto(EntryState.STOP)
            return

        # Marker bearing in camera/robot frame. Positive lateral means the target
        # lies to the right, which matches MotorDriver.drive() steer convention.
        bearing = math.atan2(obs.lateral_offset_m, max(obs.forward_m, 0.05))
        steer = self.cfg.bearing_kp * bearing
        steer = max(-self.cfg.max_steer, min(self.cfg.max_steer, steer))
        self.motors.drive(self.cfg.approach_speed, steer)

    def _turn_90(self):
        self.motors.stop()
        ok = self.motors.turn_by_angle_blocking(
            self.cfg.turn_angle_rad,
            speed=self.cfg.turn_speed,
        )
        if not ok:
            return self._halt("90-degree turn did not reach encoder target")

        self.motors.stop()
        log.info(
            "entry turn complete: %+.1f deg. Handover to furrow follower.",
            math.degrees(self.cfg.turn_angle_rad),
        )
        self._goto(EntryState.DONE)

    def run(self):
        try:
            while self.state not in (EntryState.DONE, EntryState.SAFE_HALT):
                start = time.monotonic()
                self.step()
                remain = self.cfg.loop_dt_sec - (time.monotonic() - start)
                if remain > 0:
                    time.sleep(remain)
        finally:
            self.motors.stop()

        return self.state == EntryState.DONE
