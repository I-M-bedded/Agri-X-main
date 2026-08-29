# -*- coding: utf-8 -*-
"""Lightweight marker -> furrow round-trip -> next marker -> home mission FSM.

This is intentionally independent of ROS. The hardware stack is small enough
that a single deterministic 20 Hz state machine is easier to debug on a Pi 4.
The perception model runs in its own latest-frame worker, so slow AI inference
never blocks motor/ToF safety ticks.

New field convention used by this pipeline:
  * marker 0: HOME, facing the robot on the return headland direction
  * marker 1..N: furrow entry waypoints in travel order
  * marker FIELD_END_MARKER_ID: dedicated END marker after the last furrow

Each entry marker defines a fixed stop waypoint. The robot stops at a constant
stand-off distance, turns +/-90 degrees, then lets segmented furrow geometry and
side ToF absorb residual alignment error.
"""

from dataclasses import dataclass
from enum import Enum, auto
import math
import time

from config import (
    CONTROL_LOOP_DT,
    FIELD_END_MARKER_ID,
    FIELD_END_MARKER_MAX_BEARING_RAD,
    FIELD_END_MARKER_MAX_DISTANCE_M,
    HOME_MARKER_ID,
    PUMP_ON_RETURN_LEG,
)
from logutil import get_logger
from navigation.furrow_manager import FurrowManager
from sensors.odometry import normalize_angle

log = get_logger("agri-pipeline")


class PipelineState(Enum):
    INIT = auto()
    SEARCH_MARKER = auto()
    APPROACH_MARKER = auto()
    TURN_INTO_FURROW = auto()
    ACQUIRE_FURROW = auto()
    FOLLOW_OUTBOUND = auto()
    TURN_AT_END = auto()
    FOLLOW_RETURN = auto()
    EXIT_FURROW = auto()
    TURN_TO_HEADLAND = auto()
    SEARCH_NEXT_MARKER = auto()
    RETURN_HOME_TURN = auto()
    RETURN_HOME = auto()
    HOME_APPROACH = auto()
    MISSION_COMPLETE = auto()
    SAFE_HALT = auto()


@dataclass
class PipelineConfig:
    entry_turn_rad: float = math.pi / 2.0       # +left, -right
    marker_stop_distance_m: float = 0.60
    home_stop_distance_m: float = 0.70
    marker_lateral_tolerance_m: float = 0.10

    headland_search_speed: float = 0.18
    marker_approach_speed: float = 0.22
    furrow_speed: float = 0.28                   # conservative for ~2Hz AI on Pi4
    furrow_acquire_speed: float = 0.14
    exit_speed: float = 0.18
    home_speed: float = 0.20
    turn_speed: float = 0.30

    marker_bearing_kp: float = 0.9
    heading_hold_kp: float = 0.8
    max_headland_steer: float = 0.20

    marker_scan_every_n_ticks: int = 2          # 10Hz at a 20Hz control loop
    marker_cache_sec: float = 0.20
    marker_lost_sec: float = 0.8
    marker_search_timeout_sec: float = 60.0
    marker_approach_timeout_sec: float = 20.0

    furrow_min_confidence: float = 0.25
    furrow_acquire_timeout_sec: float = 5.0
    furrow_acquire_creep_m: float = 0.45
    furrow_min_leg_m: float = 1.0
    furrow_max_leg_sec: float = 90.0
    no_guidance_timeout_sec: float = 0.8
    exit_distance_m: float = 0.35
    exit_timeout_sec: float = 4.0

    ai_startup_timeout_sec: float = 20.0
    ai_stale_timeout_sec: float = 2.5
    require_ai_safety: bool = True
    tof_emergency_mm: float = 55.0

    turn_90_timeout_sec: float = 5.0
    turn_180_timeout_sec: float = 8.0
    home_return_timeout_sec: float = 180.0
    loop_dt_sec: float = CONTROL_LOOP_DT


class AgriPipelineFSM:
    MOVING_STATES = {
        PipelineState.SEARCH_MARKER,
        PipelineState.APPROACH_MARKER,
        PipelineState.TURN_INTO_FURROW,
        PipelineState.ACQUIRE_FURROW,
        PipelineState.FOLLOW_OUTBOUND,
        PipelineState.TURN_AT_END,
        PipelineState.FOLLOW_RETURN,
        PipelineState.EXIT_FURROW,
        PipelineState.TURN_TO_HEADLAND,
        PipelineState.SEARCH_NEXT_MARKER,
        PipelineState.RETURN_HOME_TURN,
        PipelineState.RETURN_HOME,
        PipelineState.HOME_APPROACH,
    }

    MARKER_STATES = {
        PipelineState.SEARCH_MARKER,
        PipelineState.APPROACH_MARKER,
        PipelineState.SEARCH_NEXT_MARKER,
        PipelineState.RETURN_HOME,
        PipelineState.HOME_APPROACH,
    }

    def __init__(
        self,
        camera,
        aruco,
        perception,
        tof,
        odom,
        motors,
        furrow_controller,
        pump=None,
        manager=None,
        config=None,
    ):
        self.camera = camera
        self.aruco = aruco
        self.perception = perception
        self.tof = tof
        self.odom = odom
        self.motors = motors
        self.furrow_controller = furrow_controller
        self.pump = pump
        self.manager = manager or FurrowManager()
        self.cfg = config or PipelineConfig()

        self.state = PipelineState.INIT
        self.halt_reason = ""
        self._state_enter = time.monotonic()
        self._tick = 0
        self._frame_fail_streak = 0
        self._marker_cache = {}
        self._marker_cache_time = 0.0
        self._last_target_seen = 0.0
        self._headland_heading = self.odom.theta
        self._home_heading = self.odom.theta
        self._exit_heading = self.odom.theta
        self._leg_start_path = self.odom.path_length
        self._acquire_start_path = self.odom.path_length
        self._exit_start_path = self.odom.path_length
        self._no_guidance_since = None

        self._turn_start_theta = 0.0
        self._turn_delta = 0.0
        self._turn_next = None
        self._turn_timeout = 0.0

    # ------------------------------------------------------------------
    def _goto(self, state):
        if state != self.state:
            log.info("state: %s -> %s", self.state.name, state.name)
        self.state = state
        self._state_enter = time.monotonic()

    def _elapsed(self):
        return time.monotonic() - self._state_enter

    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def _halt(self, reason):
        self.motors.stop()
        if self.pump is not None:
            self.pump.turn_off()
            self.pump.set_zone(False)
        self.halt_reason = str(reason)
        log.error("SAFE_HALT: %s", reason)
        self._goto(PipelineState.SAFE_HALT)

    # ------------------------------------------------------------------
    def _markers(self, frame):
        if self.state not in self.MARKER_STATES or frame is None:
            return {}

        every = max(1, self.cfg.marker_scan_every_n_ticks)
        if self._tick % every == 0:
            self._marker_cache = self.aruco.detect_from_frame(frame)
            self._marker_cache_time = time.monotonic()

        if time.monotonic() - self._marker_cache_time <= self.cfg.marker_cache_sec:
            return self._marker_cache
        return {}

    def _heading_hold(self, speed, heading):
        # theta > target means robot points too far CCW/left, so steer right (+).
        error = normalize_angle(self.odom.theta - heading)
        steer = self._clamp(
            self.cfg.heading_hold_kp * error,
            -self.cfg.max_headland_steer,
            self.cfg.max_headland_steer,
        )
        self.motors.drive(speed, steer)

    def _approach_observation(self, obs, stop_distance, done_state):
        if obs is None:
            self.motors.stop()
            if time.monotonic() - self._last_target_seen > self.cfg.marker_lost_sec:
                # Resume headland search instead of blindly advancing.
                if done_state == PipelineState.MISSION_COMPLETE:
                    self._goto(PipelineState.RETURN_HOME)
                else:
                    self._goto(
                        PipelineState.SEARCH_NEXT_MARKER
                        if self.manager.current_index
                        else PipelineState.SEARCH_MARKER
                    )
            return

        self._last_target_seen = time.monotonic()
        if self._elapsed() > self.cfg.marker_approach_timeout_sec:
            self._halt("marker approach timeout")
            return

        if obs.forward_m <= stop_distance:
            self.motors.stop()
            if abs(obs.lateral_offset_m) > self.cfg.marker_lateral_tolerance_m:
                self._halt(
                    "marker stand-off reached with excessive lateral error "
                    f"({obs.lateral_offset_m:+.2f}m)"
                )
                return
            if done_state == PipelineState.MISSION_COMPLETE:
                self._goto(PipelineState.MISSION_COMPLETE)
            else:
                self.manager.mark_attempt()
                self._begin_turn(
                    PipelineState.TURN_INTO_FURROW,
                    self.cfg.entry_turn_rad,
                    PipelineState.ACQUIRE_FURROW,
                )
            return

        bearing = math.atan2(obs.lateral_offset_m, max(0.05, obs.forward_m))
        steer = self._clamp(
            self.cfg.marker_bearing_kp * bearing,
            -self.cfg.max_headland_steer,
            self.cfg.max_headland_steer,
        )
        self.motors.drive(self.cfg.marker_approach_speed, steer)

    # ------------------------------------------------------------------
    def _begin_turn(self, state, delta, next_state):
        self.motors.stop()
        self._turn_start_theta = self.odom.theta
        self._turn_delta = float(delta)
        self._turn_next = next_state
        self._turn_timeout = (
            self.cfg.turn_180_timeout_sec
            if abs(delta) > math.pi * 0.75
            else self.cfg.turn_90_timeout_sec
        )
        self._goto(state)

    def _step_turn(self):
        progress = self.odom.theta - self._turn_start_theta
        reached = (
            progress >= self._turn_delta
            if self._turn_delta > 0
            else progress <= self._turn_delta
        )
        if reached:
            self.motors.stop()
            self._complete_turn(self._turn_next)
            return
        if self._elapsed() > self._turn_timeout:
            self._halt(
                "turn timeout: target=%+.1fdeg actual=%+.1fdeg"
                % (math.degrees(self._turn_delta), math.degrees(progress))
            )
            return
        self.motors.rotate_in_place(
            clockwise=self._turn_delta < 0,
            speed=self.cfg.turn_speed,
        )

    def _complete_turn(self, next_state):
        if next_state == PipelineState.ACQUIRE_FURROW:
            self._acquire_start_path = self.odom.path_length
            self.furrow_controller.reset()
            if self.pump is not None:
                self.pump.set_zone(False)
        elif next_state == PipelineState.FOLLOW_RETURN:
            self._leg_start_path = self.odom.path_length
            self.furrow_controller.reset()
            self._no_guidance_since = None
        elif next_state == PipelineState.SEARCH_NEXT_MARKER:
            self.manager.mark_current_done()
            self._headland_heading = self.odom.theta
        elif next_state == PipelineState.RETURN_HOME:
            self._home_heading = self.odom.theta
        self._goto(next_state)

    # ------------------------------------------------------------------
    def _global_safety(self, left_mm, right_mm):
        if self.state not in self.MOVING_STATES:
            return True

        # Side ToF is not the main obstacle detector, but an extremely small
        # distance is a useful independent collision stop in every state.
        if min(left_mm, right_mm) < self.cfg.tof_emergency_mm:
            self._halt("side ToF emergency clearance")
            return False

        if not self.cfg.require_ai_safety:
            return True
        if not self.perception.ready:
            self._halt("AI safety perception unavailable: %s" % self.perception.last_error)
            return False

        snap = self.perception.snapshot()
        if snap is None or self.perception.age_sec() > self.cfg.ai_stale_timeout_sec:
            self._halt("AI safety perception stale")
            return False
        if snap.obstacle_detected:
            self._halt(
                "obstacle in drive corridor: %s conf=%.2f overlap=%.3f"
                % (
                    snap.obstacle_label,
                    snap.obstacle_confidence,
                    snap.obstacle_corridor_overlap,
                )
            )
            return False
        return True

    # ------------------------------------------------------------------
    @staticmethod
    def _end_marker_is_close(obs):
        if obs is None:
            return False
        bearing = abs(math.atan2(obs.lateral_offset_m, max(0.05, obs.forward_m)))
        return (
            obs.distance_m <= FIELD_END_MARKER_MAX_DISTANCE_M
            and bearing <= FIELD_END_MARKER_MAX_BEARING_RAD
        )

    def _search_next(self, markers):
        target_id = self.manager.next_marker_id()
        end_obs = markers.get(FIELD_END_MARKER_ID)
        if self._end_marker_is_close(end_obs):
            log.info(
                "END marker observed after %d completed furrows",
                self.manager.total_completed(),
            )
            self._begin_turn(
                PipelineState.RETURN_HOME_TURN,
                math.pi,
                PipelineState.RETURN_HOME,
            )
            return

        target = markers.get(target_id)
        if target is not None:
            self.motors.stop()
            self._last_target_seen = time.monotonic()
            log.info("furrow marker %d acquired", target_id)
            self._goto(PipelineState.APPROACH_MARKER)
            return

        # Seeing a later numbered marker means the expected entrance was missed.
        later = [mid for mid in markers if target_id < mid < FIELD_END_MARKER_ID]
        if later:
            self._halt("missed expected marker %d; saw %d" % (target_id, min(later)))
            return

        if self._elapsed() > self.cfg.marker_search_timeout_sec:
            self._halt("marker search timeout for id %d" % target_id)
            return
        self._heading_hold(self.cfg.headland_search_speed, self._headland_heading)

    # ------------------------------------------------------------------
    def _follow_furrow(self, left_mm, right_mm, returning=False):
        snap = self.perception.snapshot()
        furrow = snap.furrow if snap is not None else None
        control = self.furrow_controller.step(furrow, left_mm, right_mm)
        travelled = self.odom.path_length - self._leg_start_path

        if control.furrow_end_detected and travelled >= self.cfg.furrow_min_leg_m:
            self.motors.stop()
            if self.pump is not None:
                self.pump.turn_off()
            if returning:
                self._exit_start_path = self.odom.path_length
                self._exit_heading = self.odom.theta
                if self.pump is not None:
                    self.pump.set_zone(False)
                self._goto(PipelineState.EXIT_FURROW)
            else:
                self._begin_turn(
                    PipelineState.TURN_AT_END,
                    math.pi,
                    PipelineState.FOLLOW_RETURN,
                )
            return

        if self._elapsed() > self.cfg.furrow_max_leg_sec:
            self._halt("furrow leg timeout")
            return

        if control.using_vision or control.using_tof:
            self._no_guidance_since = None
        else:
            if self._no_guidance_since is None:
                self._no_guidance_since = time.monotonic()
            elif time.monotonic() - self._no_guidance_since > self.cfg.no_guidance_timeout_sec:
                self._halt("both furrow vision and side ToF unavailable")
                return

        if self.pump is not None:
            self.pump.set_zone(True)
            if not returning or PUMP_ON_RETURN_LEG:
                self.pump.turn_on()
            else:
                self.pump.turn_off()
        self.motors.drive(self.cfg.furrow_speed, control.steer)

    # ------------------------------------------------------------------
    def step(self):
        if self.state in (PipelineState.MISSION_COMPLETE, PipelineState.SAFE_HALT):
            self.motors.stop()
            return

        self._tick += 1
        self.odom.update()
        left_mm, right_mm = self.tof.read()       # exactly one ToF sample per control tick
        if self.pump is not None:
            self.pump.tick()

        frame = self.camera.capture_frame()
        if frame is None:
            self._frame_fail_streak += 1
        else:
            self._frame_fail_streak = 0
            self.perception.submit(frame)
        if self._frame_fail_streak >= 5:
            self._halt("camera failed for 5 consecutive control ticks")
            return

        markers = self._markers(frame)

        # INIT waits without moving until the asynchronous safety model has
        # produced its first snapshot. After that, stale perception is fail-safe.
        if self.state == PipelineState.INIT:
            self.motors.stop()
            if self.cfg.require_ai_safety:
                if not self.perception.ready:
                    self._halt("AI safety perception unavailable: %s" % self.perception.last_error)
                    return
                if self.perception.snapshot() is None:
                    if self._elapsed() > self.cfg.ai_startup_timeout_sec:
                        self._halt("AI perception startup timeout")
                    return
            self._headland_heading = self.odom.theta
            self._goto(PipelineState.SEARCH_MARKER)
            return

        if not self._global_safety(left_mm, right_mm):
            return

        if self.state in (PipelineState.SEARCH_MARKER, PipelineState.SEARCH_NEXT_MARKER):
            self._search_next(markers)
            return

        if self.state == PipelineState.APPROACH_MARKER:
            obs = markers.get(self.manager.next_marker_id())
            self._approach_observation(
                obs,
                self.cfg.marker_stop_distance_m,
                PipelineState.ACQUIRE_FURROW,
            )
            return

        if self.state in (
            PipelineState.TURN_INTO_FURROW,
            PipelineState.TURN_AT_END,
            PipelineState.TURN_TO_HEADLAND,
            PipelineState.RETURN_HOME_TURN,
        ):
            self._step_turn()
            return

        if self.state == PipelineState.ACQUIRE_FURROW:
            snap = self.perception.snapshot()
            furrow = snap.furrow if snap is not None else None
            vision_ok = (
                furrow is not None
                and furrow.confidence >= self.cfg.furrow_min_confidence
            )
            tof_ok = self.tof.walls_visible()
            if vision_ok or tof_ok:
                self.furrow_controller.reset()
                self._leg_start_path = self.odom.path_length
                self._no_guidance_since = None
                self._goto(PipelineState.FOLLOW_OUTBOUND)
                return

            creep = self.odom.path_length - self._acquire_start_path
            if (
                creep > self.cfg.furrow_acquire_creep_m
                or self._elapsed() > self.cfg.furrow_acquire_timeout_sec
            ):
                self._halt("furrow could not be acquired after entry turn")
                return

            # Coarse 90deg turn need not be exact; weak segmentation can already
            # provide steering while creeping until the full furrow is acquired.
            control = self.furrow_controller.step(furrow, left_mm, right_mm)
            self.motors.drive(self.cfg.furrow_acquire_speed, control.steer)
            return

        if self.state == PipelineState.FOLLOW_OUTBOUND:
            self._follow_furrow(left_mm, right_mm, returning=False)
            return

        if self.state == PipelineState.FOLLOW_RETURN:
            self._follow_furrow(left_mm, right_mm, returning=True)
            return

        if self.state == PipelineState.EXIT_FURROW:
            travelled = self.odom.path_length - self._exit_start_path
            if travelled >= self.cfg.exit_distance_m:
                self.motors.stop()
                self._begin_turn(
                    PipelineState.TURN_TO_HEADLAND,
                    self.cfg.entry_turn_rad,
                    PipelineState.SEARCH_NEXT_MARKER,
                )
                return
            if self._elapsed() > self.cfg.exit_timeout_sec:
                self._halt("furrow exit timeout")
                return
            self._heading_hold(self.cfg.exit_speed, self._exit_heading)
            return

        if self.state == PipelineState.RETURN_HOME:
            home = markers.get(HOME_MARKER_ID)
            if home is not None:
                self.motors.stop()
                self._last_target_seen = time.monotonic()
                self._goto(PipelineState.HOME_APPROACH)
                return
            if self._elapsed() > self.cfg.home_return_timeout_sec:
                self._halt("HOME marker return timeout")
                return
            self._heading_hold(self.cfg.home_speed, self._home_heading)
            return

        if self.state == PipelineState.HOME_APPROACH:
            home = markers.get(HOME_MARKER_ID)
            self._approach_observation(
                home,
                self.cfg.home_stop_distance_m,
                PipelineState.MISSION_COMPLETE,
            )
            return

    # ------------------------------------------------------------------
    def run_forever(self):
        try:
            while self.state not in (
                PipelineState.MISSION_COMPLETE,
                PipelineState.SAFE_HALT,
            ):
                start = time.monotonic()
                self.step()
                remain = self.cfg.loop_dt_sec - (time.monotonic() - start)
                if remain > 0:
                    time.sleep(remain)
        finally:
            self.motors.stop()
            if self.pump is not None:
                self.pump.turn_off()
                self.pump.set_zone(False)

        return self.state == PipelineState.MISSION_COMPLETE
