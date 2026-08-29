# -*- coding: utf-8 -*-
"""Closed-loop furrow controller for the zero-shot segmentation pipeline."""

from dataclasses import dataclass

from config import (
    HEADING_HOLD_GAIN,
    LINE_PID_D_FILTER_HZ,
    LINE_PID_INTEGRAL_LIMIT,
    LINE_PID_KD,
    LINE_PID_KI,
    LINE_PID_KP,
    MAX_STEER_CORRECTION,
    SIGN_HEADING_ERROR,
    SIGN_TOF_ERROR,
    TOF_ASSIST_WEIGHT,
    TOF_NOMINAL_WALL_DISTANCE_MM,
    VISION_HEADING_WEIGHT,
    VISION_MIN_CONFIDENCE,
)
from control.pid_controller import PIDController
from sensors.ai_perception import FurrowEstimate
from sensors.odometry import normalize_angle


@dataclass(frozen=True)
class FurrowControlResult:
    steer: float
    error: float
    using_vision: bool
    using_tof: bool
    furrow_end_detected: bool


class AIFurrowController:
    """Combine segmented centre line, side ToF and encoder heading hold."""

    def __init__(self, tof_pair, odometry=None):
        self.tof = tof_pair
        self.odom = odometry
        self.pid = PIDController(
            LINE_PID_KP,
            LINE_PID_KI,
            LINE_PID_KD,
            output_limit=MAX_STEER_CORRECTION,
            integral_limit=LINE_PID_INTEGRAL_LIMIT,
            d_filter_hz=LINE_PID_D_FILTER_HZ,
        )
        self._target_heading = 0.0

    def reset(self):
        self.pid.reset()
        self.tof.reset_end_detection()
        if self.odom is not None:
            self._target_heading = self.odom.theta

    def step(self, furrow: FurrowEstimate, left_mm: float, right_mm: float) -> FurrowControlResult:
        vision_valid = furrow is not None and furrow.confidence >= VISION_MIN_CONFIDENCE
        tof_valid = self.tof.walls_visible()

        tof_diff = (right_mm - left_mm) * SIGN_TOF_ERROR
        tof_error = tof_diff / max(1.0, TOF_NOMINAL_WALL_DISTANCE_MM)
        tof_error = max(-1.5, min(1.5, tof_error))

        if vision_valid:
            vision_error = furrow.normalized_error + VISION_HEADING_WEIGHT * furrow.heading_error
            error = vision_error
            if tof_valid:
                error = (1.0 - TOF_ASSIST_WEIGHT) * vision_error + TOF_ASSIST_WEIGHT * tof_error
            if self.odom is not None:
                self._target_heading = self.odom.theta
        elif tof_valid:
            error = tof_error
        else:
            error = 0.0
            if self.odom is not None:
                heading_error = normalize_angle(self.odom.theta - self._target_heading)
                error = SIGN_HEADING_ERROR * HEADING_HOLD_GAIN * heading_error

        error = max(-1.5, min(1.5, error))
        steer = self.pid.compute(error)
        return FurrowControlResult(
            steer=steer,
            error=error,
            using_vision=vision_valid,
            using_tof=tof_valid,
            furrow_end_detected=self.tof.both_out_of_range(),
        )
