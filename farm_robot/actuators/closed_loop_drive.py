# -*- coding: utf-8 -*-
"""
actuators/closed_loop_drive.py
-------------------------------
바퀴 속도 폐루프 구동 계층.

계층 구조 (상위 -> 하위)
  FSM / LineFollower      : drive(base_speed, steer)  - 무차원 [-1,1] (기존 그대로)
  ClosedLoopDrive (여기)  : 명령 -> 좌/우 목표 바퀴속도(m/s) -> 엔코더 실측과
                            비교해 PI 로 듀티 보정 (피드포워드 + PI)
  MotorDriver             : 듀티/방향/데드밴드 등 원시 PWM 출력

왜 이 구조인가
  - 개루프에서는 같은 듀티라도 배터리 전압·부하(흙 저항)에 따라 실제 속도가
    달라진다. 폐루프는 "명령 0.4 = 항상 같은 속도"를 만들어 PID 튜닝과
    거리 기반 로직(헤드랜드 이동 등)을 안정시킨다.
  - 상위 인터페이스는 MotorDriver 와 완전히 동일하게 유지한다.
    config.DRIVE_MODE 만 바꾸면 FSM 은 아무것도 몰라도 된다.
  - 엔코더가 죽으면 피드포워드(=기존 개루프)만 남는다. 더 나빠질 수 없다.

회전(turn_by_angle 등) 명령은 MotorDriver 의 검증된 블로킹 구현에 위임한다.
"""

import time

from actuators.motor_driver import MotorDriver
from config import (
    MAX_WHEEL_SPEED_MPS,
    WHEEL_PID_INTEGRAL_LIMIT,
    WHEEL_PID_KI,
    WHEEL_PID_KP,
)
from logutil import get_logger

log = get_logger("clc-drive")


class _WheelPI:
    """바퀴 1개분 피드포워드 + PI. 오차/출력 모두 무차원([-1,1] 명령 스케일)."""

    def __init__(self):
        self.integral = 0.0
        self._prev_time = None

    def reset(self):
        self.integral = 0.0
        self._prev_time = None

    def compute(self, command: float, measured_mps: float) -> float:
        target_mps = command * MAX_WHEEL_SPEED_MPS
        error = (target_mps - measured_mps) / MAX_WHEEL_SPEED_MPS

        now = time.monotonic()
        dt = 0.0 if self._prev_time is None else min(0.2, now - self._prev_time)
        self._prev_time = now

        self.integral += error * dt
        self.integral = max(-WHEEL_PID_INTEGRAL_LIMIT,
                            min(WHEEL_PID_INTEGRAL_LIMIT, self.integral))
        # 피드포워드(command) + PI 보정. 정지 명령이면 즉시 0 (적분 잔량 방지).
        if abs(command) < 1e-6:
            self.reset()
            return 0.0
        out = command + WHEEL_PID_KP * error + WHEEL_PID_KI * self.integral
        return max(-1.0, min(1.0, out))


class ClosedLoopDrive:
    """MotorDriver 와 동일한 인터페이스의 폐루프 래퍼."""

    def __init__(self, odometry=None):
        self._inner = MotorDriver(odometry=odometry)
        self.odom = odometry
        self._left_pi = _WheelPI()
        self._right_pi = _WheelPI()

    # --- 폐루프 경로: 매 제어 틱 호출되는 속도 명령 ---------------------
    def set_speeds(self, left_speed: float, right_speed: float):
        if self.odom is not None and self.odom.is_available():
            vl, vr = self.odom.wheel_v
            left_out = self._left_pi.compute(left_speed, vl)
            right_out = self._right_pi.compute(right_speed, vr)
        else:  # 엔코더 불가 -> 순수 피드포워드 (기존 개루프와 동일)
            left_out, right_out = left_speed, right_speed
        self._inner.set_speeds(left_out, right_out)
        # 상위가 참조하는 '명령값'은 보정 전 값으로 유지한다
        self._inner.last_left = left_speed
        self._inner.last_right = right_speed

    def drive(self, base_speed: float, steer: float):
        """부호 규약은 MotorDriver 와 동일: steer>0 = 오른쪽 조향."""
        left = base_speed + steer
        right = base_speed - steer
        peak = max(abs(left), abs(right))
        if peak > 1.0:  # 조향량 보존 스케일링 (MotorDriver.drive 와 동일 정책)
            excess = peak - 1.0
            shift = excess if base_speed >= 0 else -excess
            left -= shift
            right -= shift
        self.set_speeds(left, right)

    def forward(self, speed: float):
        self.set_speeds(speed, speed)

    def stop(self):
        self._left_pi.reset()
        self._right_pi.reset()
        self._inner.stop()

    # --- 회전/정리: 검증된 기존 구현에 위임 ------------------------------
    def rotate_in_place(self, *args, **kwargs):
        self._inner.rotate_in_place(*args, **kwargs)

    def turn_by_angle_blocking(self, *args, **kwargs):
        return self._inner.turn_by_angle_blocking(*args, **kwargs)

    def turn_180_blocking(self, *args, **kwargs):
        return self._inner.turn_180_blocking(*args, **kwargs)

    def cleanup(self):
        self.stop()
        self._inner.cleanup()

    # 상위 모듈이 참조하는 속성 통과
    @property
    def last_left(self):
        return self._inner.last_left

    @property
    def last_right(self):
        return self._inner.last_right


def create_drive(odometry=None):
    """config.DRIVE_MODE 에 따라 구동 계층을 생성한다."""
    from config import DRIVE_MODE

    if DRIVE_MODE == "serial_mega":
        # 상위(Pi5) -> 하위(아두이노 메가) USB. 속도 PID 는 메가가 돈다.
        from actuators.serial_motor_driver import SerialMotorDriver

        return SerialMotorDriver(odometry=odometry)
    if DRIVE_MODE == "closed_loop":
        return ClosedLoopDrive(odometry=odometry)
    if DRIVE_MODE != "open_loop":
        raise ValueError(f"지원하지 않는 DRIVE_MODE: {DRIVE_MODE}")
    return MotorDriver(odometry=odometry)
