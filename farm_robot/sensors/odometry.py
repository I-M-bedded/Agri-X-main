# -*- coding: utf-8 -*-
"""
sensors/odometry.py
-------------------
엔코더(+선택적으로 IMU)로 로봇의 위치(x, y)와 방향(theta)을 추정한다.

부호 규약 (config.py 참조)
  theta : 반시계(CCW, 좌회전)가 **양수**.
          d_theta = (d_right - d_left) / WHEEL_BASE 이므로
          우륜이 빠르면 좌회전 -> theta 증가. 표준 규약과 일치한다.

틱 소스 두 가지 (config.ENCODER_QUAD_PINS 로 선택)
  1) 쿼드러처 A/B (13 PPR × 4분주): 방향을 위상에서 직접 안다. 권장.
  2) 단일 채널(레거시): 방향을 모터 명령(left_dir/right_dir)으로 추정한다.
     쿼드러처 핀맵이 확정되기 전까지의 폴백.

클래스 구성
  Odometry          엔코더만 사용 (theta 도 엔코더 차동으로 적분)
  ImuFusedOdometry  거리=엔코더, theta=IMU 자이로 적분.
                    궤도 미끄러짐이 커도 회전각이 정확해진다.
                    IMU 가 죽으면 자동으로 엔코더 theta 로 폴백.
  create_odometry() config.ODOMETRY_BACKEND 에 따라 위 둘 중 하나 생성.

공개 API (기존과 동일 - FSM/모터/셀프테스트가 그대로 쓴다)
  x, y, theta, v, omega, path_length, total_ticks, wheel_v
  update() / reset() / inject_ticks() / is_available() / cleanup()
"""

import math
import threading
import time

from config import (
    DISTANCE_CALIBRATION_FACTOR,
    ENCODER_BOUNCETIME_MS,
    ENCODER_DECODE,
    ENCODER_EDGE,
    ENCODER_PINS,
    ENCODER_QUAD_PINS,
    ENCODER_QUAD_TICKS_PER_REVOLUTION,
    ODOMETRY_SOURCE,
    SIGN_LEFT_ENCODER,
    SIGN_RIGHT_ENCODER,
    TICKS_PER_REVOLUTION,
    WHEEL_BASE_M,
    WHEEL_RADIUS_M,
)
from logutil import get_logger

log = get_logger("odometry")

try:
    import RPi.GPIO as GPIO

    _HAS_GPIO = True
except ImportError:
    _HAS_GPIO = False


def normalize_angle(angle: float) -> float:
    """각도를 -pi ~ +pi 범위로 정규화."""
    return math.atan2(math.sin(angle), math.cos(angle))


class Odometry:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0          # 라디안, CCW(좌회전) 양수, 누적값(랩핑 안 함)
        self.path_length = 0.0    # 누적 이동 거리(m), 항상 증가
        self.v = 0.0              # 직진 속도 추정 (m/s)
        self.omega = 0.0          # 각속도 추정 (rad/s)
        self.wheel_v = (0.0, 0.0)  # 좌/우 바퀴 선속도 (m/s) - 폐루프 구동용

        self._lock = threading.Lock()
        self._left_ticks = 0
        self._right_ticks = 0
        self._left_distance_m = 0.0
        self._right_distance_m = 0.0
        self.total_ticks = 0      # 엔코더 생존 확인용 (회전 stall 감지)

        # 단일 채널 엔코더용: 모터 드라이버가 지시 방향을 알려준다.
        # 쿼드러처 모드에서는 위상이 방향을 주므로 이 값은 쓰이지 않는다.
        self.left_dir = 1
        self.right_dir = 1

        self._quad_left = None
        self._quad_right = None
        self._gpio_ready = False
        self._external_ready = ODOMETRY_SOURCE == "mega_usb"

        # [궤도 보정] 엔코더는 '궤도가 움직인 양'을 잰다. 흙에서 헛돌면
        #   과대평가되므로 DISTANCE_CALIBRATION_FACTOR 로 실측 보정한다
        #   (tools/setup.py 4번).
        ticks_per_rev = TICKS_PER_REVOLUTION
        if self._external_ready:
            log.info("오도메트리 입력: Arduino Mega USB STATE")
        elif _HAS_GPIO and ENCODER_QUAD_PINS:
            self._setup_quadrature()
            ticks_per_rev = ENCODER_QUAD_TICKS_PER_REVOLUTION
        elif _HAS_GPIO:
            self._setup_single_channel()
        else:
            log.info("RPi.GPIO 없음 - 엔코더는 시뮬레이션/비활성 모드로 동작합니다.")
        self._distance_per_tick = (
            (2 * math.pi * WHEEL_RADIUS_M) / ticks_per_rev
        ) * DISTANCE_CALIBRATION_FACTOR
        self._prev_time = time.monotonic()

    # ------------------------------------------------------------------
    # 틱 소스 1: 쿼드러처 A/B
    # ------------------------------------------------------------------
    def _setup_quadrature(self):
        from sensors.quadrature_encoder import QuadratureChannel

        pins = ENCODER_QUAD_PINS
        self._quad_left = QuadratureChannel(
            pins["left_a"], pins["left_b"], SIGN_LEFT_ENCODER, ENCODER_DECODE
        )
        self._quad_right = QuadratureChannel(
            pins["right_a"], pins["right_b"], SIGN_RIGHT_ENCODER, ENCODER_DECODE
        )
        self._gpio_ready = self._quad_left.ready and self._quad_right.ready
        if self._gpio_ready:
            log.info("쿼드러처 엔코더 활성 (decode=x%d)", ENCODER_DECODE)

    # ------------------------------------------------------------------
    # 틱 소스 2: 단일 채널 (레거시 폴백)
    # ------------------------------------------------------------------
    def _setup_single_channel(self):
        edge = {
            "RISING": GPIO.RISING,
            "FALLING": GPIO.FALLING,
            "BOTH": GPIO.BOTH,
        }.get(ENCODER_EDGE.upper(), GPIO.RISING)
        kwargs = {}
        if ENCODER_BOUNCETIME_MS and ENCODER_BOUNCETIME_MS > 0:
            kwargs["bouncetime"] = int(ENCODER_BOUNCETIME_MS)
        try:
            GPIO.setmode(GPIO.BCM)
            for pin in (ENCODER_PINS["left"], ENCODER_PINS["right"]):
                GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.add_event_detect(
                ENCODER_PINS["left"], edge, callback=self._left_cb, **kwargs
            )
            GPIO.add_event_detect(
                ENCODER_PINS["right"], edge, callback=self._right_cb, **kwargs
            )
            self._gpio_ready = True
        except Exception as exc:  # 핀 충돌, 권한 문제 등
            log.error("엔코더 GPIO 초기화 실패: %s (추측항법 비활성)", exc)

    def _left_cb(self, channel):
        with self._lock:
            self._left_ticks += int(SIGN_LEFT_ENCODER) * self.left_dir
            self.total_ticks += 1

    def _right_cb(self, channel):
        with self._lock:
            self._right_ticks += int(SIGN_RIGHT_ENCODER) * self.right_dir
            self.total_ticks += 1

    # ------------------------------------------------------------------
    def inject_ticks(self, left: int, right: int):
        """시뮬레이션/테스트용으로 외부에서 틱을 주입한다."""
        with self._lock:
            self._left_ticks += left
            self._right_ticks += right
            self.total_ticks += abs(left) + abs(right)

    def inject_wheel_degrees(self, left_degrees: float, right_degrees: float):
        """Inject logical wheel-output rotation reported by the Mega.

        STATE angles already use positive=vehicle-forward for both wheels, so
        no motor/encoder sign correction belongs on the Pi side.
        """
        scale = WHEEL_RADIUS_M * DISTANCE_CALIBRATION_FACTOR
        left_m = math.radians(left_degrees) * scale
        right_m = math.radians(right_degrees) * scale
        with self._lock:
            self._left_distance_m += left_m
            self._right_distance_m += right_m
            if abs(left_degrees) + abs(right_degrees) > 0.0:
                self.total_ticks += 1

    def _consume_injected_distance(self):
        with self._lock:
            left = self._left_distance_m
            right = self._right_distance_m
            self._left_distance_m = 0.0
            self._right_distance_m = 0.0
        return left, right

    def _consume_ticks(self):
        """이번 틱 구간의 좌/우 부호 있는 카운트를 소비한다."""
        if self._quad_left is not None:
            lt = self._quad_left.read_and_reset()
            rt = self._quad_right.read_and_reset()
            self.total_ticks = self._quad_left.total + self._quad_right.total
            # 시뮬레이션 주입분도 합산 (실기에서는 0)
            with self._lock:
                lt += self._left_ticks
                rt += self._right_ticks
                self._left_ticks = self._right_ticks = 0
            return lt, rt
        # 경합 방지: 락 안에서 읽고 즉시 0으로 되돌린다(스왑).
        with self._lock:
            lt, rt = self._left_ticks, self._right_ticks
            self._left_ticks = 0
            self._right_ticks = 0
        return lt, rt

    # ------------------------------------------------------------------
    def update(self):
        """메인 루프에서 매 틱 호출. x, y, theta 갱신."""
        now = time.monotonic()
        dt = now - self._prev_time
        self._prev_time = now

        lt, rt = self._consume_ticks()
        injected_left, injected_right = self._consume_injected_distance()
        d_left = lt * self._distance_per_tick + injected_left
        d_right = rt * self._distance_per_tick + injected_right
        if d_left == 0.0 and d_right == 0.0:
            if dt > 0:
                self.v = self.omega = 0.0
                self.wheel_v = (0.0, 0.0)
            self._on_no_motion(dt)
            return
        d_center = (d_left + d_right) / 2.0
        d_theta = self._delta_theta(d_left, d_right, dt)

        # 중점(midpoint) 적분: 직선 근사보다 회전 중 오차가 작다.
        mid_theta = self.theta + d_theta / 2.0
        self.x += d_center * math.cos(mid_theta)
        self.y += d_center * math.sin(mid_theta)
        self.theta += d_theta
        self.path_length += abs(d_center)

        if dt > 0:
            self.v = d_center / dt
            self.omega = d_theta / dt
            self.wheel_v = (d_left / dt, d_right / dt)

    # 서브클래스 훅 --------------------------------------------------------
    def _delta_theta(self, d_left: float, d_right: float, dt: float) -> float:
        """이번 구간의 방향 변화량. 기본은 엔코더 차동."""
        return (d_right - d_left) / WHEEL_BASE_M

    def _on_no_motion(self, dt: float):
        """틱이 없던 구간의 훅 (IMU 버전이 회전 적분에 사용)."""

    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """엔코더를 실제로 쓸 수 있는 상태인지."""
        return self._external_ready or self._gpio_ready or not _HAS_GPIO

    def reset(self):
        with self._lock:
            self._left_ticks = 0
            self._right_ticks = 0
            self._left_distance_m = 0.0
            self._right_distance_m = 0.0
        if self._quad_left is not None:
            self._quad_left.read_and_reset()
            self._quad_right.read_and_reset()
        self.x = self.y = self.theta = 0.0
        self.v = self.omega = 0.0
        self.wheel_v = (0.0, 0.0)
        self._prev_time = time.monotonic()

    def cleanup(self):
        if self._quad_left is not None:
            self._quad_left.cleanup()
            self._quad_right.cleanup()
            self._gpio_ready = False
            return
        if _HAS_GPIO and self._gpio_ready:
            for pin in (ENCODER_PINS["left"], ENCODER_PINS["right"]):
                try:
                    GPIO.remove_event_detect(pin)
                except Exception:
                    pass
            try:
                GPIO.cleanup([ENCODER_PINS["left"], ENCODER_PINS["right"]])
            except Exception:
                pass
            self._gpio_ready = False


class ImuFusedOdometry(Odometry):
    """거리 = 엔코더, theta = IMU 자이로 적분.

    바퀴가 흙에서 미끄러져도 회전각이 정확해진다(유턴·헤드랜드 선회의 핵심).
    IMU 가 없거나 죽으면 그 순간부터 엔코더 차동 theta 로 자동 폴백한다.
    시작 전 로봇이 정지한 상태에서 calibrate() 를 한 번 호출할 것.
    """

    def __init__(self, imu=None):
        super().__init__()
        if imu is None:
            from sensors.imu import MPU6050Yaw

            imu = MPU6050Yaw()
        self.imu = imu

    def calibrate(self):
        self.imu.calibrate_bias()

    def _delta_theta(self, d_left, d_right, dt):
        if self.imu.available and dt > 0:
            return self.imu.read_yaw_rate() * dt
        return super()._delta_theta(d_left, d_right, dt)

    def _on_no_motion(self, dt):
        # 바퀴 틱이 없어도(제자리 미끄러짐/외력) 실제 회전은 있을 수 있다.
        if self.imu.available and dt > 0:
            d_theta = self.imu.read_yaw_rate() * dt
            self.theta += d_theta
            self.omega = d_theta / dt

    def cleanup(self):
        super().cleanup()
        self.imu.cleanup()


def create_odometry():
    """config.ODOMETRY_BACKEND 에 따라 오도메트리를 생성한다."""
    from config import ODOMETRY_BACKEND

    if ODOMETRY_BACKEND == "encoder_imu":
        return ImuFusedOdometry()
    if ODOMETRY_BACKEND != "encoder":
        raise ValueError(f"지원하지 않는 ODOMETRY_BACKEND: {ODOMETRY_BACKEND}")
    return Odometry()
