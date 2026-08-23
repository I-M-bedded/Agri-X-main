# -*- coding: utf-8 -*-
"""
sensors/odometry.py
-------------------
좌/우 엔코더 펄스로 로봇의 위치(x, y)와 방향(theta)을 추정한다.

부호 규약 (config.py 참조)
  theta : 반시계(CCW, 좌회전)가 **양수**.
          d_theta = (d_right - d_left) / WHEEL_BASE  이므로
          우륜이 빠르면 좌회전 -> theta 증가. 표준 규약과 일치한다.
          (이전 버전 주석은 "양수=우회전"이라고 잘못 적혀 있어서,
           이를 참조하던 헤딩 유지 보정이 반대로 동작하고 있었다)

이전 버전 대비 수정 사항
  1) GPIO.BOTH -> ENCODER_EDGE(기본 RISING). BOTH는 펄스당 콜백이 2번이라
     TICKS_PER_REVOLUTION 정의와 어긋나 거리/각도가 2배로 계산되고 있었다.
  2) 인터럽트 콜백과 update() 사이의 경합(race) 제거.
     "읽고 나서 0으로 초기화" 사이에 틱이 들어오면 유실되므로,
     락을 걸고 스왑하는 방식으로 바꿨다.
  3) path_length(누적 주행거리), 속도 추정 추가 (헤드랜드 이동에 사용).
  4) cleanup() 추가 - 예전에는 이벤트 검출이 해제되지 않았다.
"""

import math
import threading
import time

from config import (
    ENCODER_BOUNCETIME_MS,
    ENCODER_EDGE,
    ENCODER_PINS,
    SIGN_LEFT_ENCODER,
    SIGN_RIGHT_ENCODER,
    TICKS_PER_REVOLUTION,
    DISTANCE_CALIBRATION_FACTOR,
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

        self._lock = threading.Lock()
        self._left_ticks = 0
        self._right_ticks = 0
        self.total_ticks = 0      # 엔코더 생존 확인용 (회전 stall 감지)

        # 단일 채널 엔코더는 방향을 알 수 없으므로 모터 드라이버가 알려준다.
        self.left_dir = 1
        self.right_dir = 1

        # [궤도 보정] 엔코더는 '궤도가 움직인 양'을 재지, 로봇이 실제로 간
        #   거리를 재지 않는다. 궤도가 흙에서 헛돌면 그 차이만큼 과대평가된다.
        #   DISTANCE_CALIBRATION_FACTOR 로 실측 보정한다 (setup.py 4번).
        self._distance_per_tick = (
            (2 * math.pi * WHEEL_RADIUS_M) / TICKS_PER_REVOLUTION
        ) * DISTANCE_CALIBRATION_FACTOR
        self._prev_time = time.monotonic()
        self._gpio_ready = False

        if _HAS_GPIO:
            self._setup_gpio()
        else:
            log.info("RPi.GPIO 없음 - 엔코더는 시뮬레이션/비활성 모드로 동작합니다.")

    # ------------------------------------------------------------------
    def _setup_gpio(self):
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
            self._gpio_ready = False

    # ------------------------------------------------------------------
    def _left_cb(self, channel):
        with self._lock:
            self._left_ticks += int(SIGN_LEFT_ENCODER) * self.left_dir
            self.total_ticks += 1

    def _right_cb(self, channel):
        with self._lock:
            self._right_ticks += int(SIGN_RIGHT_ENCODER) * self.right_dir
            self.total_ticks += 1

    def inject_ticks(self, left: int, right: int):
        """시뮬레이션/테스트용으로 외부에서 틱을 주입한다."""
        with self._lock:
            self._left_ticks += left
            self._right_ticks += right
            self.total_ticks += abs(left) + abs(right)

    # ------------------------------------------------------------------
    def update(self):
        """메인 루프에서 매 틱 호출. x, y, theta 갱신."""
        now = time.monotonic()
        dt = now - self._prev_time
        self._prev_time = now

        # 경합 방지: 락 안에서 읽고 즉시 0으로 되돌린다(스왑).
        with self._lock:
            lt, rt = self._left_ticks, self._right_ticks
            self._left_ticks = 0
            self._right_ticks = 0

        if lt == 0 and rt == 0:
            if dt > 0:
                self.v = 0.0
                self.omega = 0.0
            return

        d_left = lt * self._distance_per_tick
        d_right = rt * self._distance_per_tick

        d_center = (d_left + d_right) / 2.0
        d_theta = (d_right - d_left) / WHEEL_BASE_M

        # 중점(midpoint) 적분: 직선 근사보다 회전 중 오차가 작다.
        mid_theta = self.theta + d_theta / 2.0
        self.x += d_center * math.cos(mid_theta)
        self.y += d_center * math.sin(mid_theta)
        self.theta += d_theta
        self.path_length += abs(d_center)

        if dt > 0:
            self.v = d_center / dt
            self.omega = d_theta / dt

    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """엔코더를 실제로 쓸 수 있는 상태인지."""
        return self._gpio_ready or not _HAS_GPIO

    def reset(self):
        with self._lock:
            self._left_ticks = 0
            self._right_ticks = 0
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0
        self.omega = 0.0
        self._prev_time = time.monotonic()

    def cleanup(self):
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
