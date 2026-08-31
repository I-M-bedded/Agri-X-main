# -*- coding: utf-8 -*-
"""
sensors/quadrature_encoder.py
------------------------------
A/B 2채널(쿼드러처) 엔코더 카운터. 13 PPR × 4분주 = 모터축 1회전당 52카운트.

단일 채널 엔코더와의 차이
  - 위상(A/B 순서)으로 **방향을 직접 안다** -> 모터 명령으로 방향을 추정하던
    기존 방식(odometry.left_dir/right_dir)이 필요 없어진다.
  - 관성으로 뒤로 밀리거나 브레이크 중 미끄러져도 부호가 정확하다.

분주(디코드) 모드
  ENCODER_DECODE = 4 : A/B 모든 에지 카운트 (분해능 최대, 인터럽트 4배)
  ENCODER_DECODE = 1 : A 상승 에지에서만 카운트, B 레벨로 방향 판정
                       (Pi + Python 인터럽트 부하가 걱정되면 이걸로 낮출 것.
                        분해능은 1/4 이지만 13PPR×기어비면 보통 충분하다)

핀맵은 config.ENCODER_QUAD_PINS 로 준다. None 이면 이 모듈은 쓰이지 않고
odometry 가 기존 단일 채널 경로로 동작한다.
"""

import threading

from logutil import get_logger

log = get_logger("quad-enc")

try:
    import RPi.GPIO as GPIO

    _HAS_GPIO = True
except ImportError:
    _HAS_GPIO = False

# 4분주 상태 전이표: (이전 AB, 현재 AB) -> +1(정방향) / -1(역방향)
# 유효하지 않은 전이(2비트 동시 변화 = 노이즈/누락)는 0으로 무시한다.
_TRANSITION = {
    (0b00, 0b01): +1, (0b01, 0b11): +1, (0b11, 0b10): +1, (0b10, 0b00): +1,
    (0b00, 0b10): -1, (0b10, 0b11): -1, (0b11, 0b01): -1, (0b01, 0b00): -1,
}


class QuadratureChannel:
    """바퀴 1개분 A/B 채널. 부호 있는 카운트를 누적한다."""

    def __init__(self, pin_a: int, pin_b: int, sign: int = 1, decode: int = 4):
        self.pin_a = pin_a
        self.pin_b = pin_b
        self.sign = 1 if sign >= 0 else -1
        self.decode = decode
        self._lock = threading.Lock()
        self._count = 0          # 부호 있는 누적 카운트 (read_and_reset 로 소비)
        self.total = 0           # 생존 확인용 무부호 누적 (리셋 안 함)
        self._state = 0
        self.ready = False

        if not _HAS_GPIO:
            return
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pin_a, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.setup(pin_b, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            self._state = (GPIO.input(pin_a) << 1) | GPIO.input(pin_b)
            if decode >= 4:
                GPIO.add_event_detect(pin_a, GPIO.BOTH, callback=self._edge_cb)
                GPIO.add_event_detect(pin_b, GPIO.BOTH, callback=self._edge_cb)
            else:  # 1분주: A 상승 에지 + B 레벨로 방향
                GPIO.add_event_detect(pin_a, GPIO.RISING, callback=self._rising_cb)
            self.ready = True
        except Exception as exc:
            log.error("쿼드러처 엔코더 초기화 실패 (A=%d B=%d): %s", pin_a, pin_b, exc)

    # ------------------------------------------------------------------
    def _edge_cb(self, channel):
        new = (GPIO.input(self.pin_a) << 1) | GPIO.input(self.pin_b)
        step = _TRANSITION.get((self._state, new), 0)
        self._state = new
        if step:
            with self._lock:
                self._count += step * self.sign
                self.total += 1

    def _rising_cb(self, channel):
        # A 상승(01->11 전이) 시 B=HIGH 면 정방향 — 위 전이표와 같은 규약.
        # 실기에서 부호가 반대면 config 의 SIGN_*_ENCODER 를 뒤집으면 된다.
        step = 1 if GPIO.input(self.pin_b) else -1
        with self._lock:
            self._count += step * self.sign
            self.total += 1

    # ------------------------------------------------------------------
    def read_and_reset(self) -> int:
        """마지막 호출 이후의 부호 있는 카운트를 반환하고 0으로 되돌린다."""
        with self._lock:
            count = self._count
            self._count = 0
        return count

    def cleanup(self):
        if _HAS_GPIO and self.ready:
            for pin in (self.pin_a, self.pin_b):
                try:
                    GPIO.remove_event_detect(pin)
                except Exception:
                    pass
            try:
                GPIO.cleanup([self.pin_a, self.pin_b])
            except Exception:
                pass
            self.ready = False
