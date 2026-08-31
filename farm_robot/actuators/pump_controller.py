# -*- coding: utf-8 -*-
"""
actuators/pump_controller.py
-----------------------------
워터펌프는 LR7843 N-MOSFET 모듈로 스위칭한다.

실기 배선 (Raspberry Pi 40-pin header):
  - physical pin 22 = BCM GPIO25 -> LR7843 SIG
  - physical pin 20 = GND        -> LR7843 GND

현재 제어는 디지털 ON/OFF이며 HIGH = ON(100%), LOW = OFF이다.
펌프는 전압이 인가되는 순간 바로 작동하므로 "고랑 밖에서는 절대 켜지지
않도록" 소프트웨어 인터록을 강제한다.

안전 기능
  1) tick()이 매 제어 주기마다 MOSFET 출력 상태를 다시 확정한다.
  2) PUMP_MAX_CONTINUOUS_SEC 최대 연속 가동 워치독을 유지한다.
  3) 물 부족(water_low)일 때는 zone과 무관하게 펌프를 잠근다.
"""

import time

from config import GPIO_WARNINGS, PUMP_MAX_CONTINUOUS_SEC
from logutil import get_logger

log = get_logger("pump")

# RPi.GPIO uses BCM numbering. Physical pin 22 on the 40-pin header is BCM25.
PUMP_MOSFET_PIN = 25
PUMP_MOSFET_ACTIVE_HIGH = True

try:
    import RPi.GPIO as GPIO

    _HAS_GPIO = True
except ImportError:
    _HAS_GPIO = False


class PumpController:
    def __init__(self):
        self._in_furrow = False        # 안전 게이트: 지금 고랑 안인가
        self._requested_on = False     # 상위 로직이 켜기를 원하는가
        self._locked_out = False       # 물 부족 등으로 잠금
        self._output_state = False
        self._on_since = None
        self._gpio_ready = False

        if _HAS_GPIO:
            try:
                GPIO.setwarnings(GPIO_WARNINGS)
                GPIO.setmode(GPIO.BCM)
                # LR7843 is active-high. Explicit LOW initialization prevents
                # a startup pulse from briefly turning the pump on.
                off_level = GPIO.LOW if PUMP_MOSFET_ACTIVE_HIGH else GPIO.HIGH
                GPIO.setup(PUMP_MOSFET_PIN, GPIO.OUT, initial=off_level)
                self._gpio_ready = True
            except Exception as exc:
                log.error("펌프 MOSFET GPIO 초기화 실패: %s", exc)
        self._apply_output(False)

    # ------------------------------------------------------------------
    def _apply_output(self, on: bool):
        self._output_state = bool(on)
        if on:
            if self._on_since is None:
                self._on_since = time.monotonic()
        else:
            self._on_since = None

        if not self._gpio_ready:
            return
        level = GPIO.HIGH if (on == PUMP_MOSFET_ACTIVE_HIGH) else GPIO.LOW
        GPIO.output(PUMP_MOSFET_PIN, level)

    def _desired_state(self) -> bool:
        return self._requested_on and self._in_furrow and not self._locked_out

    # ------------------------------------------------------------------
    def set_zone(self, in_furrow: bool):
        """상위 내비게이션이 '지금 고랑 안/밖'을 알려줄 때 호출."""
        changed = self._in_furrow != bool(in_furrow)
        self._in_furrow = bool(in_furrow)
        if not self._in_furrow:
            # 고랑을 벗어나는 순간 요청 자체를 취소하고 즉시 OFF
            self._requested_on = False
        if changed:
            log.debug("펌프 zone -> %s", "고랑 안" if in_furrow else "고랑 밖")
        self._apply_output(self._desired_state())

    def set_lockout(self, locked: bool):
        """물 부족 등으로 펌프를 강제로 잠근다(공회전 방지)."""
        if self._locked_out != bool(locked):
            log.info("펌프 잠금 %s", "설정" if locked else "해제")
        self._locked_out = bool(locked)
        self._apply_output(self._desired_state())

    def turn_on(self) -> bool:
        """펌프 ON(100%) 요청. 실제로 켜졌으면 True를 반환한다."""
        self._requested_on = True
        desired = self._desired_state()
        self._apply_output(desired)
        if not desired:
            log.debug("펌프 ON 요청이 인터록에 의해 차단되었습니다.")
        return desired

    def turn_off(self):
        """OFF는 구역과 무관하게 항상 즉시 반영."""
        self._requested_on = False
        self._apply_output(False)

    def tick(self):
        """MOSFET 출력을 재확정하고 최대 연속 가동 시간을 감시한다."""
        desired = self._desired_state()

        if (
            desired
            and self._on_since is not None
            and PUMP_MAX_CONTINUOUS_SEC > 0
            and time.monotonic() - self._on_since > PUMP_MAX_CONTINUOUS_SEC
        ):
            log.warning(
                "펌프가 %.0f초 이상 연속 가동되어 워치독이 강제로 껐습니다.",
                PUMP_MAX_CONTINUOUS_SEC,
            )
            self._requested_on = False
            self._apply_output(False)
            return

        if desired != self._output_state:
            self._apply_output(desired)

    def is_on(self) -> bool:
        return self._output_state

    def cleanup(self):
        self.turn_off()
        if self._gpio_ready:
            try:
                GPIO.cleanup(PUMP_MOSFET_PIN)
            except Exception:
                pass
            self._gpio_ready = False
