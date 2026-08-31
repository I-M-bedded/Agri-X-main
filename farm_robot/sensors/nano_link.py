# -*- coding: utf-8 -*-
"""
sensors/nano_link.py
---------------------
Jetson Nano 가 USB 시리얼로 보내는 수위 상태("empty"/"ok")를 수신한다.

WaterTankSensor 와 **동일한 인터페이스**(poll / is_water_low / cleanup)를
제공하므로, config.WATER_SOURCE = "nano_usb" 로 바꾸면 FSM 은 코드 수정 없이
이 모듈을 쓴다. GPIO IR 센서 경로는 "gpio" 로 그대로 남아 있다.

프로토콜 (Nano 쪽 규약)
  줄 단위 텍스트. 줄 안에 NANO_EMPTY_KEYWORD("empty")가 있으면 물 부족,
  NANO_OK_KEYWORD("ok")가 있으면 정상. 그 외 줄은 무시(로그만).
  Nano 는 상태를 주기적으로(권장 1초) 재전송할 것 - 1회성 이벤트만 보내면
  케이블이 그 순간 빠졌을 때 로봇이 영영 모른다.

안전 정책
  - 디바운스: 같은 상태가 WATER_LEVEL_DEBOUNCE_SEC 이상 유지될 때만 확정.
  - 래치: WATER_EMPTY_LATCHED=True 면 한 번 확정된 '물 부족'은 재시작 전까지
    풀리지 않는다. 출렁임으로 empty/ok 가 번갈아 와서 복귀를 취소하고
    밭 한가운데서 다시 급수를 시작하는 사고를 막는다.
  - 링크 유실: NANO_LINK_STALE_SEC 동안 아무 줄도 없으면 '판단 불가'로
    간주하고 **마지막 확정 상태를 유지**한다(부족으로 단정하지 않음).
    link_ok() 로 상위가 상태를 조회할 수 있고, 주기적으로 경고를 남긴다.
"""

import threading
import time

from config import (
    NANO_EMPTY_KEYWORD,
    NANO_LINK_STALE_SEC,
    NANO_OK_KEYWORD,
    NANO_SERIAL_BAUD,
    NANO_SERIAL_PORT,
    WATER_EMPTY_LATCHED,
    WATER_LEVEL_DEBOUNCE_SEC,
)
from logutil import get_logger

log = get_logger("nano-link")


class NanoWaterLink:
    def __init__(self, port: str = NANO_SERIAL_PORT, baud: int = NANO_SERIAL_BAUD):
        self._raw_low = False
        self._confirmed_low = False
        self._changed_since = None
        self._last_rx = 0.0
        self._last_stale_warn = 0.0
        self._lock = threading.Lock()
        self._closed = False
        self._serial = None
        self._sim_low = False  # 시뮬레이션/테스트용 강제 값

        try:
            import serial

            self._serial = serial.Serial(port, baud, timeout=0.5)
            self._thread = threading.Thread(
                target=self._reader, name="nano-link", daemon=True
            )
            self._thread.start()
            log.info("Nano 수위 링크 연결: %s @ %d", port, baud)
        except Exception as exc:
            log.error("Nano 시리얼 연결 실패 (%s): %s - 수위는 '정상' 가정, "
                      "link_ok()=False", port, exc)

    # ------------------------------------------------------------------
    def _reader(self):
        """수신 스레드: 줄 단위로 raw 상태만 갱신한다 (판정은 poll 에서)."""
        while not self._closed:
            try:
                line = self._serial.readline().decode(errors="ignore").strip().lower()
            except Exception as exc:
                if not self._closed:
                    log.warning("Nano 링크 읽기 오류: %s", exc)
                    time.sleep(1.0)
                continue
            if not line:
                continue
            with self._lock:
                self._last_rx = time.monotonic()
                if NANO_EMPTY_KEYWORD in line:
                    self._raw_low = True
                elif NANO_OK_KEYWORD in line:
                    self._raw_low = False
                else:
                    log.debug("Nano 알 수 없는 메시지: %r", line)

    # ------------------------------------------------------------------
    def poll(self):
        """매 제어 틱 호출. 디바운스/래치/링크 감시를 진행한다."""
        now = time.monotonic()
        with self._lock:
            raw = self._raw_low if self._serial is not None else self._sim_low
            last_rx = self._last_rx

        if self._serial is not None and now - last_rx > NANO_LINK_STALE_SEC:
            # 판단 불가: 마지막 확정 상태 유지. 10초에 한 번만 경고.
            if now - self._last_stale_warn > 10.0:
                log.warning("Nano 수위 링크 무응답 %.0f초 - 마지막 상태(%s) 유지",
                            now - last_rx, "부족" if self._confirmed_low else "정상")
                self._last_stale_warn = now
            self._changed_since = None
            return

        if WATER_EMPTY_LATCHED and self._confirmed_low:
            return  # 래치: 한 번 부족이면 계속 부족

        if raw == self._confirmed_low:
            self._changed_since = None
            return
        if self._changed_since is None:
            self._changed_since = now
            return
        if now - self._changed_since >= WATER_LEVEL_DEBOUNCE_SEC:
            self._confirmed_low = raw
            self._changed_since = None
            log.info("수위 상태 확정 -> %s", "물 부족" if raw else "정상")

    def is_water_low(self) -> bool:
        """디바운스로 확정된 상태만 반환 (측정하지 않음)."""
        return self._confirmed_low

    def link_ok(self) -> bool:
        """최근 NANO_LINK_STALE_SEC 안에 수신이 있었는가."""
        if self._serial is None:
            return False
        return time.monotonic() - self._last_rx <= NANO_LINK_STALE_SEC

    def cleanup(self):
        self._closed = True
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
