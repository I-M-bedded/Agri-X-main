# -*- coding: utf-8 -*-
"""
actuators/serial_motor_driver.py
---------------------------------
라즈베리파이 5(상위) → 아두이노 메가(하위) USB 시리얼 구동 계층.

계층 분담
    Pi 5   비전 / ToF / 마커 / FSM.  "좌우 바퀴를 몇 m/s 로" 만 내려보낸다.
    Mega   엔코더를 읽어 **속도 PID** 로 그 명령을 추종한다(100Hz).

왜 나누는가
    Pi 는 리눅스라 수십 ms 지터가 있어 100Hz 속도 루프가 흔들리고, 파이썬
    콜백으로는 엔코더 인터럽트를 놓친다(13PPR x 4체배 x 감속비).
    Mega 는 지터 없이 돌고 인터럽트를 놓치지 않는다.

인터페이스
    MotorDriver / ClosedLoopDrive 와 **동일**하다(drive/set_speeds/stop/
    turn_by_angle_blocking/cleanup). 따라서 FSM 은 어느 구동 계층인지 모른다.
    config.DRIVE_MODE = "serial_mega" 로 선택한다.

펌웨어
    firmware/agrix_motor_mega/agrix_motor_mega.ino
    프로토콜(줄 단위 ASCII, 115200):
        Pi -> Mega : "V <l_mps> <r_mps>", "S", "P <kp> <ki> <kd>", "?"
        Mega -> Pi : "T <l_mps> <r_mps> <l_ticks> <r_ticks> <l_duty> <r_duty> <flags>"

안전
    Mega 가 WATCHDOG_MS(400ms) 무명령 시 **스스로 정지**한다. 따라서 이 클래스는
    정지 상태에서도 주기적으로 명령을 보내야 한다(_tick 이 그 역할).
    USB 가 빠지거나 Pi 가 죽어도 로봇이 계속 달리지 않는다.
"""

import threading
import time
from typing import Optional, Tuple

from config import (
    MAX_WHEEL_SPEED_MPS,
    SERIAL_MEGA_BAUD,
    SERIAL_MEGA_KEEPALIVE_SEC,
    SERIAL_MEGA_PORT,
    SIGN_LEFT_MOTOR,
    SIGN_RIGHT_MOTOR,
)
from logutil import get_logger

log = get_logger("mega")


class SerialMotorDriver:
    """아두이노 메가 하위 제어기와 USB 로 대화하는 구동 계층."""

    def __init__(self, odometry=None, port: str = SERIAL_MEGA_PORT,
                 baud: int = SERIAL_MEGA_BAUD):
        self.odom = odometry
        self.last_left = 0.0
        self.last_right = 0.0
        self.available = False
        self.last_error = ""

        # 메가가 보고한 실측값 (텔레메트리)
        self.measured_mps: Tuple[float, float] = (0.0, 0.0)
        self.ticks: Tuple[int, int] = (0, 0)
        self.watchdog_stopped = False
        self._last_rx = 0.0

        self._serial = None
        self._lock = threading.Lock()
        self._closed = False
        self._last_tx = 0.0

        try:
            import serial

            self._serial = serial.Serial(port, baud, timeout=0.2)
            time.sleep(2.0)          # 메가 리셋 부팅 대기
            self._serial.reset_input_buffer()
            self._thread = threading.Thread(target=self._reader, name="mega-rx",
                                            daemon=True)
            self._thread.start()
            self.available = True
            log.info("아두이노 메가 연결: %s @ %d", port, baud)
        except Exception as exc:
            self.last_error = str(exc)
            log.error("메가 시리얼 연결 실패 (%s): %s — 모터가 동작하지 않습니다.",
                      port, exc)

    # ------------------------------------------------------------------
    def _reader(self):
        """텔레메트리 수신. 실측 속도/엔코더를 오도메트리에 넘긴다."""
        while not self._closed:
            try:
                line = self._serial.readline().decode(errors="ignore").strip()
            except Exception as exc:
                if not self._closed:
                    log.warning("메가 수신 오류: %s", exc)
                    time.sleep(0.5)
                continue
            if not line or not line.startswith("T "):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                lm, rm = float(parts[1]), float(parts[2])
                lt, rt = int(parts[3]), int(parts[4])
                flags = int(parts[7])
            except ValueError:
                continue

            with self._lock:
                prev_lt, prev_rt = self.ticks
                self.measured_mps = (lm, rm)
                self.ticks = (lt, rt)
                self.watchdog_stopped = bool(flags & 1)
                self._last_rx = time.monotonic()

            # 오도메트리에 **증분 틱**을 넘긴다(Odometry.inject_ticks 계약).
            if self.odom is not None:
                self.odom.inject_ticks(lt - prev_lt, rt - prev_rt)

    # ------------------------------------------------------------------
    def _send(self, text: str):
        if not self.available:
            return
        try:
            self._serial.write((text + "\n").encode())
            self._last_tx = time.monotonic()
        except Exception as exc:
            self.last_error = str(exc)
            log.warning("메가 송신 실패: %s", exc)

    def set_speeds(self, left_speed: float, right_speed: float):
        """무차원 [-1,1] 명령 -> m/s 로 환산해 메가에 내려보낸다."""
        left = max(-1.0, min(1.0, left_speed))
        right = max(-1.0, min(1.0, right_speed))
        self.last_left, self.last_right = left, right
        lm = left * SIGN_LEFT_MOTOR * MAX_WHEEL_SPEED_MPS
        rm = right * SIGN_RIGHT_MOTOR * MAX_WHEEL_SPEED_MPS
        self._send(f"V {lm:.4f} {rm:.4f}")

    def drive(self, base_speed: float, steer: float):
        """부호 규약은 기존과 동일: steer>0 = 오른쪽 조향."""
        left = base_speed + steer
        right = base_speed - steer
        peak = max(abs(left), abs(right))
        if peak > 1.0:      # 조향량 보존 스케일링
            excess = peak - 1.0
            shift = excess if base_speed >= 0 else -excess
            left -= shift
            right -= shift
        self.set_speeds(left, right)

    def forward(self, speed: float):
        self.set_speeds(speed, speed)

    def stop(self):
        self.last_left = self.last_right = 0.0
        self._send("S")

    def keepalive(self):
        """메가 워치독(400ms)이 걸리지 않게 주기적으로 현재 명령을 재전송.

        FSM 의 매 제어 틱에서 호출한다. 정지 중에도 호출해야 한다 —
        그래야 '연결이 살아 있음'과 '정말 정지 명령'을 메가가 구분한다.
        """
        if time.monotonic() - self._last_tx >= SERIAL_MEGA_KEEPALIVE_SEC:
            self.set_speeds(self.last_left, self.last_right)

    # ------------------------------------------------------------------
    def rotate_in_place(self, clockwise: bool = True, speed: float = 0.3):
        if clockwise:
            self.set_speeds(speed, -speed)
        else:
            self.set_speeds(-speed, speed)

    def turn_by_angle_blocking(self, delta_rad: float, speed: float = 0.3) -> bool:
        """제자리 회전(블로킹). 각도 판정은 상위의 오도메트리로 한다.

        메가는 '속도'만 책임진다. 회전 각도는 오도메트리(엔코더/IMU)를 가진
        상위가 판정해야 하므로, 기존 MotorDriver 와 같은 방식으로 여기서 돈다.
        """
        import math

        from config import TURN_180_DURATION_SEC, TURN_TIMEOUT_MARGIN

        if abs(delta_rad) < 1e-6:
            return True
        nominal = TURN_180_DURATION_SEC * (abs(delta_rad) / math.pi)
        timeout = max(0.5, nominal * TURN_TIMEOUT_MARGIN)
        clockwise = delta_rad < 0

        if self.odom is None:
            self.rotate_in_place(clockwise=clockwise, speed=speed)
            time.sleep(nominal)
            self.stop()
            return True

        start = self.odom.theta
        target = start + delta_rad
        t0 = time.monotonic()
        self.rotate_in_place(clockwise=clockwise, speed=speed)
        reached = False
        while True:
            self.odom.update()
            self.keepalive()
            if (delta_rad > 0 and self.odom.theta >= target) or \
               (delta_rad < 0 and self.odom.theta <= target):
                reached = True
                break
            if time.monotonic() - t0 > timeout:
                log.warning("회전 타임아웃(%.1fs). 실제 %.2f rad 에서 정지합니다.",
                            timeout, self.odom.theta - start)
                break
            time.sleep(0.005)
        self.stop()
        return reached

    def turn_180_blocking(self, speed: float = 0.3) -> bool:
        import math

        return self.turn_by_angle_blocking(-math.pi, speed=speed)

    # ------------------------------------------------------------------
    def link_ok(self) -> bool:
        """최근 텔레메트리를 받고 있는가(=하위 제어기가 살아 있는가)."""
        with self._lock:
            return self.available and (time.monotonic() - self._last_rx) < 1.0

    def cleanup(self):
        self.stop()
        self._closed = True
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
        self.available = False
