#!/usr/bin/env python3
"""USB interface for the hybrid Arduino Mega motion controller.

The Mega owns encoder sampling and wheel PID. The Pi uses two command modes:

``DRIVE left_rpm right_rpm``
    Continuous differential-wheel velocity for the 20 Hz navigation loop.
    Each navigation tick refreshes the command; the firmware stops after a
    short silence, so a stalled Pi loop cannot leave the robot driving.

``MOVE seq left_deg right_deg max_rpm``
    Relative wheel-output angles for precise turns and finite moves. This
    method waits for ``ACK seq`` and ``DONE seq`` while sending ``HB``.

Firmware: ``firmware/agrix_motor_mega/agrix_motor_mega.ino``
Transport: USB CDC serial, ASCII lines, 115200 baud.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import math
from pathlib import Path
import sys
import threading
import time
from typing import Callable, Optional

_FARM_ROBOT = str(Path(__file__).resolve().parents[1])
if _FARM_ROBOT not in sys.path:
    sys.path.insert(0, _FARM_ROBOT)

from config import (  # noqa: E402
    MEGA_ACK_TIMEOUT_SEC,
    MEGA_DRIVE_MAX_RPM,
    MEGA_HEARTBEAT_SEC,
    MEGA_MOVE_TIMEOUT_SEC,
    SERIAL_MEGA_BAUD,
    SERIAL_MEGA_PORT,
    SERIAL_MEGA_RESET_DELAY_SEC,
    WHEEL_BASE_M,
    WHEEL_RADIUS_M,
)
from logutil import get_logger  # noqa: E402

log = get_logger("mega-motion")


class MegaMotionError(RuntimeError):
    """Base class for Mega transport/protocol failures."""


class MegaProtocolError(MegaMotionError):
    """The Mega explicitly rejected or aborted a command."""


class MegaTimeoutError(MegaMotionError):
    """The Mega did not acknowledge or complete a command in time."""


@dataclass(frozen=True)
class MegaState:
    sequence: int = 0
    mode: str = "UNKNOWN"
    left_degrees: float = 0.0
    right_degrees: float = 0.0
    left_rpm: float = 0.0
    right_rpm: float = 0.0
    received_at: float = 0.0


def find_port() -> str:
    """Find the stable Linux USB path for an Arduino Mega."""

    by_id = []
    for pattern in ("/dev/serial/by-id/*Mega*", "/dev/serial/by-id/*Arduino*"):
        by_id.extend(glob.glob(pattern))
    if by_id:
        return sorted(set(by_id))[0]
    candidates = sorted(glob.glob("/dev/ttyACM*"))
    if candidates:
        return candidates[0]
    raise MegaMotionError("Arduino Mega USB serial port not found")


class MegaMotion:
    """Thread-safe controller used by the real-hardware navigation stack."""

    def __init__(
        self,
        odometry=None,
        port: Optional[str] = SERIAL_MEGA_PORT,
        baud: int = SERIAL_MEGA_BAUD,
        *,
        serial_factory: Optional[Callable] = None,
        reset_delay: float = SERIAL_MEGA_RESET_DELAY_SEC,
    ):
        self.odom = odometry
        self.last_left = 0.0
        self.last_right = 0.0
        self.available = False
        self.faulted = False
        self.protocol_ready = False
        self.last_error = ""
        self.state = MegaState()

        self._serial = None
        self._closed = False
        self._stop_reader = threading.Event()
        self._write_lock = threading.Lock()
        self._condition = threading.Condition()
        self._acked: set[int] = set()
        self._done: set[int] = set()
        self._error_generation = 0
        self._next_sequence = int(time.time()) & 0x7FFFFFFF
        self._last_rx = 0.0
        self._previous_degrees: Optional[tuple[float, float]] = None
        self._thread: Optional[threading.Thread] = None

        try:
            device = port or find_port()
            if serial_factory is None:
                import serial

                serial_factory = serial.Serial
            self._serial = serial_factory(
                device, baud, timeout=0.1, write_timeout=1.0
            )
            reset_input = getattr(self._serial, "reset_input_buffer", None)
            if reset_input is not None:
                reset_input()
            if reset_delay > 0:
                time.sleep(reset_delay)  # opening USB CDC resets a Mega
            self.available = True
            self._thread = threading.Thread(
                target=self._reader, name="mega-motion-rx", daemon=True
            )
            self._thread.start()
            deadline = time.monotonic() + 1.0
            with self._condition:
                while not self.protocol_ready and time.monotonic() < deadline:
                    self._condition.wait(timeout=0.05)
            if not self.protocol_ready:
                raise MegaProtocolError(
                    "READY MEGA_MOTION_V2 not received; firmware/interface mismatch"
                )
            log.info("Mega Motion USB 연결: %s @ %d", device, baud)
        except Exception as exc:
            self.last_error = str(exc)
            self.available = False
            self.faulted = True
            self._stop_reader.set()
            if self._serial is not None:
                try:
                    self._serial.close()
                except Exception:
                    pass
            if self._thread is not None:
                self._thread.join(timeout=0.5)
            log.error("Mega Motion USB 연결 실패 (%s): %s", port, exc)

    @staticmethod
    def _clamp(value: float, limit: float = 1.0) -> float:
        return max(-limit, min(limit, float(value)))

    def _send(self, line: str) -> bool:
        if not self.available or self._serial is None or self._closed:
            return False
        try:
            payload = (line + "\n").encode("ascii")
            with self._write_lock:
                self._serial.write(payload)
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self.available = False
            log.error("Mega Motion 송신 실패: %s", exc)
            with self._condition:
                self._error_generation += 1
                self._condition.notify_all()
            return False

    def _reader(self):
        while not self._stop_reader.is_set():
            try:
                raw = self._serial.readline()
            except Exception as exc:
                if not self._closed:
                    self.last_error = str(exc)
                    self.available = False
                    log.error("Mega Motion 수신 실패: %s", exc)
                    with self._condition:
                        self._error_generation += 1
                        self._condition.notify_all()
                return
            if not raw:
                continue
            line = raw.decode("ascii", errors="replace").strip()
            if line:
                self._handle_line(line)

    def _handle_line(self, line: str):
        now = time.monotonic()
        self._last_rx = now
        parts = line.split()

        try:
            if len(parts) == 2 and parts[0] == "ACK":
                sequence = int(parts[1])
                with self._condition:
                    self._acked.add(sequence)
                    self._condition.notify_all()
                return
            if len(parts) == 2 and parts[0] == "DONE":
                sequence = int(parts[1])
                with self._condition:
                    self._done.add(sequence)
                    self._condition.notify_all()
                return
            if len(parts) == 7 and parts[0] == "STATE":
                state = MegaState(
                    sequence=int(parts[1]),
                    mode=parts[2],
                    left_degrees=float(parts[3]),
                    right_degrees=float(parts[4]),
                    left_rpm=float(parts[5]),
                    right_rpm=float(parts[6]),
                    received_at=now,
                )
                self._update_state(state)
                return
        except ValueError:
            log.warning("Mega Motion malformed response: %r", line)
            return

        if line.startswith("READY "):
            log.info("Mega firmware: %s", line)
            if line == "READY MEGA_MOTION_V2":
                with self._condition:
                    self.protocol_ready = True
                    self._condition.notify_all()
        elif line.startswith("ERR "):
            self.last_error = line
            self.faulted = True
            log.error("Mega Motion protocol error: %s", line)
            with self._condition:
                self._error_generation += 1
                self._condition.notify_all()

    def _update_state(self, state: MegaState):
        previous = self._previous_degrees
        self._previous_degrees = (state.left_degrees, state.right_degrees)
        self.state = state
        if previous is None or self.odom is None:
            return
        inject = getattr(self.odom, "inject_wheel_degrees", None)
        if inject is not None:
            inject(
                state.left_degrees - previous[0],
                state.right_degrees - previous[1],
            )

    # Continuous navigation commands. Values remain normalized at the FSM
    # boundary, but USB carries physical output-shaft RPM.
    def set_speeds(self, left_speed: float, right_speed: float):
        left = self._clamp(left_speed)
        right = self._clamp(right_speed)
        self.last_left, self.last_right = left, right
        self._send(
            f"DRIVE {left * MEGA_DRIVE_MAX_RPM:.3f} "
            f"{right * MEGA_DRIVE_MAX_RPM:.3f}"
        )

    def drive(self, base_speed: float, steer: float):
        """Arcade mix; positive steer turns right, matching the FSM convention."""

        left = float(base_speed) + float(steer)
        right = float(base_speed) - float(steer)
        peak = max(abs(left), abs(right))
        if peak > 1.0:
            excess = peak - 1.0
            shift = math.copysign(
                excess, base_speed if abs(base_speed) > 1e-9 else 1.0
            )
            left -= shift
            right -= shift
        self.set_speeds(left, right)

    def forward(self, speed: float):
        self.set_speeds(speed, speed)

    def rotate_in_place(self, clockwise: bool = True, speed: float = 0.3):
        if clockwise:
            self.set_speeds(speed, -speed)
        else:
            self.set_speeds(-speed, speed)

    def stop(self):
        self.last_left = self.last_right = 0.0
        self._send("STOP")

    # Finite encoder-position moves.
    def _new_sequence(self) -> int:
        with self._condition:
            self._next_sequence = (self._next_sequence + 1) & 0x7FFFFFFF
            if self._next_sequence == 0:
                self._next_sequence = 1
            return self._next_sequence

    def _wait_for(self, predicate, deadline: float, error_generation: int) -> bool:
        next_heartbeat = time.monotonic() + MEGA_HEARTBEAT_SEC
        while True:
            now = time.monotonic()
            if predicate():
                return True
            if self._error_generation != error_generation:
                raise MegaProtocolError(self.last_error or "Mega command failed")
            if now >= deadline:
                return False
            if now >= next_heartbeat:
                self._send("HB")
                next_heartbeat = now + MEGA_HEARTBEAT_SEC
            with self._condition:
                self._condition.wait(timeout=min(0.05, deadline - now))

    def move(
        self,
        left_degrees: float,
        right_degrees: float,
        max_rpm: float,
        *,
        timeout: float = MEGA_MOVE_TIMEOUT_SEC,
        sequence: Optional[int] = None,
    ) -> bool:
        """Execute one relative wheel move and wait for encoder settlement."""

        if not self.available:
            raise MegaMotionError(self.last_error or "Mega Motion is unavailable")
        values = (left_degrees, right_degrees, max_rpm, timeout)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("move arguments must be finite")
        if max_rpm <= 0 or timeout <= 0:
            raise ValueError("max_rpm and timeout must be positive")

        seq = self._new_sequence() if sequence is None else int(sequence)
        if seq < 0:
            raise ValueError("sequence must be non-negative")
        with self._condition:
            self._acked.discard(seq)
            self._done.discard(seq)
            error_generation = self._error_generation

        command = (
            f"MOVE {seq} {float(left_degrees):.3f} "
            f"{float(right_degrees):.3f} {float(max_rpm):.3f}"
        )
        if not self._send(command):
            raise MegaMotionError(self.last_error or "failed to send MOVE")

        ack_deadline = time.monotonic() + min(timeout, MEGA_ACK_TIMEOUT_SEC)
        if not self._wait_for(lambda: seq in self._acked, ack_deadline, error_generation):
            self.stop()
            raise MegaTimeoutError(f"MOVE {seq} ACK timeout")

        done_deadline = time.monotonic() + timeout
        if not self._wait_for(lambda: seq in self._done, done_deadline, error_generation):
            self.stop()
            raise MegaTimeoutError(f"MOVE {seq} completion timeout")
        return True

    def move_distance_blocking(
        self, distance_m: float, max_rpm: float = MEGA_DRIVE_MAX_RPM, **kwargs
    ) -> bool:
        wheel_degrees = math.degrees(float(distance_m) / WHEEL_RADIUS_M)
        return self.move(wheel_degrees, wheel_degrees, max_rpm, **kwargs)

    def turn_by_angle_blocking(self, delta_rad: float, speed: float = 0.3) -> bool:
        if abs(delta_rad) < 1e-9:
            return True
        wheel_travel = float(delta_rad) * WHEEL_BASE_M / 2.0
        wheel_degrees = math.degrees(wheel_travel / WHEEL_RADIUS_M)
        max_rpm = max(5.0, abs(self._clamp(speed)) * MEGA_DRIVE_MAX_RPM)
        scale = abs(delta_rad) / math.pi
        timeout = max(MEGA_MOVE_TIMEOUT_SEC * max(0.25, scale), 2.0)
        try:
            return self.move(-wheel_degrees, wheel_degrees, max_rpm, timeout=timeout)
        except MegaMotionError as exc:
            self.last_error = str(exc)
            log.error("Mega 정밀 회전 실패: %s", exc)
            return False

    def turn_180_blocking(self, speed: float = 0.3) -> bool:
        return self.turn_by_angle_blocking(-math.pi, speed=speed)

    def request_status(self):
        self._send("STATUS")

    def link_ok(self, max_age: float = 0.5) -> bool:
        return (
            self.available
            and not self.faulted
            and (time.monotonic() - self._last_rx) <= max_age
        )

    def cleanup(self):
        if self._closed:
            return
        self.stop()
        self._closed = True
        self._stop_reader.set()
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.available = False

    close = cleanup


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Move left/right wheel output shafts through the Mega V2 USB protocol"
    )
    parser.add_argument("left_deg", type=float)
    parser.add_argument("right_deg", type=float)
    parser.add_argument("max_rpm", type=float)
    parser.add_argument("--port", help="default: config port")
    parser.add_argument("--seq", type=int)
    parser.add_argument("--timeout", type=float, default=MEGA_MOVE_TIMEOUT_SEC)
    args = parser.parse_args()

    motion = MegaMotion(port=args.port or SERIAL_MEGA_PORT)
    try:
        motion.move(
            args.left_deg,
            args.right_deg,
            args.max_rpm,
            timeout=args.timeout,
            sequence=args.seq,
        )
        print("Move complete")
        return 0
    except MegaProtocolError as exc:
        print(f"Mega rejected move: {exc}", file=sys.stderr)
        return 2
    except MegaTimeoutError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except MegaMotionError as exc:
        print(f"Serial error: {exc}", file=sys.stderr)
        return 1
    finally:
        motion.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
