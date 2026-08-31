#!/usr/bin/env python3
"""Interactive WASD jog controller for Raspberry Pi -> Arduino Mega Motion.

Controls:
  W / S : forward / reverse
  A / D : rotate left / right in place
  X or Space : stop immediately
  P : request and print latest Mega state
  + / - : increase / decrease normalized speed
  Q : stop and quit

Safety:
  A motion key is a dead-man command. The Pi refreshes DRIVE only while recent
  W/A/S/D key-repeat events are arriving. When input stops for ``--deadman``
  seconds, STOP is sent. The Mega firmware's independent 400 ms DRIVE watchdog
  remains active if this process stalls or the USB link is interrupted.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import select
import sys
import termios
import time
import tty

_FARM_ROBOT = str(Path(__file__).resolve().parents[1])
if _FARM_ROBOT not in sys.path:
    sys.path.insert(0, _FARM_ROBOT)

from config import SERIAL_MEGA_PORT  # noqa: E402
from control.mega_motion import MegaMotion  # noqa: E402


REFRESH_SEC = 0.10
DEFAULT_SPEED = 0.25
DEFAULT_DEADMAN_SEC = 0.70
SPEED_STEP = 0.05
MIN_SPEED = 0.05
MAX_SPEED = 1.00


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def print_help() -> None:
    print(
        "\nControls: W forward | S reverse | A left | D right | "
        "X/Space stop | P status | +/- speed | Q quit"
    )
    print("Hold/repeat W/A/S/D to keep moving; releasing the key auto-stops.\n")


def print_state(motion: MegaMotion) -> None:
    motion.request_status()
    time.sleep(0.06)
    state = motion.state
    print(
        f"\nSTATE mode={state.mode} "
        f"L={state.left_rpm:+.1f}rpm R={state.right_rpm:+.1f}rpm "
        f"Ldeg={state.left_degrees:+.1f} Rdeg={state.right_degrees:+.1f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safe interactive WASD jog controller for Agri-X Mega Motion"
    )
    parser.add_argument(
        "--port",
        default=SERIAL_MEGA_PORT,
        help="Mega serial device; omit/empty config value to auto-detect",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEED,
        help="initial normalized wheel speed, 0.05..1.0 (default: 0.25)",
    )
    parser.add_argument(
        "--deadman",
        type=float,
        default=DEFAULT_DEADMAN_SEC,
        help="seconds without a motion key before STOP (default: 0.70)",
    )
    args = parser.parse_args()

    if not sys.stdin.isatty():
        print("This controller requires an interactive terminal (TTY).", file=sys.stderr)
        return 2

    speed = clamp(args.speed, MIN_SPEED, MAX_SPEED)
    deadman = clamp(args.deadman, 0.20, 2.00)
    motion = MegaMotion(port=args.port or None)
    if not motion.available or motion.faulted:
        print(f"Mega Motion connection failed: {motion.last_error}", file=sys.stderr)
        motion.cleanup()
        return 1

    print("Mega Motion connected.")
    print(f"speed={speed:.2f}, deadman={deadman:.2f}s, refresh={REFRESH_SEC:.2f}s")
    print_help()

    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    active_command: tuple[float, float] | None = None
    command_deadline = 0.0
    next_refresh = 0.0

    key_to_command = {
        "w": lambda v: (v, v),
        "s": lambda v: (-v, -v),
        "a": lambda v: (-v, v),
        "d": lambda v: (v, -v),
    }

    try:
        tty.setcbreak(fd)
        while True:
            now = time.monotonic()

            readable, _, _ = select.select([sys.stdin], [], [], 0.02)
            if readable:
                key = sys.stdin.read(1).lower()

                if key in key_to_command:
                    active_command = key_to_command[key](speed)
                    command_deadline = now + deadman
                    next_refresh = 0.0
                elif key in ("x", " "):
                    active_command = None
                    command_deadline = 0.0
                    motion.stop()
                    print("\rSTOP                              ", end="", flush=True)
                elif key == "p":
                    print_state(motion)
                elif key in ("+", "="):
                    speed = clamp(speed + SPEED_STEP, MIN_SPEED, MAX_SPEED)
                    print(f"\rspeed={speed:.2f}                         ", end="", flush=True)
                elif key in ("-", "_"):
                    speed = clamp(speed - SPEED_STEP, MIN_SPEED, MAX_SPEED)
                    print(f"\rspeed={speed:.2f}                         ", end="", flush=True)
                elif key == "q":
                    break
                elif key == "h":
                    print_help()

            now = time.monotonic()
            if active_command is not None:
                if now >= command_deadline:
                    active_command = None
                    motion.stop()
                    print("\rDEADMAN STOP                       ", end="", flush=True)
                elif now >= next_refresh:
                    left, right = active_command
                    if not motion.set_speeds(left, right):
                        raise RuntimeError(
                            motion.last_error or "failed to send DRIVE command"
                        )
                    next_refresh = now + REFRESH_SEC
                    print(
                        f"\rDRIVE L={left:+.2f} R={right:+.2f} speed={speed:.2f}  ",
                        end="",
                        flush=True,
                    )

            if motion.faulted:
                raise RuntimeError(motion.last_error or "Mega Motion fault")

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"\nController stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            motion.stop()
        finally:
            motion.cleanup()
            termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        print("\nStopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
