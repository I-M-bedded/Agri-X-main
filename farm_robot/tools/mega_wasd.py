#!/usr/bin/env python3
"""Interactive Raspberry Pi WASD jog controller for the Mega motion firmware.

This bring-up tool talks to the same USB serial interface as Arduino Serial
Monitor, but uses the firmware's target-speed command instead of fixed raw WASD
speed:

    DRIVE <left_rpm> <right_rpm>

The default straight-line target is deliberately moderate (40 RPM): high enough
to avoid the sluggish 20 RPM bring-up setting, while remaining well below the
80 RPM raw-manual command used by the firmware's single-byte WASD interface.
In-place turns use 80% of the straight target to reduce track scrub/current.

The active DRIVE command is refreshed every 100 ms. If terminal key-repeat
stops, this tool sends STOP after the dead-man interval; the Mega's independent
400 ms DRIVE watchdog remains the final safety layer if the Pi process or USB
link stalls.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
import select
import sys
import termios
import time
import tty

_FARM_ROBOT = str(Path(__file__).resolve().parents[1])
if _FARM_ROBOT not in sys.path:
    sys.path.insert(0, _FARM_ROBOT)

from config import (  # noqa: E402
    SERIAL_MEGA_BAUD,
    SERIAL_MEGA_PORT,
    SERIAL_MEGA_RESET_DELAY_SEC,
)


REFRESH_SEC = 0.10
DEFAULT_RPM = 40.0
DEFAULT_TURN_SCALE = 0.80
DEFAULT_DEADMAN_SEC = 0.55
RPM_STEP = 5.0
MIN_RPM = 10.0
MAX_RPM = 80.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def find_serial_port() -> str:
    """Find a likely Arduino/USB-serial device on Raspberry Pi."""
    patterns = (
        "/dev/serial/by-id/*Mega*",
        "/dev/serial/by-id/*Arduino*",
        "/dev/serial/by-id/*CH340*",
        "/dev/serial/by-id/*CH341*",
    )
    by_id: list[str] = []
    for pattern in patterns:
        by_id.extend(glob.glob(pattern))
    if by_id:
        return sorted(set(by_id))[0]

    candidates = sorted(glob.glob("/dev/ttyACM*")) + sorted(glob.glob("/dev/ttyUSB*"))
    if candidates:
        return candidates[0]
    raise RuntimeError("Mega serial port not found (/dev/ttyACM* or /dev/ttyUSB*)")


def print_help(rpm: float, turn_scale: float) -> None:
    print(
        "\nControls: W forward | S reverse | A left | D right | "
        "X/Space stop | P state | +/- speed | Q quit"
    )
    print(
        f"Straight={rpm:.0f} RPM, turn={rpm * turn_scale:.0f} RPM. "
        f"+/- changes straight target by {RPM_STEP:.0f} RPM.\n"
    )


def write_line(ser, line: str) -> None:
    ser.write((line + "\n").encode("ascii"))
    ser.flush()


def send_stop(ser) -> None:
    write_line(ser, "STOP")


def command_for_key(key: str, rpm: float, turn_scale: float) -> tuple[float, float]:
    turn_rpm = rpm * turn_scale
    if key == "w":
        return rpm, rpm
    if key == "s":
        return -rpm, -rpm
    if key == "a":
        return -turn_rpm, turn_rpm
    if key == "d":
        return turn_rpm, -turn_rpm
    raise ValueError(f"unsupported motion key: {key}")


def send_drive(ser, key: str, rpm: float, turn_scale: float) -> tuple[float, float]:
    left, right = command_for_key(key, rpm, turn_scale)
    write_line(ser, f"DRIVE {left:.3f} {right:.3f}")
    return left, right


def drain_lines(ser, rx_buffer: bytearray) -> list[str]:
    waiting = int(getattr(ser, "in_waiting", 0))
    if waiting > 0:
        rx_buffer.extend(ser.read(waiting))

    lines: list[str] = []
    while b"\n" in rx_buffer:
        raw, _, rest = rx_buffer.partition(b"\n")
        rx_buffer[:] = rest
        line = raw.decode("ascii", errors="replace").strip()
        if line:
            lines.append(line)
    return lines


def verify_link(ser, timeout: float = 1.0) -> tuple[bool, list[str], bytearray]:
    """Request one STATE line and confirm Pi <-> Mega communication."""
    rx_buffer = bytearray()
    seen: list[str] = []

    ser.reset_input_buffer()
    write_line(ser, "STATUS")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for line in drain_lines(ser, rx_buffer):
            seen.append(line)
            if line.startswith("STATE "):
                return True, seen, rx_buffer
        time.sleep(0.01)
    return False, seen, rx_buffer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Target-RPM WASD jog test for Raspberry Pi -> Agri-X Mega"
    )
    parser.add_argument(
        "--port",
        default=SERIAL_MEGA_PORT,
        help="serial device, e.g. /dev/ttyACM0; auto-detected when omitted",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=SERIAL_MEGA_BAUD,
        help=f"serial baud rate (default: {SERIAL_MEGA_BAUD})",
    )
    parser.add_argument(
        "--rpm",
        type=float,
        default=DEFAULT_RPM,
        help=f"initial straight target RPM ({MIN_RPM:.0f}..{MAX_RPM:.0f}, default: {DEFAULT_RPM:.0f})",
    )
    parser.add_argument(
        "--turn-scale",
        type=float,
        default=DEFAULT_TURN_SCALE,
        help="in-place turn RPM / straight RPM (default: 0.80)",
    )
    parser.add_argument(
        "--deadman",
        type=float,
        default=DEFAULT_DEADMAN_SEC,
        help="seconds without a repeated motion key before STOP (default: 0.55)",
    )
    parser.add_argument(
        "--reset-delay",
        type=float,
        default=SERIAL_MEGA_RESET_DELAY_SEC,
        help="delay after opening serial because Mega may reset",
    )
    args = parser.parse_args()

    if not sys.stdin.isatty():
        print("This controller must be run in an interactive terminal.", file=sys.stderr)
        return 2

    try:
        import serial
    except ImportError:
        print(
            "pyserial is not installed. Run: sudo apt install python3-serial",
            file=sys.stderr,
        )
        return 2

    try:
        device = args.port or find_serial_port()
    except Exception as exc:
        print(f"Serial device detection failed: {exc}", file=sys.stderr)
        return 1

    rpm = clamp(args.rpm, MIN_RPM, MAX_RPM)
    turn_scale = clamp(args.turn_scale, 0.40, 1.00)
    deadman = clamp(args.deadman, 0.35, 1.00)

    try:
        ser = serial.Serial(
            device,
            args.baud,
            timeout=0,
            write_timeout=1.0,
            exclusive=True,
        )
    except TypeError:
        # Older pyserial versions may not expose the Linux exclusive option.
        ser = serial.Serial(device, args.baud, timeout=0, write_timeout=1.0)
    except Exception as exc:
        print(f"Cannot open {device}: {exc}", file=sys.stderr)
        print("Check permissions and make sure no serial monitor owns the port.")
        return 1

    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)

    try:
        print(f"Opened {device} @ {args.baud}.")
        if args.reset_delay > 0:
            print(f"Waiting {args.reset_delay:.1f}s for Mega reset/startup...")
            time.sleep(args.reset_delay)

        ok, startup_lines, rx_buffer = verify_link(ser)
        for line in startup_lines:
            if line.startswith("READY ") or line.startswith("ERR "):
                print(f"Mega: {line}")

        if not ok:
            print("Mega did not return STATE after STATUS.", file=sys.stderr)
            print(
                "Try --port explicitly and verify that the working serial monitor "
                "uses the same device and 115200 baud.",
                file=sys.stderr,
            )
            return 1

        print("Mega serial link verified (received STATE).")
        print(f"refresh={REFRESH_SEC:.2f}s, deadman={deadman:.2f}s")
        print_help(rpm, turn_scale)

        active_key: str | None = None
        command_deadline = 0.0
        next_refresh = 0.0
        show_next_state = False

        tty.setcbreak(fd)

        while True:
            now = time.monotonic()

            readable, _, _ = select.select([sys.stdin], [], [], 0.02)
            if readable:
                key = sys.stdin.read(1).lower()

                if key in "wasd":
                    active_key = key
                    command_deadline = now + deadman
                    left, right = send_drive(ser, active_key, rpm, turn_scale)
                    next_refresh = now + REFRESH_SEC
                    print(
                        f"\rDRIVE L={left:+.0f} R={right:+.0f} RPM            ",
                        end="",
                        flush=True,
                    )
                elif key in ("x", " "):
                    active_key = None
                    command_deadline = 0.0
                    send_stop(ser)
                    print("\rSTOP                                      ", end="", flush=True)
                elif key == "p":
                    show_next_state = True
                    write_line(ser, "STATUS")
                elif key in ("+", "="):
                    rpm = clamp(rpm + RPM_STEP, MIN_RPM, MAX_RPM)
                    print(
                        f"\rtarget={rpm:.0f} RPM, turn={rpm * turn_scale:.0f} RPM       ",
                        end="",
                        flush=True,
                    )
                elif key in ("-", "_"):
                    rpm = clamp(rpm - RPM_STEP, MIN_RPM, MAX_RPM)
                    print(
                        f"\rtarget={rpm:.0f} RPM, turn={rpm * turn_scale:.0f} RPM       ",
                        end="",
                        flush=True,
                    )
                elif key == "q":
                    break
                elif key == "h":
                    print_help(rpm, turn_scale)

            now = time.monotonic()
            if active_key is not None:
                if now >= command_deadline:
                    active_key = None
                    send_stop(ser)
                    print("\rDEADMAN STOP                               ", end="", flush=True)
                elif now >= next_refresh:
                    left, right = send_drive(ser, active_key, rpm, turn_scale)
                    next_refresh = now + REFRESH_SEC
                    print(
                        f"\rDRIVE L={left:+.0f} R={right:+.0f} RPM            ",
                        end="",
                        flush=True,
                    )

            for line in drain_lines(ser, rx_buffer):
                if line.startswith("ERR "):
                    print(f"\nMega: {line}", file=sys.stderr)
                elif line == "STOPPED":
                    print("\rMega: STOPPED                              ", end="", flush=True)
                elif show_next_state and line.startswith("STATE "):
                    print(f"\nMega: {line}")
                    show_next_state = False

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"\nController stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            if ser.is_open:
                try:
                    send_stop(ser)
                except Exception:
                    pass
                ser.close()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        print("\nStopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
