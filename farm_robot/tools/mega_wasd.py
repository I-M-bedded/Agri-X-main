#!/usr/bin/env python3
"""Direct Raspberry Pi keyboard bridge to the Mega firmware's WASD interface.

This intentionally uses the exact same single-byte commands as Arduino Serial
Monitor: w/a/s/d/x/p. It is a hardware bring-up tool, not the navigation stack.

The active motion byte is refreshed periodically so the Mega's 400 ms DRIVE
watchdog stays alive while a key is held/repeated. If keyboard input stops,
the Pi sends x after the dead-man timeout. The Mega watchdog remains the final
independent safety layer if this process or USB link dies.
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
DEFAULT_DEADMAN_SEC = 0.70


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


def print_help() -> None:
    print(
        "\nControls: W forward | S reverse | A left | D right | "
        "X/Space stop | P state | Q quit"
    )
    print(
        "Commands are the same raw bytes used by Arduino Serial Monitor. "
        "Motion speed is firmware MANUAL_RPM.\n"
    )


def write_key(ser, key: str) -> None:
    ser.write(key.encode("ascii"))
    ser.flush()


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
    """Ask for state using raw 'p' and confirm that the Mega answers."""
    rx_buffer = bytearray()
    seen: list[str] = []

    write_key(ser, "p")
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
        description="Raw Serial-Monitor-equivalent WASD test for Agri-X Mega"
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
        "--deadman",
        type=float,
        default=DEFAULT_DEADMAN_SEC,
        help="seconds without a motion key before raw 'x' STOP (default: 0.70)",
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

    deadman = clamp(args.deadman, 0.20, 2.00)

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
            print("Mega did not return STATE after raw 'p'.", file=sys.stderr)
            print(
                "This means the Pi serial path itself is not matching the working "
                "Serial Monitor path. Try --port explicitly and verify baud=115200.",
                file=sys.stderr,
            )
            return 1

        print("Mega raw serial link verified (received STATE).")
        print_help()

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
                    next_refresh = 0.0
                elif key in ("x", " "):
                    active_key = None
                    command_deadline = 0.0
                    write_key(ser, "x")
                    print("\rSTOP                              ", end="", flush=True)
                elif key == "p":
                    show_next_state = True
                    write_key(ser, "p")
                elif key == "q":
                    break
                elif key == "h":
                    print_help()

            now = time.monotonic()
            if active_key is not None:
                if now >= command_deadline:
                    active_key = None
                    write_key(ser, "x")
                    print("\rDEADMAN STOP                       ", end="", flush=True)
                elif now >= next_refresh:
                    write_key(ser, active_key)
                    next_refresh = now + REFRESH_SEC
                    print(
                        f"\rTX raw '{active_key}' every {REFRESH_SEC:.2f}s   ",
                        end="",
                        flush=True,
                    )

            for line in drain_lines(ser, rx_buffer):
                if line.startswith("ERR "):
                    print(f"\nMega: {line}", file=sys.stderr)
                elif line == "STOPPED":
                    print("\rMega: STOPPED                      ", end="", flush=True)
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
                    write_key(ser, "x")
                except Exception:
                    pass
                ser.close()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        print("\nStopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
