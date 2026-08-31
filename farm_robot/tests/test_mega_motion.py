from pathlib import Path
import queue
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control.mega_motion import MegaMotion, MegaProtocolError
from sensors.odometry import Odometry


class FakeSerial:
    def __init__(self, *args, **kwargs):
        self.rx = queue.Queue()
        self.rx.put("READY MEGA_MOTION_V2")
        self.writes = []
        self.closed = False
        self.on_write = None

    def reset_input_buffer(self):
        pass

    def write(self, payload):
        line = payload.decode("ascii").strip()
        self.writes.append(line)
        if self.on_write is not None:
            self.on_write(line)
        return len(payload)

    def readline(self):
        if self.closed:
            return b""
        try:
            return (self.rx.get(timeout=0.02) + "\n").encode("ascii")
        except queue.Empty:
            return b""

    def emit(self, line):
        self.rx.put(line)

    def close(self):
        self.closed = True


class FakeOdometry:
    def __init__(self):
        self.deltas = []

    def inject_wheel_degrees(self, left, right):
        self.deltas.append((left, right))


class MegaMotionTest(unittest.TestCase):
    def make_motion(self, odometry=None):
        serial = FakeSerial()
        motion = MegaMotion(
            odometry=odometry,
            port="FAKE",
            serial_factory=lambda *args, **kwargs: serial,
            reset_delay=0.0,
        )
        self.addCleanup(motion.cleanup)
        return motion, serial

    def test_drive_maps_normalized_arcade_command_to_rpm(self):
        motion, serial = self.make_motion()
        motion.drive(0.5, 0.2)
        self.assertEqual(serial.writes[-1], "DRIVE 56.000 24.000")
        self.assertAlmostEqual(motion.last_left, 0.7)
        self.assertAlmostEqual(motion.last_right, 0.3)

    def test_drive_saturates_without_losing_steering_difference(self):
        motion, serial = self.make_motion()
        motion.drive(0.9, 0.35)
        self.assertEqual(serial.writes[-1], "DRIVE 80.000 24.000")

    def test_move_waits_for_matching_ack_and_done(self):
        motion, serial = self.make_motion()

        def respond(line):
            if line.startswith("MOVE "):
                sequence = line.split()[1]
                serial.emit(f"ACK {sequence}")
                serial.emit(f"DONE {sequence}")

        serial.on_write = respond
        self.assertTrue(motion.move(90.0, -90.0, 40.0, timeout=1.0, sequence=17))
        self.assertEqual(serial.writes[0], "MOVE 17 90.000 -90.000 40.000")

    def test_state_angles_feed_odometry_as_deltas(self):
        odom = FakeOdometry()
        motion, serial = self.make_motion(odom)
        serial.emit("STATE 0 IDLE 10.00 20.00 0.00 0.00")
        serial.emit("STATE 0 MANUAL 13.50 18.00 5.00 -4.00")
        deadline = time.monotonic() + 0.5
        while not odom.deltas and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(odom.deltas, [(3.5, -2.0)])
        self.assertEqual(motion.state.mode, "MANUAL")

    def test_protocol_error_latches_fault_and_blocks_new_motion(self):
        motion, serial = self.make_motion()
        motion._handle_line("ERR DRIVE_TIMEOUT")
        before = list(serial.writes)

        self.assertTrue(motion.faulted)
        self.assertFalse(motion.set_speeds(0.5, 0.5))
        self.assertEqual(serial.writes, before)
        self.assertTrue(motion.stop())
        self.assertEqual(serial.writes[-1], "STOP")

        with self.assertRaises(MegaProtocolError):
            motion.move(90.0, 90.0, 40.0, timeout=0.2, sequence=9)

    def test_unexpected_ready_faults_and_resets_odometry_baseline(self):
        odom = FakeOdometry()
        motion, _serial = self.make_motion(odom)
        motion._handle_line("STATE 0 MANUAL 100.00 120.00 5.00 5.00")
        motion._handle_line("STATE 0 MANUAL 105.00 125.00 5.00 5.00")
        self.assertEqual(odom.deltas, [(5.0, 5.0)])

        motion._handle_line("READY MEGA_MOTION_V2")
        self.assertTrue(motion.faulted)
        self.assertIn("reboot", motion.last_error.lower())

        # Firmware counters restart from zero after a Mega reboot. That first
        # post-reset STATE must become a new baseline, not a -105/-125 deg jump.
        motion._handle_line("STATE 0 IDLE 0.00 0.00 0.00 0.00")
        self.assertEqual(odom.deltas, [(5.0, 5.0)])

    def test_cleanup_sends_stop(self):
        motion, serial = self.make_motion()
        motion.cleanup()
        self.assertIn("STOP", serial.writes)
        self.assertFalse(motion.available)

    def test_mega_wheel_degrees_update_odometry(self):
        odom = Odometry()
        self.addCleanup(odom.cleanup)
        odom.inject_wheel_degrees(360.0, 360.0)
        odom.update()
        self.assertAlmostEqual(odom.path_length, 2.0 * 3.141592653589793 * 0.033,
                               places=6)
        self.assertAlmostEqual(odom.theta, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()