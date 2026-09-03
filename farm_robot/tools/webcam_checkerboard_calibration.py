# -*- coding: utf-8 -*-
"""USB webcam checkerboard calibration for Agri-X.

The default pattern is 9 x 6 inner corners (a printed board has 10 x 7
squares). Measure one square and pass its size with --square-mm.

Keys: SPACE capture, A auto capture, U undo, C/ENTER calibrate, Q/ESC quit.
The default output is calibration/webcam_0.json.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit("opencv-contrib-python and numpy are required") from exc


FARM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = FARM_ROOT / "calibration" / "webcam_0.json"
sys.path.insert(0, str(FARM_ROOT))

from config import CAMERA_INDEX, CAMERA_RESOLUTION  # noqa: E402


def make_object_points(cols, rows, square_m):
    points = np.zeros((rows * cols, 3), np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points[:, :2] *= float(square_m)
    return points


def find_corners(gray, board_size):
    """Return sub-pixel checkerboard corners."""
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray, board_size, cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            return True, corners.astype(np.float32)
        return False, None

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, board_size, flags)
    if not found:
        return False, None
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    return True, cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def view_descriptor(corners, image_size):
    """Describe image position, size and rotation for duplicate rejection."""
    width, height = image_size
    xy = corners.reshape(-1, 2)
    low = xy.min(axis=0)
    high = xy.max(axis=0)
    center = (low + high) * 0.5
    area = ((high[0] - low[0]) * (high[1] - low[1])) / max(1.0, width * height)
    edge = xy[-1] - xy[0]
    angle = math.atan2(float(edge[1]), float(edge[0]))
    return np.asarray(
        [center[0] / width, center[1] / height, math.sqrt(max(0.0, area)),
         angle / math.pi],
        dtype=np.float64,
    )


def is_diverse(descriptor, saved_descriptors):
    if not saved_descriptors:
        return True
    scales = np.asarray([0.16, 0.16, 0.10, 0.18], dtype=np.float64)
    return all(
        np.linalg.norm((descriptor - old) / scales) >= 1.0
        for old in saved_descriptors
    )


def reprojection_errors(objects, images, rvecs, tvecs, matrix, distortion):
    values = []
    for obj, image, rvec, tvec in zip(objects, images, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, matrix, distortion)
        error = cv2.norm(image, projected, cv2.NORM_L2) / math.sqrt(len(projected))
        values.append(float(error))
    return np.asarray(values, dtype=np.float64)


def _fit(image_points, image_size, template):
    objects = [template.copy() for _ in image_points]
    rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        objects, image_points, image_size, None, None)
    errors = reprojection_errors(
        objects, image_points, rvecs, tvecs, matrix, distortion)
    return rms, matrix, distortion, errors


def calibrate(image_points, image_size, cols, rows, square_m):
    """Fit once, reject clear outlier views, then fit again."""
    original_count = len(image_points)
    template = make_object_points(cols, rows, square_m)
    rms, matrix, distortion, errors = _fit(image_points, image_size, template)

    median = float(np.median(errors))
    mad = float(np.median(np.abs(errors - median)))
    cutoff = max(0.5, median * 2.0, median + 3.0 * max(mad, 1e-6))
    keep = [i for i, error in enumerate(errors) if error <= cutoff]
    if 10 <= len(keep) < original_count:
        image_points = [image_points[i] for i in keep]
        rms, matrix, distortion, errors = _fit(
            image_points, image_size, template)

    return {
        "rms": float(rms),
        "camera_matrix": matrix,
        "dist_coeffs": distortion.reshape(-1),
        "errors": errors,
        "samples_used": len(image_points),
        "samples_rejected": original_count - len(image_points),
    }


def save_result(path, result, args, image_size, captured_count):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = result["errors"]
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "camera_index": args.camera_index,
        "image_width": image_size[0],
        "image_height": image_size[1],
        "checkerboard_inner_corners": [args.cols, args.rows],
        "square_size_mm": args.square_mm,
        "samples_captured": captured_count,
        "samples_used": result["samples_used"],
        "samples_rejected": result["samples_rejected"],
        "rms_reprojection_error_px": result["rms"],
        "mean_reprojection_error_px": float(np.mean(errors)),
        "max_reprojection_error_px": float(np.max(errors)),
        "camera_matrix": result["camera_matrix"].tolist(),
        "dist_coeffs": result["dist_coeffs"].tolist(),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path, payload


def draw_overlay(frame, found, count, target, auto, message):
    dark = frame.copy()
    cv2.rectangle(dark, (0, 0), (frame.shape[1], 100), (0, 0, 0), -1)
    cv2.addWeighted(dark, 0.55, frame, 0.45, 0, frame)
    color = (80, 230, 100) if found else (70, 180, 255)
    cv2.putText(
        frame, f"CHECKERBOARD: {'FOUND' if found else 'NOT FOUND'}", (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    cv2.putText(
        frame, f"SAMPLES: {count}/{target}   AUTO: {'ON' if auto else 'OFF'}",
        (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1,
        cv2.LINE_AA)
    cv2.putText(
        frame, "SPACE capture | A auto | U undo | C calibrate | Q quit",
        (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 190), 1,
        cv2.LINE_AA)
    if message:
        cv2.putText(
            frame, message, (12, frame.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (70, 220, 255), 2,
            cv2.LINE_AA)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Agri-X USB webcam checkerboard calibration")
    parser.add_argument("--camera-index", type=int, default=CAMERA_INDEX)
    parser.add_argument("--width", type=int, default=CAMERA_RESOLUTION[0])
    parser.add_argument("--height", type=int, default=CAMERA_RESOLUTION[1])
    parser.add_argument("--cols", type=int, default=9,
                        help="horizontal inner corners (default: 9)")
    parser.add_argument("--rows", type=int, default=6,
                        help="vertical inner corners (default: 6)")
    parser.add_argument("--square-mm", type=float, required=True,
                        help="measured size of one checker square in mm")
    parser.add_argument("--target", type=int, default=20)
    parser.add_argument("--min-samples", type=int, default=12)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    if args.cols < 3 or args.rows < 3 or args.square_mm <= 0:
        parser.error("board dimensions and --square-mm must be positive")
    if args.min_samples < 8 or args.target < args.min_samples:
        parser.error("--target must be >= --min-samples, and minimum is 8")
    return args


def main(argv=None):
    args = parse_args(argv)
    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if not cap.isOpened():
        cap.release()
        print(f"Cannot open webcam index {args.camera_index}", file=sys.stderr)
        return 2

    samples = []
    descriptors = []
    image_size = None
    auto = args.auto
    last_auto = 0.0
    message = "Move board across corners, distances, and tilts"
    message_until = time.monotonic() + 5.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                message = "CAMERA READ FAILED"
                message_until = time.monotonic() + 2.0
                continue

            image_size = (frame.shape[1], frame.shape[0])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = find_corners(gray, (args.cols, args.rows))
            descriptor = None
            if found:
                cv2.drawChessboardCorners(
                    frame, (args.cols, args.rows), corners, True)
                descriptor = view_descriptor(corners, image_size)

            now = time.monotonic()
            if auto and found and now - last_auto >= 0.7:
                last_auto = now
                if is_diverse(descriptor, descriptors):
                    samples.append(corners.copy())
                    descriptors.append(descriptor)
                    message = "CAPTURED"
                else:
                    message = "Move/tilt board more (duplicate view)"
                message_until = now + 0.8

            draw_overlay(
                frame, found, len(samples), args.target, auto,
                message if now < message_until else "")
            cv2.imshow("Agri-X webcam checkerboard calibration", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                print("Calibration cancelled")
                return 0
            if key == ord("a"):
                auto = not auto
                message = f"AUTO {'ON' if auto else 'OFF'}"
                message_until = now + 1.5
            elif key == ord("u") and samples:
                samples.pop()
                descriptors.pop()
                message = "LAST SAMPLE REMOVED"
                message_until = now + 1.5
            elif key == ord(" "):
                if not found:
                    message = "Checkerboard not found"
                elif not is_diverse(descriptor, descriptors):
                    message = "Duplicate view: move or tilt board"
                else:
                    samples.append(corners.copy())
                    descriptors.append(descriptor)
                    message = "CAPTURED"
                message_until = now + 1.5
            elif key in (ord("c"), 10, 13):
                if len(samples) >= args.min_samples:
                    break
                message = f"Need at least {args.min_samples} samples"
                message_until = now + 2.0

            if auto and len(samples) >= args.target:
                break
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    print(
        f"Calculating calibration: {len(samples)} samples, "
        f"{image_size[0]}x{image_size[1]}")
    result = calibrate(
        samples, image_size, args.cols, args.rows, args.square_mm / 1000.0)
    saved, payload = save_result(
        args.output, result, args, image_size, len(samples))

    error = payload["mean_reprojection_error_px"]
    quality = "GOOD" if error < 0.7 else "USABLE" if error < 1.2 else "REDO"
    print(f"Saved: {saved}")
    print(f"RMS error: {payload['rms_reprojection_error_px']:.4f} px")
    print(f"Mean reprojection error: {error:.4f} px [{quality}]")
    print(
        f"Samples used: {payload['samples_used']}/{len(samples)} "
        f"(rejected {payload['samples_rejected']})")
    if quality == "REDO":
        print("Redo with fixed focus and more edge/tilted views.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
