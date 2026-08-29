# -*- coding: utf-8 -*-
"""ArUco detector variant that consumes a frame supplied by the main loop.

The legacy ArucoDetector captures its own frame.  The new mission pipeline keeps
camera ownership in one place so ArUco, AI segmentation and logging can all use
the same image without reopening or re-reading the camera.
"""

import math
from typing import Dict

from config import MARKER_CONFIRM_FRAMES, SIGN_MARKER_LATERAL, SIGN_MARKER_YAW
from sensors.aruco_detector import ArucoDetector, MarkerObservation

try:
    import cv2
    import numpy as np
    _HAS_CV = True
except ImportError:
    cv2 = None
    np = None
    _HAS_CV = False


class FrameArucoDetector(ArucoDetector):
    def detect_from_frame(self, frame) -> Dict[int, MarkerObservation]:
        if not _HAS_CV or frame is None or self._detector is None and not self._legacy:
            return {}

        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        corners, ids = self._detect_markers(gray)
        if ids is None or len(ids) == 0:
            self._decay_streaks(set())
            return {}

        raw: Dict[int, MarkerObservation] = {}
        for i, marker_id in enumerate(ids.flatten()):
            image_points = corners[i].reshape(4, 2).astype(np.float64)
            ok, rvec, tvec = cv2.solvePnP(
                self._object_points,
                image_points,
                self._camera_matrix,
                self._dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok:
                continue

            t = tvec.reshape(3)
            rot_mat, _ = cv2.Rodrigues(rvec)
            yaw = float(math.atan2(rot_mat[0, 2], rot_mat[2, 2]))
            mid = int(marker_id)
            raw[mid] = MarkerObservation(
                marker_id=mid,
                distance_m=float(np.linalg.norm(t)),
                forward_m=float(t[2]),
                lateral_offset_m=float(t[0]) * SIGN_MARKER_LATERAL,
                yaw_error_rad=yaw * SIGN_MARKER_YAW,
            )

        self._decay_streaks(set(raw.keys()))
        return {
            mid: obs
            for mid, obs in raw.items()
            if self._seen_streak.get(mid, 0) >= MARKER_CONFIRM_FRAMES
        }
