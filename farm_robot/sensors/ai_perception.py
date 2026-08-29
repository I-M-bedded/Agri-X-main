# -*- coding: utf-8 -*-
"""Asynchronous zero-shot perception for the lightweight mission pipeline.

One YOLOE segmentation model is shared for two jobs:
  * furrow region segmentation -> geometric centre-line estimate
  * always-on near-field obstacle / non-traversable-area watchdog

The inference worker keeps only the newest frame.  Old frames are dropped instead
of queued so a slow Raspberry Pi never builds seconds of perception latency.
For Pi 4 deployment, export the prompted YOLOE-26n model to NCNN first; see
``tools/export_yoloe_rpi.py``.
"""

from dataclasses import dataclass
import os
import threading
import time
from typing import Optional

from logutil import get_logger

log = get_logger("ai-perception")

try:
    import cv2
    import numpy as np
    _HAS_CV = True
except ImportError:
    cv2 = None
    np = None
    _HAS_CV = False


FURROW_PROMPTS = (
    "furrow",
    "farm furrow",
    "soil trench",
)

# Keep this list deliberately short.  YOLOE compares region features with every
# prompt on every inference, so a large vocabulary directly increases latency.
OBSTACLE_PROMPTS = (
    "person",
    "animal",
    "vehicle",
    "tractor",
    "rock",
    "log",
    "box",
    "farm equipment",
    "hole",
    "water puddle",
)

PROMPT_CLASSES = FURROW_PROMPTS + OBSTACLE_PROMPTS


@dataclass(frozen=True)
class FurrowEstimate:
    normalized_error: float = 0.0   # + = centre is to the robot's right
    heading_error: float = 0.0      # + = furrow heads to the right
    confidence: float = 0.0
    coverage: float = 0.0


@dataclass(frozen=True)
class PerceptionSnapshot:
    timestamp: float
    inference_sec: float
    furrow: FurrowEstimate
    obstacle_detected: bool
    obstacle_label: str = ""
    obstacle_confidence: float = 0.0
    obstacle_corridor_overlap: float = 0.0


class ZeroShotFieldPerception:
    """Latest-frame asynchronous YOLOE segmentation worker."""

    def __init__(
        self,
        model_path: str = "yoloe-26n-seg.pt",
        imgsz: int = 320,
        inference_hz: float = 2.0,
        conf_threshold: float = 0.25,
        obstacle_overlap_threshold: float = 0.015,
        cpu_threads: int = 3,
    ):
        self.model_path = model_path
        self.imgsz = int(imgsz)
        self.inference_hz = max(0.2, float(inference_hz))
        self.conf_threshold = float(conf_threshold)
        self.obstacle_overlap_threshold = float(obstacle_overlap_threshold)

        self.ready = False
        self.last_error = ""
        self._model = None
        self._latest_frame = None
        self._snapshot: Optional[PerceptionSnapshot] = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None

        if not _HAS_CV:
            self.last_error = "opencv/numpy unavailable"
            log.error(self.last_error)
            return

        try:
            # Leave one Pi 4 core for the 20 Hz control loop and camera/IO.
            try:
                import torch
                torch.set_num_threads(max(1, int(cpu_threads)))
            except Exception:
                pass

            from ultralytics import YOLO, YOLOE

            # Exported NCNN/ONNX models already contain the baked class profile.
            if os.path.exists(model_path) and not model_path.endswith(".pt"):
                self._model = YOLO(model_path)
            else:
                # For development this can auto-download the public checkpoint.
                # On the Pi, prefer an offline exported NCNN directory.
                self._model = YOLOE(model_path)
                self._model.set_classes(list(PROMPT_CLASSES))

            self.ready = True
            self._thread = threading.Thread(
                target=self._worker,
                name="field-perception",
                daemon=True,
            )
            self._thread.start()
            log.info(
                "zero-shot perception ready: model=%s imgsz=%d max=%.1fHz",
                model_path, self.imgsz, self.inference_hz,
            )
        except Exception as exc:
            self.last_error = str(exc)
            log.error("zero-shot perception init failed: %s", exc)

    # ------------------------------------------------------------------
    def submit(self, frame) -> None:
        """Submit the newest BGR frame; any older pending frame is discarded."""
        if not self.ready or frame is None:
            return
        with self._lock:
            self._latest_frame = frame.copy()
        self._wake.set()

    def snapshot(self) -> Optional[PerceptionSnapshot]:
        with self._lock:
            return self._snapshot

    def age_sec(self) -> float:
        snap = self.snapshot()
        if snap is None:
            return float("inf")
        return max(0.0, time.monotonic() - snap.timestamp)

    # ------------------------------------------------------------------
    def _worker(self):
        min_period = 1.0 / self.inference_hz
        next_allowed = 0.0

        while not self._stop.is_set():
            self._wake.wait(timeout=0.1)
            self._wake.clear()
            if self._stop.is_set():
                break

            now = time.monotonic()
            if now < next_allowed:
                time.sleep(min(next_allowed - now, 0.05))
                self._wake.set()
                continue

            with self._lock:
                frame = self._latest_frame
                self._latest_frame = None
            if frame is None:
                continue

            start = time.monotonic()
            try:
                result = self._model.predict(
                    frame,
                    imgsz=self.imgsz,
                    conf=self.conf_threshold,
                    verbose=False,
                )[0]
                snapshot = self._parse_result(result, frame.shape[:2], start)
                with self._lock:
                    self._snapshot = snapshot
                self.last_error = ""
            except Exception as exc:
                self.last_error = str(exc)
                log.warning("zero-shot inference failed: %s", exc)

            next_allowed = time.monotonic() + min_period

    # ------------------------------------------------------------------
    @staticmethod
    def _class_name(names, cls_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(cls_id, cls_id))
        try:
            return str(names[cls_id])
        except Exception:
            return str(cls_id)

    @staticmethod
    def _to_numpy(value):
        if value is None:
            return None
        try:
            return value.detach().cpu().numpy()
        except Exception:
            return np.asarray(value)

    def _parse_result(self, result, hw, start_time: float) -> PerceptionSnapshot:
        h, w = hw
        furrow_masks = []
        obstacle_masks = []

        boxes = getattr(result, "boxes", None)
        masks_obj = getattr(result, "masks", None)
        if boxes is not None and masks_obj is not None and masks_obj.data is not None:
            cls_ids = self._to_numpy(boxes.cls).astype(int)
            confs = self._to_numpy(boxes.conf).astype(float)
            masks = self._to_numpy(masks_obj.data)
            names = getattr(result, "names", {})

            count = min(len(cls_ids), len(confs), len(masks))
            for i in range(count):
                conf = float(confs[i])
                if conf < self.conf_threshold:
                    continue
                label = self._class_name(names, int(cls_ids[i])).strip().lower()
                mask = masks[i]
                if mask.shape != (h, w):
                    mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                binary = mask > 0.5

                if label in FURROW_PROMPTS:
                    furrow_masks.append((binary, conf, label))
                else:
                    obstacle_masks.append((binary, conf, label))

        furrow = self._estimate_furrow(furrow_masks, h, w)
        detected, label, conf, overlap = self._estimate_obstacle(obstacle_masks, h, w)
        end = time.monotonic()
        return PerceptionSnapshot(
            timestamp=end,
            inference_sec=end - start_time,
            furrow=furrow,
            obstacle_detected=detected,
            obstacle_label=label,
            obstacle_confidence=conf,
            obstacle_corridor_overlap=overlap,
        )

    # ------------------------------------------------------------------
    def _select_furrow_component(self, mask, h: int, w: int):
        """Choose the component connected most strongly to the robot's near ROI."""
        u8 = mask.astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
        if n <= 1:
            return mask

        y0 = int(h * 0.72)
        x0, x1 = int(w * 0.28), int(w * 0.72)
        best_label = 0
        best_score = -1.0
        for lab in range(1, n):
            component = labels == lab
            area = float(stats[lab, cv2.CC_STAT_AREA])
            near = float(component[y0:h, x0:x1].sum())
            bottom = float(component[int(h * 0.88):h, :].sum())
            score = near * 4.0 + bottom * 2.0 + area * 0.05
            if score > best_score:
                best_score = score
                best_label = lab
        return labels == best_label if best_label else mask

    def _estimate_furrow(self, candidates, h: int, w: int) -> FurrowEstimate:
        if not candidates:
            return FurrowEstimate()

        # Merge aliases, then retain the component that actually reaches the robot.
        combined = np.zeros((h, w), dtype=bool)
        best_conf = 0.0
        for mask, conf, _ in candidates:
            combined |= mask
            best_conf = max(best_conf, conf)
        combined = self._select_furrow_component(combined, h, w)

        roi_y0, roi_y1 = int(h * 0.45), int(h * 0.95)
        roi = combined[roi_y0:roi_y1, :]
        if roi.size == 0:
            return FurrowEstimate()
        coverage = float(roi.mean())

        band_count = 6
        centers = []
        roi_h = roi.shape[0]
        for band_idx in range(band_count):
            b0 = int(roi_h * band_idx / band_count)
            b1 = int(roi_h * (band_idx + 1) / band_count)
            ys, xs = np.nonzero(roi[b0:b1, :])
            if xs.size < max(20, int((b1 - b0) * w * 0.01)):
                continue
            # Midpoint of both boundaries is more stable than the pixel centroid
            # when one ridge casts a larger shadow than the other.
            left = float(np.percentile(xs, 5))
            right = float(np.percentile(xs, 95))
            centre = 0.5 * (left + right)
            x_norm = (centre - 0.5 * w) / (0.5 * w)
            y_norm = (b0 + b1) * 0.5 / max(1.0, roi_h)
            centers.append((y_norm, x_norm))

        if len(centers) < 2:
            return FurrowEstimate(confidence=0.0, coverage=coverage)

        ys = np.asarray([p[0] for p in centers], dtype=np.float32)
        xs = np.asarray([p[1] for p in centers], dtype=np.float32)
        slope, intercept = np.polyfit(ys, xs, 1)
        predicted = slope * ys + intercept
        residual = float(np.std(xs - predicted))

        offset = float(np.clip(slope + intercept, -1.0, 1.0))
        heading = float(np.clip(-slope, -1.0, 1.0))
        band_score = len(centers) / float(band_count)
        consistency = max(0.0, 1.0 - residual / 0.22)
        # Confidence is intentionally conservative: geometric consistency must
        # accompany the network's class confidence.
        confidence = float(np.clip(best_conf * band_score * consistency, 0.0, 1.0))
        return FurrowEstimate(offset, heading, confidence, coverage)

    def _estimate_obstacle(self, candidates, h: int, w: int):
        if not candidates:
            return False, "", 0.0, 0.0

        corridor = np.zeros((h, w), dtype=np.uint8)
        polygon = np.asarray(
            [
                [int(w * 0.20), h - 1],
                [int(w * 0.80), h - 1],
                [int(w * 0.60), int(h * 0.52)],
                [int(w * 0.40), int(h * 0.52)],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(corridor, polygon, 1)
        corridor_area = max(1, int(corridor.sum()))

        best = (False, "", 0.0, 0.0)
        for mask, conf, label in candidates:
            intersection = int(np.logical_and(mask, corridor > 0).sum())
            overlap = intersection / float(corridor_area)
            if overlap > best[3]:
                best = (overlap >= self.obstacle_overlap_threshold, label, conf, overlap)
        return best

    # ------------------------------------------------------------------
    def close(self):
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.ready = False
