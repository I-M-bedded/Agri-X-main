from __future__ import annotations

"""Asynchronous ONNX row-boundary inference for the 20 Hz controller."""

import queue
from pathlib import Path
import threading
import time
from typing import Optional

import cv2
import numpy as np

from config import (
    SIGN_VISION_ERROR,
    VISION_ONNX_INPUT_SIZE,
    VISION_ONNX_LINE_THRESHOLD,
    VISION_ONNX_MAX_ABS_ERROR,
    VISION_ONNX_MAX_ABS_HEADING_ERROR,
    VISION_ONNX_MAX_HALF_THICKNESS_PX,
    VISION_ONNX_MAX_RESULT_AGE_SEC,
    VISION_ONNX_MAX_TOP_FRACTION,
    VISION_ONNX_MODEL_PATH,
    VISION_ONNX_SUBMIT_INTERVAL_SEC,
    VISION_ONNX_THREADS,
)
from logutil import get_logger
from sensors.furrow_geometry import boundary_mask_to_geometry
from sensors.vision_line_detector import VisionLineResult

log = get_logger("onnx-furrow")


class ONNXFurrowLineDetector:
    """Drop-in, non-blocking replacement for ``VisionLineDetector``.

    It produces steering geometry only and must not be treated as an obstacle
    detector. ``compute`` submits the newest frame and returns the last completed
    result, so network latency cannot stall ToF/pump/motor safety ticks.
    """

    def __init__(self, camera, model_path: str | Path = VISION_ONNX_MODEL_PATH):
        self._camera = camera
        path = Path(model_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        self.model_path = path
        self.available = False
        self.last_error = ""
        self._session = None
        self._input_name = ""
        self._output_name = ""
        self._frames: queue.Queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._latest: Optional[VisionLineResult] = None
        self._latest_time = 0.0
        self._last_submit = 0.0
        self._closed = False
        self._thread: Optional[threading.Thread] = None

        try:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, int(VISION_ONNX_THREADS))
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self._session = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            self.available = True
            self._thread = threading.Thread(
                target=self._worker, name="furrow-onnx", daemon=True
            )
            self._thread.start()
            log.info("ONNX 고랑선 모델 준비: %s", path)
        except Exception as exc:
            self.last_error = str(exc)
            log.error("ONNX 고랑선 모델을 열지 못했습니다: %s", exc)

    @staticmethod
    def _softmax_line_probability(logits: np.ndarray) -> np.ndarray:
        if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 2:
            raise ValueError(f"Expected [1,2,H,W] logits, got {logits.shape}")
        shifted = logits[0] - logits[0].max(axis=0, keepdims=True)
        exp = np.exp(shifted)
        return exp[0] / np.maximum(1e-8, exp.sum(axis=0))

    def _infer(self, frame: np.ndarray) -> VisionLineResult:
        width, height = VISION_ONNX_INPUT_SIZE
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - np.asarray((0.485, 0.456, 0.406), np.float32)) / np.asarray(
            (0.229, 0.224, 0.225), np.float32
        )
        tensor = np.transpose(tensor, (2, 0, 1))[None]
        logits = self._session.run([self._output_name], {self._input_name: tensor})[0]
        probability = self._softmax_line_probability(logits)
        probability = cv2.resize(
            probability, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR
        )
        mask = np.where(probability >= VISION_ONNX_LINE_THRESHOLD, 255, 0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        geometry = boundary_mask_to_geometry(mask)

        foreground = mask > 0
        top_fraction = float(foreground[: int(mask.shape[0] * 0.30)].mean())
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        half_thickness = float(distance[foreground].mean()) if foreground.any() else 0.0
        quality_ok = (
            geometry.centerline is not None
            and geometry.line_count >= 2
            and top_fraction <= VISION_ONNX_MAX_TOP_FRACTION
            and half_thickness <= VISION_ONNX_MAX_HALF_THICKNESS_PX
            and abs(geometry.estimate.normalized_error) <= VISION_ONNX_MAX_ABS_ERROR
            and abs(geometry.estimate.heading_error) <= VISION_ONNX_MAX_ABS_HEADING_ERROR
        )
        estimate = geometry.estimate
        return VisionLineResult(
            normalized_error=SIGN_VISION_ERROR * estimate.normalized_error,
            heading_error=SIGN_VISION_ERROR * estimate.heading_error,
            confidence=estimate.confidence if quality_ok else 0.0,
            coverage=estimate.coverage,
        )

    def _worker(self):
        while not self._closed:
            frame = self._frames.get()
            if frame is None:
                return
            try:
                result = self._infer(frame)
                with self._lock:
                    self._latest = result
                    self._latest_time = time.monotonic()
                self.last_error = ""
            except Exception as exc:
                self.last_error = str(exc)
                log.warning("ONNX 고랑선 추론 실패: %s", exc)

    def compute(self) -> Optional[VisionLineResult]:
        if not self.available:
            return VisionLineResult(0.0, 0.0, 0.0, 0.0)
        now = time.monotonic()
        if now - self._last_submit >= VISION_ONNX_SUBMIT_INTERVAL_SEC:
            frame = self._camera.capture_frame()
            if frame is not None:
                if self._frames.full():
                    try:
                        self._frames.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    self._frames.put_nowait(frame.copy())
                    self._last_submit = now
                except queue.Full:
                    pass
        with self._lock:
            result = self._latest
            age = now - self._latest_time
        if result is None or age > VISION_ONNX_MAX_RESULT_AGE_SEC:
            return VisionLineResult(0.0, 0.0, 0.0, 0.0)
        return result

    def cleanup(self):
        self._closed = True
        try:
            if self._frames.full():
                self._frames.get_nowait()
            self._frames.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.available = False
