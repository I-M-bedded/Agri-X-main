# -*- coding: utf-8 -*-
"""Asynchronous CCRDNet (3-class) ONNX detector for the 20 Hz controller.

Drop-in for ``VisionLineDetector`` like ``ONNXFurrowLineDetector``, but for the
AgriCCRDNet-v0 model (classes: OTHER / STRUCTURE / NAV_BAND) trained by
``tools/train_ccrdnet.py``.  Two ways to obtain the navigation line, selected
by ``CCRDNET_LINE_SOURCE``:

  "nav_band"       the directly predicted central band (CCRDNet philosophy:
                   one navigation-critical target, no row association step)
  "structure_mid"  multi-line route: midline derived from the segmented
                   row/ridge STRUCTURE between the flanks of the tracked
                   corridor (works when the band head is weak, e.g. bare
                   seeded ridges)

Whichever is primary, the other is computed as a cross-check: a large
disagreement halves confidence, and if the primary finds no line the secondary
is used at reduced confidence.  Final gating stays where it always was —
``LineFollower`` only trusts results above ``VISION_MIN_CONFIDENCE``.

Inference is a latest-frame-only worker thread (no queue, no backlog), capped
by ``VISION_ONNX_SUBMIT_INTERVAL_SEC`` so ToF/pump/motor ticks never wait.
"""

from __future__ import annotations

import queue
from pathlib import Path
import threading
import time
from typing import Optional

import cv2
import numpy as np

from config import (
    CCRDNET_COMPONENT_MODE,
    CCRDNET_CROSSCHECK_MAX_DIFF_PX,
    CCRDNET_FALLBACK_CONF_SCALE,
    CCRDNET_LINE_SOURCE,
    CCRDNET_MIDLINE_BAND_WIDTH_PX,
    CCRDNET_ONNX_MODEL_PATH,
    SIGN_VISION_ERROR,
    VISION_ONNX_MAX_RESULT_AGE_SEC,
    VISION_ONNX_SUBMIT_INTERVAL_SEC,
    VISION_ONNX_THREADS,
)
from logutil import get_logger
from perception.ccrdnet.postprocess import (
    NavigationResult,
    PostprocessConfig,
    extract_navigation_line,
)
from perception.ccrdnet.structure_midline import derive_nav_band
from sensors.vision_line_detector import VisionLineResult

log = get_logger("ccrdnet")

_CLASS_STRUCTURE = 1
_CLASS_NAV_BAND = 2


class CCRDNetLineDetector:
    """Drop-in, non-blocking replacement for ``VisionLineDetector``."""

    def __init__(self, camera, model_path: str | Path = CCRDNET_ONNX_MODEL_PATH):
        self._camera = camera
        path = Path(model_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        self.model_path = path
        self.available = False
        self.last_error = ""
        self._postprocess = PostprocessConfig(component_mode=CCRDNET_COMPONENT_MODE)
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
            shape = self._session.get_inputs()[0].shape  # [1, 3, H, W]
            self._input_hw = (int(shape[2]), int(shape[3]))
            self._thread = threading.Thread(
                target=self._worker, name="ccrdnet-onnx", daemon=True
            )
            self._thread.start()
            self.available = True
            log.info("CCRDNet ONNX 모델 준비: %s (입력 %s, 소스 %s)",
                     path, self._input_hw, CCRDNET_LINE_SOURCE)
        except Exception as exc:
            self.last_error = str(exc)
            log.error("CCRDNet ONNX 모델을 열지 못했습니다: %s", exc)

    # ------------------------------------------------------------------
    def _lines(self, frame: np.ndarray):
        """Run the network; return (nav_band result, structure_mid result)."""
        h, w = self._input_hw
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        tensor = np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))[None]
        logits = self._session.run(None, {self._input_name: tensor})[0][0]
        shifted = logits - logits.max(axis=0, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.maximum(1e-8, exp.sum(axis=0))
        classes = probs.argmax(axis=0).astype(np.uint8)

        nav_mask = (classes == _CLASS_NAV_BAND).astype(np.uint8)
        direct = extract_navigation_line(
            nav_mask, self._postprocess, probs[_CLASS_NAV_BAND]
        )

        structure = (classes == _CLASS_STRUCTURE).astype(np.uint8)
        mid_band = derive_nav_band(
            structure, CCRDNET_MIDLINE_BAND_WIDTH_PX,
            max_missing_rows=max(8, h // 6),
        )
        derived = extract_navigation_line(mid_band, self._postprocess)
        return direct, derived

    def _infer(self, frame: np.ndarray) -> VisionLineResult:
        direct, derived = self._lines(frame)
        if CCRDNET_LINE_SOURCE == "structure_mid":
            primary, secondary = derived, direct
        else:
            primary, secondary = direct, derived

        result: NavigationResult
        confidence_scale = 1.0
        if primary.line is not None:
            result = primary
            if CCRDNET_CROSSCHECK_MAX_DIFF_PX > 0 and secondary.line is not None:
                diff = abs(primary.line.x_near - secondary.line.x_near)
                if diff > CCRDNET_CROSSCHECK_MAX_DIFF_PX:
                    confidence_scale = 0.5  # 두 기하 해석이 크게 어긋남
        elif secondary.line is not None:
            result = secondary
            confidence_scale = CCRDNET_FALLBACK_CONF_SCALE
        else:
            return VisionLineResult(0.0, 0.0, 0.0, 0.0)

        estimate = result.estimate
        return VisionLineResult(
            normalized_error=SIGN_VISION_ERROR * estimate.normalized_error,
            heading_error=SIGN_VISION_ERROR * estimate.heading_error,
            confidence=estimate.confidence * confidence_scale,
            coverage=estimate.coverage,
        )

    # ------------------------------------------------------------------
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
                log.warning("CCRDNet 추론 실패: %s", exc)

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
