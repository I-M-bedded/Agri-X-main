# -*- coding: utf-8 -*-
"""
sensors/camera.py
------------------
전방 RGB 카메라 캡처 공용 모듈.
카메라 장치는 보통 한 번에 하나의 프로그램만 열 수 있으므로,
ArUco 검출과 비전 라인검출이 이 객체 하나를 공유한다.

[중요] capture_frame() 은 **항상 BGR** 배열을 반환한다.
  picamera2 의 format="RGB888" 은 실제 numpy 배열에서 BGR 순서로 나온다
  (picamera2 의 잘 알려진 함정). cv2.VideoCapture 도 BGR 이다.
  예전 코드는 이 프레임에 COLOR_RGB2HSV 를 적용해서 흙(H≈10, 주황갈색)이
  파랑(H≈110)으로 뒤집혔고, SOIL_HSV 범위에 절대 들어가지 않아
  비전 신뢰도가 항상 0 이었다 -> 주 제어가 사실상 죽어 있었다.
"""

import time
from typing import Optional

from config import CAMERA_INDEX, CAMERA_RESOLUTION, CAMERA_WARMUP_SEC
from logutil import get_logger

log = get_logger("camera")

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class Camera:
    def __init__(self, use_picamera2: bool = True):
        self._use_picamera2 = use_picamera2
        self._cap = None
        self._picam = None
        self.available = False
        self.fail_count = 0
        self._init_camera()

    # ------------------------------------------------------------------
    def _init_camera(self):
        if self._use_picamera2:
            try:
                from picamera2 import Picamera2

                self._picam = Picamera2()
                cfg = self._picam.create_video_configuration(
                    main={"size": tuple(CAMERA_RESOLUTION), "format": "RGB888"}
                )
                self._picam.configure(cfg)
                self._picam.start()
                time.sleep(CAMERA_WARMUP_SEC)
                self.available = True
                log.info("picamera2 초기화 완료 %s (출력=BGR)", CAMERA_RESOLUTION)
                return
            except Exception as exc:
                log.info("picamera2 사용 불가(%s) - V4L2 로 대체 시도", exc)
                self._picam = None

        if not _HAS_CV2:
            log.warning("OpenCV 없음 - 카메라를 사용할 수 없습니다.")
            return

        try:
            self._cap = cv2.VideoCapture(CAMERA_INDEX)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
            # 버퍼에 쌓인 오래된 프레임을 쓰지 않도록 최소화 (지원 시)
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            self.available = bool(self._cap.isOpened())
            if self.available:
                log.info("V4L2 카메라 초기화 완료 %s (출력=BGR)", CAMERA_RESOLUTION)
            else:
                log.warning("카메라를 열지 못했습니다 (index=%s)", CAMERA_INDEX)
        except Exception as exc:
            log.error("카메라 초기화 실패: %s", exc)
            self._cap = None

    # ------------------------------------------------------------------
    def capture_frame(self) -> Optional["np.ndarray"]:
        """가장 최근 프레임 1장을 BGR numpy 배열로 반환. 실패 시 None."""
        try:
            if self._picam is not None:
                frame = self._picam.capture_array()
                self.fail_count = 0
                return frame
            if self._cap is not None:
                ok, frame = self._cap.read()
                if ok:
                    self.fail_count = 0
                    return frame
        except Exception as exc:
            log.warning("프레임 캡처 예외: %s", exc)

        self.fail_count += 1
        return None

    def healthy(self) -> bool:
        return self.available and self.fail_count < 20

    def close(self):
        try:
            if self._picam is not None:
                self._picam.stop()
                self._picam.close()
        except Exception:
            pass
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self.available = False
