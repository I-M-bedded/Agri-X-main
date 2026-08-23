# -*- coding: utf-8 -*-
"""
logutil.py
-----------
print() 대신 쓰는 공용 로거. 레벨 필터링/파일 기록/타임스탬프를 한 곳에서 관리한다.
"""

import logging
import sys

from config import LOG_FILE_PATH, LOG_LEVEL, LOG_TO_FILE

_CONFIGURED = False


def _configure():
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger("farm_robot")
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    root.propagate = False

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    if LOG_TO_FILE:
        try:
            fh = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except OSError:
            pass

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(f"farm_robot.{name}")
