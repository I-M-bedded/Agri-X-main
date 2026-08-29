# -*- coding: utf-8 -*-
"""Backend-neutral perception data contracts for the lightweight mission FSM.

Concrete perception models (SegFormer, YOLO segmentation, etc.) should expose
these values without coupling navigation/control code to a specific network.
"""

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen=True)
class FurrowEstimate:
    """Geometric estimate of the traversable corridor."""

    normalized_error: float = 0.0   # + = corridor centre is to robot's right
    heading_error: float = 0.0      # + = corridor heading bends to the right
    confidence: float = 0.0
    coverage: float = 0.0


@dataclass(frozen=True)
class PerceptionSnapshot:
    """Latest asynchronous perception result consumed by the mission FSM."""

    timestamp: float
    inference_sec: float
    furrow: FurrowEstimate
    obstacle_detected: bool
    obstacle_label: str = ""
    obstacle_confidence: float = 0.0
    obstacle_corridor_overlap: float = 0.0


class FieldPerception(Protocol):
    """Minimal interface required by ``AgriPipelineFSM``."""

    ready: bool
    last_error: str

    def submit(self, frame) -> None:
        ...

    def snapshot(self) -> Optional[PerceptionSnapshot]:
        ...

    def age_sec(self) -> float:
        ...
