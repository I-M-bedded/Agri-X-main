# -*- coding: utf-8 -*-
"""
navigation/furrow_manager.py
-----------------------------
고랑 번호와 완료 여부를 관리한다. 고랑 개수는 코드에 하드코딩하지 않는다.

[주의] 이 파일의 예전 docstring 은 "다음 고랑 마커를 못 찾으면 임무 완료로
처리한다"고 적혀 있었는데, 이는 mission_state_machine 의 실제 안전 설계와
정반대였다(부재 기반 추론 금지). 문서와 코드가 어긋나면 나중에 누군가
문서를 믿고 고치다가 안전 로직을 되돌려 놓게 되므로 여기서 바로잡는다.

실제 규칙:
  - 완료 확정은 전용 "밭 끝(END) 마커"를 **직접 관측**했을 때만.
  - 다음 고랑 마커가 안 보이는 것은 "모른다"이며 SAFE_HALT 사유다.
"""

from dataclasses import dataclass, field
from typing import List

from config import furrow_marker_id


@dataclass
class FurrowManager:
    current_index: int = 0                 # 완료한 고랑 번호 (0 = 아직 시작 전)
    completed: List[int] = field(default_factory=list)
    attempted: List[int] = field(default_factory=list)

    def next_index(self) -> int:
        """다음에 방문해야 할 고랑 번호."""
        return self.current_index + 1

    def next_marker_id(self):
        """다음 고랑 입구 팻말의 마커 ID (고랑당 1개)."""
        return furrow_marker_id(self.next_index())

    def mark_attempt(self):
        idx = self.next_index()
        if idx not in self.attempted:
            self.attempted.append(idx)

    def mark_current_done(self):
        """현재 진행하던 고랑을 완료 처리."""
        idx = self.next_index()
        if idx not in self.completed:
            self.completed.append(idx)
        self.current_index = idx

    def total_completed(self) -> int:
        return len(self.completed)

    def summary(self) -> str:
        return (
            f"완료 {len(self.completed)}개 {self.completed} / "
            f"시도 {len(self.attempted)}개"
        )

    def reset(self):
        self.current_index = 0
        self.completed = []
        self.attempted = []
