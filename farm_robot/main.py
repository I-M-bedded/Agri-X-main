# -*- coding: utf-8 -*-
"""
main.py
--------
로봇 실행 진입점.

    $ python3 main.py                # 실제 임무 실행
    $ python3 selftest.py            # 하드웨어 없이 자체 점검 (실행 전 항상 권장)
    $ python3 tools/setup.py 2       # 바퀴를 들고 조향 부호 확인

Ctrl+C 로 중단하면 모터/펌프/GPIO가 안전하게 정리된다.
"""

import sys

from logutil import get_logger
from navigation.mission_state_machine import MissionState, MissionStateMachine

log = get_logger("main")


def main() -> int:
    log.info("농장형 자율주행 로봇 임무를 시작합니다.")
    fsm = MissionStateMachine()
    fsm.run_forever()

    # 종료 코드로 결과를 알린다 (systemd/스크립트에서 활용 가능)
    if fsm.state == MissionState.MISSION_COMPLETE:
        return 0
    if fsm.state == MissionState.SAFE_HALT:
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
