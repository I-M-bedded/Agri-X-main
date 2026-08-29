# -*- coding: utf-8 -*-
"""Legacy entrypoint for the original large mission state machine."""

import sys

from logutil import get_logger
from navigation.mission_state_machine import MissionState, MissionStateMachine

log = get_logger("legacy-main")


def main() -> int:
    log.info("legacy mission state machine start")
    fsm = MissionStateMachine()
    fsm.run_forever()
    if fsm.state == MissionState.MISSION_COMPLETE:
        return 0
    if fsm.state == MissionState.SAFE_HALT:
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
