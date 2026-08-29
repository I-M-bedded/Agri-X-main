# -*- coding: utf-8 -*-
"""Agri-X runtime entrypoint.

This feature branch defaults to the lightweight AI/ArUco/ToF FSM:

    python3 main.py

Use the previous large mission state machine only when explicitly requested:

    python3 main.py --legacy
"""

import sys


def main() -> int:
    if "--legacy" in sys.argv:
        sys.argv.remove("--legacy")
        from legacy_main import main as legacy_main
        return legacy_main()

    from pipeline_main import main as pipeline_main
    return pipeline_main()


if __name__ == "__main__":
    sys.exit(main())
