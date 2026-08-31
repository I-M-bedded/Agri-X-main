# -*- coding: utf-8 -*-
"""
sim_gazebo/scripts/run_sweep.py
--------------------------------
policy_harness.py 를 조건별로 반복 실행하는 스윕 러너. **컨테이너 안**에서 돈다.

    python3 /sim/scripts/run_sweep.py --tilt 30 --stage entry \
        --out /sim/out/sweep_t30.jsonl

한 시행마다
    1. cmd_vel 0 발행 -> 로봇 정지
    2. set_pose 로 시작 자세 재설정 (x, y, yaw 모두 변경)
    3. policy_harness.py 실행, "RESULT {...}" 한 줄을 회수
결과는 jsonl 로 누적한다(중간에 끊겨도 거기까지는 남는다).
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

HARNESS = "/sim/scripts/policy_harness.py"

# 시작 위치: 헤드랜드 어딘가에서 밭을 보고 출발한다는 전제.
#   0번 팻말(이랑0, x=-0.5) 근방이지만 위치는 매번 다르다.
STARTS = [
    (-1.20, -1.00,  90.0),      # 팻말 0 보다 왼쪽, 정면
    (-0.50, -1.20,  90.0),      # 팻말 0 바로 앞
    (0.00, -1.00,  90.0),       # 목표 고랑(1) 정면
    (-0.90, -1.60,  90.0),      # 멀리서
    (-0.60, -1.00,  70.0),      # 헤딩 -20도 오차
    (-0.60, -1.00, 110.0),      # 헤딩 +20도 오차
]


def yaw_quat(deg):
    h = math.radians(deg) / 2.0
    return math.sin(h), math.cos(h)


def reset(x, y, yaw_deg):
    z, w = yaw_quat(yaw_deg)
    req = (f'name: "agv", position: {{x: {x}, y: {y}, z: 0.05}}, '
           f'orientation: {{z: {z:.6f}, w: {w:.6f}}}')
    subprocess.run(["gz", "service", "-s", "/world/field/set_pose",
                    "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
                    "--timeout", "3000", "--req", req],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2.0)


def stop_robot():
    subprocess.run(["gz", "topic", "-t", "/cmd_vel", "-m", "gz.msgs.Twist",
                    "-p", "linear: {x: 0}, angular: {z: 0}"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tilt", type=float, required=True, help="현재 월드의 팻말 각")
    ap.add_argument("--marker-cm", type=float, default=20.0)
    ap.add_argument("--stage", default="entry", choices=["entry", "full"])
    ap.add_argument("--policies",
                    default="rotate:survey,sweep:survey,creep:survey,"
                            "back:survey,rotate:tof,sweep:tof,creep:tof,back:tof")
    ap.add_argument("--starts", default="all",
                    help="all 또는 인덱스 목록(예: 0,2,4)")
    ap.add_argument("--vision", default="measured",
                    choices=["measured", "blind"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pols = [p.split(":") for p in args.policies.split(",")]
    idx = (range(len(STARTS)) if args.starts == "all"
           else [int(i) for i in args.starts.split(",")])
    starts = [STARTS[i] for i in idx]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    total = len(pols) * len(starts)
    n = 0
    with open(args.out, "a", encoding="utf-8") as fp:
        for search, entry in pols:
            for (sx, sy, syaw) in starts:
                n += 1
                stop_robot()
                reset(sx, sy, syaw)
                cmd = [sys.executable, HARNESS,
                       "--search", search, "--entry", entry,
                       "--stage", args.stage,
                       "--marker-cm", str(args.marker_cm),
                       "--tilt", str(args.tilt),
                       "--start-x", str(sx), "--start-y", str(sy),
                       "--start-yaw", str(syaw), "--seed", str(args.seed),
                       "--vision", args.vision]
                t0 = time.time()
                try:
                    out = subprocess.run(cmd, capture_output=True, text=True,
                                         timeout=260).stdout
                except subprocess.TimeoutExpired:
                    out = ""
                line = next((l for l in out.splitlines()
                             if l.startswith("RESULT ")), None)
                if line is None:
                    rec = {"search": search, "entry": entry, "tilt": args.tilt,
                           "start": [sx, sy, syaw], "fail": "harness_crash"}
                else:
                    rec = json.loads(line[7:])
                rec["wall_sec"] = round(time.time() - t0, 1)
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fp.flush()
                print(f"[{n}/{total}] tilt{args.tilt:.0f} {search}/{entry} "
                      f"start({sx},{sy},{syaw:.0f}) -> "
                      f"진입 {'O' if rec.get('entry_ok') else 'X'} "
                      f"고랑{rec.get('entered_furrow')} "
                      f"이탈{rec.get('max_dev_cm', 0):.1f}cm "
                      f"검출{rec.get('detect_frames', 0)}f "
                      f"{rec.get('fail', '')}", flush=True)
    stop_robot()


if __name__ == "__main__":
    main()
