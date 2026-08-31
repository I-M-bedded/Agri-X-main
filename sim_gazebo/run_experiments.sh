#!/bin/bash
# 주행 시퀀스 실험 묶음. 컨테이너 안에서 실행.
# 매 시나리오 전에 로봇을 고랑1 시작점으로 리셋한다.
reset_robot () {
  gz service -s /world/field/set_pose --reqtype gz.msgs.Pose \
    --reptype gz.msgs.Boolean --timeout 3000 \
    --req "name: \"agv\", position: {x: 0.0, y: 0.4, z: 0.05}, orientation: {z: 0.7071, w: 0.7071}" >/dev/null 2>&1
  sleep 2
}
run () {  # $1=vision $2=veto $3=label
  reset_robot
  python3 /sim/scripts/drive_harness.py --scenario furrow_drive \
    --vision "$1" --veto "$2" --seconds 16 --seed "${4:-0}" 2>/dev/null
}
echo "=================================================================="
echo " 고랑 주행 실험 (실제 ToF + 실제 ArUco + 실측근사 비전오차)"
echo " 고랑 폭 0.4m / 차체 0.2m -> 좌우 여유 각 10cm"
echo "=================================================================="
run measured 0.15
run blind    0.15
run measured 0.0
