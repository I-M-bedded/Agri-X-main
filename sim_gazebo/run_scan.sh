#!/bin/bash
# 헤드랜드 여러 지점에서 제자리 회전하며 실제 ArUco 검출률 측정
scan_at () {
  gz service -s /world/field/set_pose --reqtype gz.msgs.Pose \
    --reptype gz.msgs.Boolean --timeout 3000 \
    --req "name: \"agv\", position: {x: $1, y: $2, z: 0.05}, orientation: {z: 0.7071, w: 0.7071}" >/dev/null 2>&1
  sleep 2
  echo "--- 시작 위치 x=$1 y=$2 ---"
  python3 /sim/scripts/drive_harness.py --scenario marker_scan --seconds 13 2>/dev/null | head -3
}
echo "마커: 이랑 중심 x=-0.5, 0.5, 1.5, 2.5, 3.5 (ID 0,1,2,3,4) / y=0"
scan_at -0.9 -1.0
scan_at  0.0 -1.0
scan_at  0.5 -1.5
