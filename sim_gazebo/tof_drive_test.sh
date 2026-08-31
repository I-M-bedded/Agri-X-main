#!/bin/bash
# 고랑 주행 중 ToF 로 좌우 이랑 벽을 읽는지 확인.
# [수정] 예전에는 odom 의 x(로봇 로컬 전진거리)를 표시해 월드 좌표와 혼동했다.
#        이제 월드 pose(x, y)를 직접 읽는다. 로봇은 +y 로 전진한다.
world_pose () {
  timeout 6 gz topic -e -t /world/field/dynamic_pose/info -n 1 2>/dev/null \
    | grep -A6 'name: "agv"' | grep -E '^\s+(x|y):' | head -2 \
    | awk '{printf "%s ", $2}'
}
read_row () {
  L=$(timeout 6 gz topic -e -t /tof_left  -n 3 2>/dev/null | grep ranges | tail -1 | awk '{print $2}')
  R=$(timeout 6 gz topic -e -t /tof_right -n 3 2>/dev/null | grep ranges | tail -1 | awk '{print $2}')
  P=$(world_pose)
  printf "  월드(x y)= %-22.22s | 좌 %-9.9s | 우 %-9.9s\n" "$P" "$L" "$R"
}
echo "이랑 x=-0.5,0.5,... (폭0.6) / 고랑1 중심 x=0 -> 좌우 벽까지 각 0.2m 기대"
echo "이랑 구간은 y=0~6"
sleep 2   # 스폰 직후 안정화(바운스) 대기
read_row
for step in 1 2 3 4 5; do
  for i in $(seq 1 10); do
    gz topic -t /cmd_vel -m gz.msgs.Twist -p "linear: {x: 0.25}" 2>/dev/null
    sleep 0.1
  done
  gz topic -t /cmd_vel -m gz.msgs.Twist -p "linear: {x: 0.0}" 2>/dev/null
  sleep 0.6
  read_row
done
