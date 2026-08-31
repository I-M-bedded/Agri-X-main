#!/bin/bash
# 지속 cmd_vel 발행 후 주행 결과 확인
for i in $(seq 1 40); do
  gz topic -t /cmd_vel -m gz.msgs.Twist -p "linear: {x: 0.3}" 2>/dev/null
  sleep 0.1
done
gz topic -t /cmd_vel -m gz.msgs.Twist -p "linear: {x: 0.0}" 2>/dev/null
echo "--- 주행 후 odom ---"
timeout 8 gz topic -e -t /odom -n 1 2>/dev/null | grep -A3 "position" | head -4
