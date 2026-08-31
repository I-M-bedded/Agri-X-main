#!/usr/bin/env bash
# Gazebo Harmonic GUI 를 WSLg 로 Windows 화면에 띄운다.
#
#   wsl -d Ubuntu -- bash /mnt/c/Users/JUN/Documents/Agri-X-main/sim_gazebo/run_gui.sh
#
# WSL 안에서 실행해야 한다(WSLg 환경변수와 소켓이 거기 있다).
set -e

SIM_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLD="${1:-/sim/worlds/field.sdf}"

# WSLg 소켓/환경변수가 있어야 창이 뜬다
if [ ! -d /tmp/.X11-unix ]; then
  echo "오류: /tmp/.X11-unix 가 없습니다. WSL 안에서 실행하세요." >&2
  exit 1
fi

echo "월드: $WORLD"
echo "리소스: $SIM_DIR/models"

docker run --rm -it \
  -e DISPLAY="${DISPLAY:-:0}" \
  -e WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-}" \
  -e XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/mnt/wslg/runtime-dir}" \
  -e PULSE_SERVER="${PULSE_SERVER:-}" \
  -e GZ_SIM_RESOURCE_PATH=/sim/models \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /mnt/wslg:/mnt/wslg \
  -v "$SIM_DIR":/sim \
  --name agrix-gz \
  agrix-gazebo:harmonic \
  gz sim -v4 "$WORLD"
