# Gazebo Harmonic (gz-sim) 최소 이미지
#   osrf 공식 apt 저장소 사용. WSLg 로 GUI 를 띄우기 위한 라이브러리 포함.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg lsb-release ca-certificates \
        libgl1 libglx-mesa0 libegl1 mesa-utils \
        x11-apps libxkbcommon-x11-0 \
    && curl -sSL https://packages.osrfoundation.org/gazebo.gpg \
        -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
http://packages.osrfoundation.org/gazebo/ubuntu-stable noble main" \
        > /etc/apt/sources.list.d/gazebo-stable.list \
    && apt-get update && apt-get install -y --no-install-recommends gz-harmonic \
    && rm -rf /var/lib/apt/lists/*
ENV GZ_SIM_RESOURCE_PATH=/sim/models
WORKDIR /sim
