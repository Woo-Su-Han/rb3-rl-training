# =========================================================================
# rbpodo_rl — ROS2 Humble + Gazebo Classic + RL 학습 환경
#
# 베이스: osrf/ros:humble-desktop-full
#   - ROS2 Humble
#   - Gazebo Classic (gazebo11)
#   - Python 3.10
#
# 빌드:
#   docker build -t rbpodo_rl:latest .
#
# 사용 (Gazebo GUI 포함):
#   docker compose up gazebo   # Gazebo 실행
#   docker compose up train    # 학습 실행
#
# 사용 (Gazebo 없이 FK 시뮬레이션만):
#   docker compose up train-noros
# =========================================================================

FROM osrf/ros:humble-desktop-full

# ── 시스템 패키지 ──────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Gazebo ROS 패키지
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros2-control \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-joint-state-broadcaster \
    ros-humble-joint-trajectory-controller \
    ros-humble-robot-state-publisher \
    ros-humble-xacro \
    # 빌드 도구
    python3-colcon-common-extensions \
    python3-pip \
    python3-rosdep \
    # 유틸
    git \
    wget \
    curl \
    vim \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# ── Python 패키지 ──────────────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# ── 워크스페이스 복사 & 빌드 ───────────────────────────────────────────────
WORKDIR /ros2_ws

# src 폴더만 먼저 복사 (레이어 캐시 활용)
COPY src/ src/

# rosdep 초기화 및 의존성 설치
RUN rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y

# colcon 빌드
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install \
        --packages-select rbpodo_description rbpodo_bringup rbpodo_msgs \
                          rbpodo_gazebo rbpodo_rl && \
    echo 'Build complete'"

# ── 환경 설정 ──────────────────────────────────────────────────────────────
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc && \
    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /root/.bashrc

# 엔트리포인트: ROS2 환경 자동 source
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["bash"]
