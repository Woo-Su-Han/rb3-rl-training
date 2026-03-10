#!/bin/bash
# ROS2 + 워크스페이스 환경을 자동으로 source 하고 명령 실행
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash 2>/dev/null || true
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

exec "$@"
