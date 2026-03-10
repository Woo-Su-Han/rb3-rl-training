"""
ros2_interface.py 테스트 스크립트.

실행 전: ros2 launch rbpodo_gazebo rb3_gazebo.launch.py

실행:
    cd ~/rbpodo_ws/src/rbpodo_rl/rbpodo_rl
    python3 test_interface.py
"""
import sys
import os
import time
import threading
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rclpy
from rbpodo_rl.envs.ros2_interface import ROS2Interface


def main():
    rclpy.init()
    ros = ROS2Interface()

    spin_thread = threading.Thread(target=rclpy.spin, args=(ros,), daemon=True)
    spin_thread.start()

    print("\n" + "="*50)
    print("TEST 1: joint_states 수신 확인")
    print("="*50)
    ok = ros.wait_for_joint_states(timeout=5.0)
    if not ok:
        print("FAIL: joint_states 수신 못함. Gazebo 켜져 있는지 확인하세요.")
        return
    pos = ros.get_joint_pos()
    print(f"OK: 현재 조인트 각도 = {np.round(pos, 4)}")

    print("\n" + "="*50)
    print("TEST 2: 조인트 명령 전송 (rb3_1_base → 0.3 rad)")
    print("="*50)
    ros.send_joint_positions(np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0]), duration_sec=2.0)
    time.sleep(2.5)
    pos = ros.get_joint_pos()
    print(f"명령 후 조인트 각도 = {np.round(pos, 4)}")
    if abs(pos[0] - 0.3) < 0.1:
        print("OK: base 관절 이동 성공")
    else:
        print(f"WARN: base 예상 0.3, 실제 {pos[0]:.3f} (차이 {abs(pos[0]-0.3):.3f})")

    print("\n" + "="*50)
    print("TEST 3: 홈 복귀 (tolerance 확인)")
    print("="*50)
    ok = ros.reset_to_home(duration_sec=2.0, tolerance=0.05)
    pos = ros.get_joint_pos()
    print(f"홈 복귀 후 조인트 = {np.round(pos, 4)}")
    print(f"결과: {'OK' if ok else 'WARN: 타임아웃, 그래도 계속 진행'}")

    print("\n" + "="*50)
    print("TEST 4: RL 에피소드 흐름 시뮬레이션")
    print("  (목표 설정 → delta 이동 10스텝 → 홈 복귀)")
    print("="*50)

    # 목표 설정
    target = np.array([0.4, -0.3, 0.2, 0.0, 0.0, 0.0])
    print(f"목표 조인트     = {np.round(target, 3)}")

    # 10스텝 delta 이동 (실제 RL step 흉내)
    for step in range(10):
        current = ros.get_joint_pos()
        direction = np.sign(target - current)
        delta = direction * 0.05  # MAX_DELTA
        ros.send_joint_delta(delta, duration_sec=0.1)
        time.sleep(0.15)

    pos = ros.get_joint_pos()
    dist = np.linalg.norm(pos - target)
    print(f"10스텝 후 조인트 = {np.round(pos, 3)}")
    print(f"목표까지 거리    = {dist:.3f} rad")

    # 에피소드 종료 → 홈 복귀
    ok = ros.reset_to_home(duration_sec=2.0, tolerance=0.05)
    print(f"홈 복귀: {'OK' if ok else 'WARN: 타임아웃'}")

    # TEST 5: reset_simulation()은 ~1분 소요 (컨트롤러 재초기화 때문)
    # RL 학습 중 에피소드 리셋은 reset_to_home()으로 충분.
    # 물체 추가 시에는 set_object_pose()로 물체 위치만 이동.
    # reset_simulation()은 물리 폭발 등 비상 상황에만 사용.

    print("\n" + "="*50)
    print("전체 테스트 완료")
    print("="*50)

    # shutdown 먼저 → spin() 루프 종료 → 스레드 join
    rclpy.shutdown()
    spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
