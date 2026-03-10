"""
ros2_interface.py — Gazebo ↔ RL 통신 인터페이스 (순수 ROS2 I/O)

역할
----
  - /joint_states 구독 → 현재 관절 각도/속도 읽기
  - /rb3_1_joint_trajectory_controller/joint_trajectory 발행 → 관절 명령 전송
  - 홈 복귀(reset_to_home), 에피소드 리셋(episode_reset)
  - Gazebo 서비스: pause/unpause/reset_simulation, set_entity_state

FK 계산 / 충돌 감지 / EE 자세 계산은 모두 robot_fk_rb3.py 에서 처리.
"""

from __future__ import annotations

import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_srvs.srv import Empty
from builtin_interfaces.msg import Duration
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose, Twist


JOINT_NAMES = [
    "rb3_1_base", "rb3_1_shoulder", "rb3_1_elbow",
    "rb3_1_wrist1", "rb3_1_wrist2", "rb3_1_wrist3",
]

JOINT_LIMITS = np.array(
    [[-np.pi, np.pi]] * 6, dtype=np.float32
)


class ROS2Interface(Node):
    """
    Gazebo ↔ RL 통신 노드.

    - 관절 상태 읽기 / 관절 명령 전송 만 담당.
    - FK, 충돌 감지, 자세 오차 등은 robot_fk_rb3.py에서 처리.
    """

    def __init__(self):
        super().__init__("rbpodo_rl_interface")

        self._joint_pos = np.zeros(6, dtype=np.float32)
        self._joint_vel = np.zeros(6, dtype=np.float32)
        self._lock = threading.Lock()
        self._joint_received = False

        # ── 구독 / 발행 ──────────────────────────────────────────
        self.create_subscription(
            JointState, "/joint_states", self._joint_cb, 10
        )
        self._traj_pub = self.create_publisher(
            JointTrajectory,
            "/rb3_1_joint_trajectory_controller/joint_trajectory",
            10,
        )

        # ── Gazebo 서비스 ─────────────────────────────────────────
        self._reset_cli   = self.create_client(Empty, "/reset_simulation")
        self._pause_cli   = self.create_client(Empty, "/pause_physics")
        self._unpause_cli = self.create_client(Empty, "/unpause_physics")
        self._set_ent_cli = self.create_client(SetEntityState, "/set_entity_state")

        self.get_logger().info("ROS2Interface 초기화 완료")

    # ── 콜백 ─────────────────────────────────────────────────────

    def _joint_cb(self, msg: JointState):
        with self._lock:
            for i, name in enumerate(JOINT_NAMES):
                if name in msg.name:
                    idx = msg.name.index(name)
                    self._joint_pos[i] = msg.position[idx]
                    if msg.velocity:
                        self._joint_vel[i] = msg.velocity[idx]
            self._joint_received = True

    # ── 읽기 ─────────────────────────────────────────────────────

    def get_joint_pos(self) -> np.ndarray:
        """현재 관절 각도 (6,) 복사본."""
        with self._lock:
            return self._joint_pos.copy()

    def get_joint_vel(self) -> np.ndarray:
        """현재 관절 속도 (6,) 복사본."""
        with self._lock:
            return self._joint_vel.copy()

    def get_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        """(joint_pos, joint_vel) 동시 스냅샷."""
        with self._lock:
            return self._joint_pos.copy(), self._joint_vel.copy()

    # ── 대기 ─────────────────────────────────────────────────────

    def wait_for_joint_states(self, timeout: float = 5.0) -> bool:
        start = time.time()
        while not self._joint_received:
            if time.time() - start > timeout:
                self.get_logger().error("joint_states 수신 타임아웃")
                return False
            time.sleep(0.05)
        return True

    # ── 쓰기 ─────────────────────────────────────────────────────

    def send_joint_positions(
        self,
        positions: np.ndarray,
        duration_sec: float = 0.5,
    ):
        """관절 위치 명령 전송 (joint_limits 클리핑 포함)."""
        positions = np.clip(
            np.asarray(positions, dtype=np.float32),
            JOINT_LIMITS[:, 0],
            JOINT_LIMITS[:, 1],
        )
        msg = JointTrajectory()
        msg.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = positions.tolist()
        pt.time_from_start = Duration(
            sec=int(duration_sec),
            nanosec=int((duration_sec % 1) * 1_000_000_000),
        )
        msg.points = [pt]
        self._traj_pub.publish(msg)

    def send_joint_delta(self, delta: np.ndarray, duration_sec: float = 0.1):
        """현재 위치 기준으로 delta만큼 이동."""
        with self._lock:
            current = self._joint_pos.copy()
        self.send_joint_positions(current + delta, duration_sec)

    def reset_to_home(
        self,
        duration_sec: float = 2.0,
        tolerance: float = 0.05,
        timeout: float = 8.0,
    ) -> bool:
        """홈 자세(q=0)로 이동 후 도달 확인."""
        self.send_joint_positions(np.zeros(6), duration_sec)
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                current = self._joint_pos.copy()
            if np.all(np.abs(current) < tolerance):
                return True
            time.sleep(0.05)
        self.get_logger().warn(
            f"reset_to_home 타임아웃: max_err={np.max(np.abs(current)):.3f} rad"
        )
        return False

    # ── 에피소드 리셋 ─────────────────────────────────────────────

    def episode_reset(
        self,
        object_poses: dict | None = None,
        home_duration: float = 1.5,
        home_tolerance: float = 0.05,
    ) -> bool:
        """
        에피소드 리셋:
          1. (선택) 물체 위치 리셋 (pause → set_entity_state → unpause)
          2. 홈 복귀
        """
        if object_poses:
            self.pause_physics()
            for name, (x, y, z) in object_poses.items():
                self.set_object_pose(name, x, y, z)
            self.unpause_physics()
        return self.reset_to_home(home_duration, home_tolerance)

    # ── Gazebo 서비스 ─────────────────────────────────────────────

    def _call_empty(self, client, timeout: float = 3.0) -> bool:
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().warn(f"서비스 대기 타임아웃: {client.srv_name}")
            return False
        future = client.call_async(Empty.Request())
        start = time.time()
        while not future.done():
            if time.time() - start > timeout:
                return False
            time.sleep(0.02)
        return True

    def pause_physics(self)   -> bool: return self._call_empty(self._pause_cli)
    def unpause_physics(self) -> bool: return self._call_empty(self._unpause_cli)

    def set_object_pose(
        self,
        name: str,
        x: float, y: float, z: float,
        timeout: float = 3.0,
    ) -> bool:
        if not self._set_ent_cli.wait_for_service(timeout_sec=timeout):
            return False
        req = SetEntityState.Request()
        req.state = EntityState()
        req.state.name = name
        req.state.pose = Pose()
        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        req.state.pose.position.z = float(z)
        req.state.pose.orientation.w = 1.0
        req.state.twist = Twist()
        future = self._set_ent_cli.call_async(req)
        start = time.time()
        while not future.done():
            if time.time() - start > timeout:
                return False
            time.sleep(0.02)
        return bool(future.result().success)

    def reset_simulation(self, timeout: float = 10.0) -> bool:
        """
        전체 Gazebo 시뮬레이션 리셋 (느림 ~1분).
        에피소드 리셋은 episode_reset() 을 사용할 것.
        """
        self.get_logger().warn("reset_simulation 호출 — 완료까지 수십 초 소요")
        if not self._reset_cli.wait_for_service(timeout_sec=timeout):
            self.get_logger().error("/reset_simulation 서비스 없음")
            return False
        future = self._reset_cli.call_async(Empty.Request())
        start = time.time()
        while not future.done():
            if time.time() - start > timeout:
                return False
            time.sleep(0.05)
        time.sleep(1.5)
        with self._lock:
            self._joint_received = False
        return self.wait_for_joint_states(timeout=max(timeout, 10.0))
