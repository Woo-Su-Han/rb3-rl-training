"""
single_robot_env_3dof.py — rb3_730es_u 3-DOF 위치 도달 강화학습 환경

================================================================================
[6-DOF 환경(single_robot_env.py)과의 차이]
================================================================================
  항목              | 6-DOF (single_robot_env)       | 3-DOF (이 파일)
  ─────────────────────────────────────────────────────────────────────
  제어 관절         | q1~q6 (암 3 + 손목 3)          | q1~q3 (암만)
  손목(q4~q6)       | 학습 대상                       | 항상 0 고정 (홈 자세)
  obs 차원          | 28 (q6+e_pos3+e_ori3+topk16)  | 22 (q3+e_pos3+topk16)
  action 차원       | 6                               | 3
  태스크            | 위치 + 자세 (P/O/FT 모드)       | 위치만
  FK 백엔드         | robot_fk_rb3.py                 | robot_fk_rb3_3dof.py
  ROS 전송          | positions (6,)                  | [q1,q2,q3, 0,0,0] (6,)

================================================================================
[ros=None 모드]
================================================================================
  ROS2/Gazebo 없이 순수 FK 시뮬레이션으로 동작.
  Code_fk_3 dof/Environment_2.py 와 동일한 방식으로 학습 가능.
  → 학습 후 ros=ROS2Interface() 를 주입하면 실제 Gazebo에서 실행.

================================================================================
[Gazebo 연동 방법]
================================================================================
  from .ros2_interface import ROS2Interface
  from .single_robot_env_3dof import RobotReach3DOFEnv

  ros = ROS2Interface()
  env = RobotReach3DOFEnv(ros=ros, reward_mode="P", seed=0)
  obs, info = env.reset()
  # → Gazebo에서 홈 복귀 후 학습 시작
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .robot_fk_rb3_3dof import (
    fk_3dof,
    discretize_from_fk_output,
    pairwise_distance_matrix,
    nearest_nonlocal_distance,
    topk_dmin_with_indices,
    JOINT_LIMITS_3DOF,
)
from .ros2_interface import ROS2Interface


class RobotReach3DOFEnv(gym.Env):
    """
    rb3_730es_u 3-DOF 위치 도달 환경.

    Observation 22D:
        q(3) + e_pos(3) + topk_d(8) + topk_idx_norm(8)

    Action 3D:
        Δq = action * dq_max  (q1~q3만 제어, q4~q6=0 고정)

    ros=None : Gazebo 없이 FK 시뮬레이션만으로 동작 (학습 전용)
    ros=ROS2Interface() : Gazebo 실시간 연동
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    # rb3 작업공간 (3-DOF, 손목 고정 기준)
    # 실제 workspace는 q1~q3 + wrist=0 기준으로 결정됨
    _WS_Z = (0.05, 0.75)   # TCP Z 범위 [m]
    _WS_R = (0.10, 0.55)   # TCP XY 수평거리 범위 [m]

    def __init__(
        self,
        ros: ROS2Interface | None = None,
        *,
        # 스텝 제어
        step_sleep: float = 0.12,       # Gazebo 연동 시 명령 간격 [s]
        dq_max: float = np.deg2rad(5.0),
        # 셀프 충돌
        neighbor_exclusion: int = 20,
        thickness_margin: float = 0.04,
        safe_margin: float = 0.03,
        # 성공 허용 오차
        tol_pos: float = 0.05,
        # 에피소드
        max_steps: int = 200,
        topk: int = 8,
        # 보상 가중치
        progress_weight: float = 7.0,
        success_bonus: float = 15.0,
        collision_penalty: float = 10.0,
        safety_weight: float = 0.5,
        action_smoothing_weight: float = 0.02,
        action_l2_weight: float = 0.001,
        time_penalty_weight: float = 0.005,
        # 자세 편향 (Z 낮음 / 퍼짐)
        z_floor: float = 0.12,
        z_low_min_weight: float = 0.4,
        z_low_mean_weight: float = 0.2,
        spread_target: float = 0.28,
        spread_weight: float = 0.35,
        # 위치 목표 커리큘럼 (near → far)
        curriculum_r_start: float = 0.08,
        curriculum_r_end: float = 0.50,
        curriculum_episodes: int = 1800,
        goal_sample_max_tries: int = 50,
        # 초기 자세 랜덤화
        randomize_start: bool = False,
        q0_sample_max_tries: int = 200,
        q0_safe_threshold: float | None = None,
        q0_move_duration: float = 2.5,
        # 도메인 랜덤화
        obs_noise_std: float = 0.0,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()

        self.ros        = ros
        self.render_mode = render_mode

        self.step_sleep = float(step_sleep)
        self.dq_max     = float(dq_max)

        self.neighbor_exclusion = int(neighbor_exclusion)
        self.thickness_margin   = float(thickness_margin)
        self.safe_margin        = float(safe_margin)

        self.tol_pos   = float(tol_pos)
        self.max_steps = int(max_steps)
        self.topk      = int(topk)

        self.progress_weight         = float(progress_weight)
        self.success_bonus           = float(success_bonus)
        self.collision_penalty       = float(collision_penalty)
        self.safety_weight           = float(safety_weight)
        self.action_smoothing_weight = float(action_smoothing_weight)
        self.action_l2_weight        = float(action_l2_weight)
        self.time_penalty_weight     = float(time_penalty_weight)

        self.z_floor          = float(z_floor)
        self.z_low_min_weight = float(z_low_min_weight)
        self.z_low_mean_weight= float(z_low_mean_weight)
        self.spread_target    = float(spread_target)
        self.spread_weight    = float(spread_weight)

        self.curriculum_r_start    = float(curriculum_r_start)
        self.curriculum_r_end      = float(curriculum_r_end)
        self.curriculum_episodes   = int(curriculum_episodes)
        self.goal_sample_max_tries = int(goal_sample_max_tries)

        self.randomize_start     = bool(randomize_start)
        self.q0_sample_max_tries = int(q0_sample_max_tries)
        self.q0_safe_threshold   = float(
            safe_margin if q0_safe_threshold is None else q0_safe_threshold
        )
        self.q0_move_duration = float(q0_move_duration)
        self.obs_noise_std    = float(obs_noise_std)

        # 관절 한계 (3-DOF)
        self.joint_limits = JOINT_LIMITS_3DOF.copy()

        # Gym spaces: obs 22D, action 3D
        obs_dim = 3 + 3 + self.topk + self.topk
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(3,), dtype=np.float32
        )

        # 카운터
        self.np_random     = None
        self._seed         = seed
        self.episode_count = 0
        self.steps         = 0

        # 에피소드 상태 (q는 3-DOF)
        self.q      = np.zeros(3, dtype=np.float32)
        self.p_goal = np.zeros(3, dtype=np.float32)

        self.a_prev   = np.zeros(3, dtype=np.float32)
        self._cache: dict[str, Any] = {}

    # ── 외부 API ──────────────────────────────────────────────────

    def set_ros(self, ros: ROS2Interface):
        self.ros = ros

    # ── 커리큘럼 ─────────────────────────────────────────────────

    def _curriculum_radius(self) -> float:
        t = min(self.episode_count / max(self.curriculum_episodes, 1), 1.0)
        return (1.0 - t) * self.curriculum_r_start + t * self.curriculum_r_end

    # ── FK 헬퍼 ──────────────────────────────────────────────────

    def _clip_q(self, q: np.ndarray) -> np.ndarray:
        return np.clip(
            np.asarray(q, dtype=np.float32),
            self.joint_limits[:, 0],
            self.joint_limits[:, 1],
        )

    def _compute_d_eff_only(self, q: np.ndarray) -> float:
        """q(3)에서 d_effective_min만 계산 (q0 안전 판정용)."""
        _, joints, _, poly = fk_3dof(q)
        disc = discretize_from_fk_output(poly, joints, n_points=100)
        dmin = nearest_nonlocal_distance(
            pairwise_distance_matrix(disc), self.neighbor_exclusion
        )
        return float(np.nanmin(dmin) - self.thickness_margin)

    def _compute_all(self) -> np.ndarray:
        """self.q(3) → FK → obs(22D) + _cache 저장."""
        T_tcp, joints, _, poly = fk_3dof(self.q)
        p_end = T_tcp[:3, 3].astype(np.float32)
        e_pos = (self.p_goal - p_end).astype(np.float32)

        disc  = discretize_from_fk_output(poly, joints, n_points=100).astype(np.float32)
        dmin  = nearest_nonlocal_distance(
            pairwise_distance_matrix(disc), self.neighbor_exclusion
        )
        d_eff = float(np.nanmin(dmin) - self.thickness_margin)
        topk_d, topk_idx, topk_idx_norm = topk_dmin_with_indices(dmin, k=self.topk)

        obs = np.concatenate([
            self.q, e_pos, topk_d, topk_idx_norm,
        ]).astype(np.float32)

        if self.obs_noise_std > 0.0:
            obs += self.np_random.normal(
                0, self.obs_noise_std, size=obs.shape
            ).astype(np.float32)

        self._cache = dict(
            p_end=p_end, e_pos=e_pos, disc=disc,
            dmin=dmin, d_eff=d_eff,
            topk_d=topk_d, topk_idx=topk_idx, topk_idx_norm=topk_idx_norm,
        )
        return obs

    # ── 목표 / q0 샘플 ────────────────────────────────────────────

    def _sample_goal(self, p_end: np.ndarray) -> np.ndarray:
        """커리큘럼 반경 + rb3 3-DOF 작업공간 조건 내 목표 위치."""
        z_lo, z_hi = self._WS_Z
        r_lo, r_hi = self._WS_R
        r_max = self._curriculum_radius()

        for _ in range(self.goal_sample_max_tries):
            v = self.np_random.normal(size=3)
            n = np.linalg.norm(v)
            if n < 1e-9:
                continue
            g  = (p_end + (v / n) * self.np_random.uniform(0.0, r_max)).astype(np.float32)
            hd = float(np.linalg.norm(g[:2]))
            if r_lo <= hd <= r_hi and z_lo <= float(g[2]) <= z_hi:
                return g

        # fallback: 작업공간 내 직접 샘플
        theta = self.np_random.uniform(0.0, 2 * np.pi)
        hr    = self.np_random.uniform(r_lo, r_hi)
        return np.array([
            hr * np.cos(theta),
            hr * np.sin(theta),
            self.np_random.uniform(z_lo, z_hi),
        ], dtype=np.float32)

    def _sample_q0_safe(self) -> tuple[np.ndarray, float, bool]:
        """충돌 없는 초기 자세 q0 샘플링."""
        lo, hi   = self.joint_limits[:, 0], self.joint_limits[:, 1]
        best_q   = np.zeros(3, dtype=np.float32)
        best_d   = -np.inf

        for _ in range(self.q0_sample_max_tries):
            q_try = self._clip_q(
                self.np_random.uniform(lo, hi).astype(np.float32)
            )
            d = self._compute_d_eff_only(q_try)
            if d > best_d:
                best_d, best_q = d, q_try.copy()
            if d > self.q0_safe_threshold:
                return best_q, best_d, True

        return best_q, best_d, False

    # ── Gazebo I/O ────────────────────────────────────────────────

    def _send_q_to_gazebo(self, q3: np.ndarray, duration: float):
        """q(3) → [q1,q2,q3, 0,0,0] 6D로 변환하여 전송."""
        q6 = np.zeros(6, dtype=np.float32)
        q6[:3] = q3
        self.ros.send_joint_positions(q6, duration_sec=duration)

    def _execute_action(self, dq: np.ndarray):
        """dq(3) 적용 → ROS2 전송 또는 시뮬 직접 갱신."""
        new_q = self._clip_q(self.q + dq)
        if self.ros is not None:
            self._send_q_to_gazebo(new_q, duration=self.step_sleep)
            time.sleep(self.step_sleep)
            # 실제 관절값 읽기 (q4~q6 버림)
            self.q = self.ros.get_joint_pos()[:3].astype(np.float32)
        else:
            self.q = new_q

    # ── Gym API ──────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            super().reset(seed=seed)
            self.np_random = np.random.default_rng(seed)
        elif self.np_random is None:
            self.np_random = np.random.default_rng(self._seed)

        self.steps  = 0
        self.a_prev = np.zeros(3, dtype=np.float32)
        self.episode_count += 1

        # Gazebo 홈 복귀
        if self.ros is not None:
            self.ros.episode_reset()

        # 초기 자세 설정
        if self.randomize_start:
            q0, q0_d_eff, q0_ok = self._sample_q0_safe()
            if self.ros is not None:
                self._send_q_to_gazebo(q0, duration=self.q0_move_duration)
                time.sleep(self.q0_move_duration + 0.3)
                self.q = self.ros.get_joint_pos()[:3].astype(np.float32)
            else:
                self.q = q0
        else:
            if self.ros is not None:
                self.q = self.ros.get_joint_pos()[:3].astype(np.float32)
            else:
                self.q = np.zeros(3, dtype=np.float32)
            q0_ok    = True
            q0_d_eff = self._compute_d_eff_only(self.q)

        # 현재 TCP로 목표 샘플
        T_tcp_0, _, _, _ = fk_3dof(self.q)
        p_end_start = T_tcp_0[:3, 3].astype(np.float32)
        self.p_goal = self._sample_goal(p_end_start)
        self.goal_init_dist = float(np.linalg.norm(self.p_goal - p_end_start))

        obs = self._compute_all()

        info = {
            "p_goal"          : self.p_goal.copy(),
            "p_end"           : self._cache["p_end"].copy(),
            "e_pos_norm"      : float(np.linalg.norm(self._cache["e_pos"])),
            "d_effective_min" : self._cache["d_eff"],
            "goal_init_dist"  : self.goal_init_dist,
            "curriculum_r_max": self._curriculum_radius(),
            "episode_count"   : self.episode_count,
            "q0_ok"           : q0_ok,
            "q0_d_eff"        : q0_d_eff,
            "randomize_start" : self.randomize_start,
        }
        return obs, info

    def step(self, action: np.ndarray):
        self.steps += 1

        # 1. action → dq, 페널티
        a  = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        dq = a * self.dq_max
        r_l2     = -self.action_l2_weight * float(np.dot(dq, dq))
        r_smooth = -self.action_smoothing_weight * float(
            np.dot(a - self.a_prev, a - self.a_prev)
        )
        self.a_prev = a.copy()

        prev_e_norm = float(np.linalg.norm(self._cache.get("e_pos", np.zeros(3))))

        # 2. 관절 이동
        self._execute_action(dq)

        # 3. FK → obs
        obs        = self._compute_all()
        e_pos_norm = float(np.linalg.norm(self._cache["e_pos"]))
        d_eff      = float(self._cache["d_eff"])

        # 4. 종료 판정
        collision  = d_eff <= 0.0
        success    = e_pos_norm <= self.tol_pos
        truncated  = self.steps >= self.max_steps
        terminated = bool(collision or success)

        # 5. 보상
        # 위치 진척
        progress = prev_e_norm - e_pos_norm
        r_prog   = self.progress_weight * progress
        r_succ   = self.success_bonus if success else 0.0

        # 안전 shaping
        if collision:
            r_safe = -self.collision_penalty
        elif d_eff < self.safe_margin:
            r_safe = -self.safety_weight * (self.safe_margin - d_eff) / max(self.safe_margin, 1e-6)
        else:
            r_safe = 0.0

        # 자세 편향
        pts   = self._cache["disc"]
        z     = pts[:, 2]
        z_min = float(np.min(z))
        viol_min  = max(0.0, self.z_floor - z_min)
        viol_mean = float(np.mean(np.maximum(0.0, self.z_floor - z)))
        r_z   = -(self.z_low_min_weight * viol_min + self.z_low_mean_weight * viol_mean)

        v      = pts[:, :2]
        spread = float(np.mean(np.linalg.norm(v - np.mean(v, axis=0), axis=1)))
        r_spread = -self.spread_weight * max(0.0, spread - self.spread_target)

        r_time = -self.time_penalty_weight

        reward = float(np.clip(
            r_prog + r_succ + r_safe + r_l2 + r_smooth + r_z + r_spread + r_time,
            -8.0, 8.0,
        ))

        info = {
            "p_goal"          : self.p_goal.copy(),
            "p_end"           : self._cache["p_end"].copy(),
            "e_pos_norm"      : e_pos_norm,
            "d_effective_min" : d_eff,
            "collision"       : collision,
            "success"         : success,
            "progress"        : float(progress),
            "r_progress"      : float(r_prog),
            "r_success"       : float(r_succ),
            "r_safety"        : float(r_safe),
            "r_l2"            : float(r_l2),
            "r_smooth"        : float(r_smooth),
            "r_z"             : float(r_z),
            "r_spread"        : float(r_spread),
            "z_min"           : z_min,
            "spread"          : spread,
            "steps"           : self.steps,
            "curriculum_r_max": self._curriculum_radius(),
            "episode_count"   : self.episode_count,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        pass


# ── 동작 확인 (Gazebo 없이) ───────────────────────────────────────

if __name__ == "__main__":
    env = RobotReach3DOFEnv(
        ros=None,
        seed=0,
        randomize_start=True,
        dq_max=np.deg2rad(5.0),
        max_steps=200,
        curriculum_r_start=0.08,
        curriculum_r_end=0.50,
        curriculum_episodes=1800,
    )

    obs, info = env.reset()
    print(f"obs shape : {obs.shape}")
    print(f"e_pos     : {info['e_pos_norm']:.4f} m")
    print(f"d_eff     : {info['d_effective_min']:.4f} m")
    print(f"p_goal    : {info['p_goal']}")
    print(f"p_end     : {info['p_end']}")

    ep_r = 0.0
    for t in range(50):
        obs, r, term, trunc, info = env.step(
            np.random.uniform(-0.3, 0.3, size=3).astype(np.float32)
        )
        ep_r += r
        if t < 3 or term or trunc:
            print(f"[{t:3d}] r={r:+.3f}  e_pos={info['e_pos_norm']:.4f}"
                  f"  d_eff={info['d_effective_min']:.4f}"
                  f"  success={info['success']}  collision={info['collision']}")
        if term or trunc:
            print(f"  → 종료")
            break
    print(f"\n누적 보상: {ep_r:.3f}")
