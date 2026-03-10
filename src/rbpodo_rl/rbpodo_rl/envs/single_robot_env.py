"""
single_robot_env.py — rb3_730es_u Gazebo + ROS2 강화학습 환경

Observation  28D : q(6) + e_pos(3) + e_ori(3) + topk_d(8) + topk_idx_norm(8)
Action        6D : Δq, [-1,1]. dq[:3]=a[:3]*dq_max_pos, dq[3:]=a[3:]*dq_max_ori
Reward            : progress(pos/ori) + success_bonus - collision - posture - action - time
Curriculum        : 위치반경 / 자세각도 / 손목(wrist) / 성공률 자동전환
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .robot_fk_rb3 import (
    fk_full,
    discretize_from_fk_output,
    pairwise_distance_matrix,
    nearest_nonlocal_distance,
    topk_dmin_with_indices,
    orientation_error_rotvec,
)
from .ros2_interface import ROS2Interface, JOINT_LIMITS

_TWO_PI = 2.0 * np.pi


def _axis_angle_to_R(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues 공식: (unit_axis, angle[rad]) → 3×3 회전행렬."""
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3)
    a /= n
    K = np.array([[0., -a[2], a[1]], [a[2], 0., -a[0]], [-a[1], a[0], 0.]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


class RobotReachEnv(gym.Env):
    """
    rb3_730es_u 단일 로봇 Gazebo RL 환경.
    ros=None 이면 ROS2 없이 FK 시뮬레이션만으로 동작.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    # rb3 작업공간 상수 (goal sampling)
    _WS_Z   = (0.10, 0.80)
    _WS_R   = (0.10, 0.55)

    def __init__(
        self,
        ros: ROS2Interface | None = None,
        *,
        # 스텝 제어
        step_sleep: float = 0.12,
        dq_max_pos: float = np.deg2rad(5.0),
        dq_max_ori: float = np.deg2rad(5.0),
        ddq_max_pos: float = np.deg2rad(8.0),
        ddq_max_ori: float = np.deg2rad(8.0),
        # 셀프 충돌
        neighbor_exclusion: int = 20,
        thickness_margin: float = 0.04,
        safe_margin: float = 0.03,
        # 성공 허용 오차
        tol_pos: float = 0.05,
        tol_ori: float = np.deg2rad(10.0),
        tol_gate: float = 0.08,
        # 에피소드
        max_steps: int = 200,
        topk: int = 8,
        # 위치 보상
        progress_weight_pos: float = 7.0,
        success_bonus_pos: float = 15.0,
        # 자세 보상
        progress_weight_ori: float = 2.5,
        success_bonus_ori: float = 12.0,
        pos_hold_weight: float = 2.0,
        pos_success_bonus_in_O: float = 6.0,
        # 안전
        collision_penalty: float = 10.0,
        safety_weight: float = 0.5,
        # 액션 정규화
        action_smoothing_weight: float = 0.02,
        action_l2_weight: float = 0.001,
        # 시간 패널티
        time_penalty_weight: float = 0.005,
        # 자세 편향
        z_floor: float = 0.12,
        z_low_min_weight: float = 0.4,
        z_low_mean_weight: float = 0.2,
        spread_target: float = 0.22,
        spread_weight: float = 0.8,
        perp_weight: float = 0.08,
        back_weight: float = 0.04,
        goal_xy_min_norm: float = 0.05,
        # 위치 커리큘럼
        curriculum_r_start: float = 0.08,
        curriculum_r_end: float = 0.50,
        curriculum_episodes: int = 3000,
        goal_sample_max_tries: int = 50,
        # 자세 커리큘럼
        ori_angle_start: float = np.deg2rad(0.0),
        ori_angle_end: float = np.deg2rad(60.0),
        ori_curriculum_episodes: int = 1800,
        # Stage1 손목 커리큘럼 (total_steps 기준)
        stage1_wrist_fixed_steps: int = 200_000,
        stage1_wrist_ramp_steps: int = 200_000,
        stage1_wrist_amp_start: float = 0.0,
        stage1_wrist_amp_end: float = np.deg2rad(60.0),
        # 성공률 커리큘럼
        sr_window: int = 100,
        sr_pos_advance_thresh: float = 0.60,
        sr_ori_advance_thresh: float = 0.50,
        auto_advance_mode: bool = False,
        # 랜덤 초기 자세
        randomize_start: bool = False,
        q0_sample_max_tries: int = 200,
        q0_safe_threshold: float | None = None,
        q0_move_duration: float = 2.5,
        # 도메인 랜덤화
        obs_noise_std: float = 0.0,
        # 보상 모드
        reward_mode: str = "P",
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()

        if reward_mode not in ("P", "O", "FT"):
            raise ValueError('reward_mode must be "P", "O", or "FT"')

        self.ros         = ros
        self.reward_mode = reward_mode
        self.render_mode = render_mode

        # 모션
        self.step_sleep  = float(step_sleep)
        self.dq_max_pos  = float(dq_max_pos);  self.dq_max_ori  = float(dq_max_ori)
        self.ddq_max_pos = float(ddq_max_pos); self.ddq_max_ori = float(ddq_max_ori)

        # 충돌
        self.neighbor_exclusion = int(neighbor_exclusion)
        self.thickness_margin   = float(thickness_margin)
        self.safe_margin        = float(safe_margin)

        # 허용 오차
        self.tol_pos  = float(tol_pos)
        self.tol_ori  = float(tol_ori)
        self.tol_gate = float(tol_gate)

        # 에피소드
        self.max_steps = int(max_steps)
        self.topk      = int(topk)

        # 보상 가중치
        self.progress_weight_pos    = float(progress_weight_pos)
        self.success_bonus_pos      = float(success_bonus_pos)
        self.progress_weight_ori    = float(progress_weight_ori)
        self.success_bonus_ori      = float(success_bonus_ori)
        self.pos_hold_weight        = float(pos_hold_weight)
        self.pos_success_bonus_in_O = float(pos_success_bonus_in_O)
        self.collision_penalty      = float(collision_penalty)
        self.safety_weight          = float(safety_weight)
        self.action_smoothing_weight= float(action_smoothing_weight)
        self.action_l2_weight       = float(action_l2_weight)
        self.time_penalty_weight    = float(time_penalty_weight)

        # 자세 편향
        self.z_floor           = float(z_floor)
        self.z_low_min_weight  = float(z_low_min_weight)
        self.z_low_mean_weight = float(z_low_mean_weight)
        self.spread_target     = float(spread_target)
        self.spread_weight     = float(spread_weight)
        self.perp_weight       = float(perp_weight)
        self.back_weight       = float(back_weight)
        self.goal_xy_min_norm  = float(goal_xy_min_norm)

        # 커리큘럼
        self.curriculum_r_start    = float(curriculum_r_start)
        self.curriculum_r_end      = float(curriculum_r_end)
        self.curriculum_episodes   = int(curriculum_episodes)
        self.goal_sample_max_tries = int(goal_sample_max_tries)
        self.ori_angle_start          = float(ori_angle_start)
        self.ori_angle_end            = float(ori_angle_end)
        self.ori_curriculum_episodes  = int(ori_curriculum_episodes)
        self.stage1_wrist_fixed_steps = int(stage1_wrist_fixed_steps)
        self.stage1_wrist_ramp_steps  = int(stage1_wrist_ramp_steps)
        self.stage1_wrist_amp_start   = float(stage1_wrist_amp_start)
        self.stage1_wrist_amp_end     = float(stage1_wrist_amp_end)

        # 성공률
        self.sr_window             = int(sr_window)
        self.sr_pos_advance_thresh = float(sr_pos_advance_thresh)
        self.sr_ori_advance_thresh = float(sr_ori_advance_thresh)
        self.auto_advance_mode     = bool(auto_advance_mode)

        # 랜덤 초기 자세
        self.randomize_start     = bool(randomize_start)
        self.q0_sample_max_tries = int(q0_sample_max_tries)
        self.q0_safe_threshold   = float(safe_margin if q0_safe_threshold is None
                                         else q0_safe_threshold)
        self.q0_move_duration    = float(q0_move_duration)

        # 도메인 랜덤화
        self.obs_noise_std = float(obs_noise_std)

        # Gym spaces
        self.joint_limits = np.asarray(JOINT_LIMITS, dtype=np.float32)
        obs_dim = 6 + 3 + 3 + self.topk + self.topk
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        # 학습 카운터
        self.np_random     = None
        self._seed         = seed
        self.episode_count = 0
        self.steps         = 0
        self.total_steps   = 0

        # 에피소드 상태
        self.q      = np.zeros(6, dtype=np.float32)
        self.p_goal = np.zeros(3, dtype=np.float32)
        self.R_goal = np.eye(3, dtype=np.float32)

        self.pos_bonus_given       = False
        self.ori_bonus_given       = False
        self.pos_reached_once      = False
        self._pos_bonus_in_O_given = False

        self.a_prev   = np.zeros(6, dtype=np.float32)
        self._dq_prev = np.zeros(6, dtype=np.float32)
        self._cache: dict[str, Any] = {}

        self._wrist_mode = "none"
        self._wrist_amp  = 0.0

        self._sr_pos_buf  = deque(maxlen=self.sr_window)
        self._sr_pose_buf = deque(maxlen=self.sr_window)

    # ── 외부 API ──────────────────────────────────────────────────

    def set_ros(self, ros: ROS2Interface)        : self.ros = ros

    def set_reward_mode(self, mode: str):
        if mode not in ("P", "O", "FT"):
            raise ValueError('mode must be "P", "O", or "FT"')
        self.reward_mode = mode

    def set_tolerances(
        self,
        tol_pos: float | None = None,
        tol_ori: float | None = None,
    ):
        if tol_pos is not None: self.tol_pos = float(tol_pos)
        if tol_ori is not None: self.tol_ori = float(tol_ori)

    # ── 성공률 ────────────────────────────────────────────────────

    @staticmethod
    def _buf_rate(buf: deque) -> float:
        return sum(buf) / len(buf) if buf else 0.0

    def get_success_rate_pos(self)  -> float: return self._buf_rate(self._sr_pos_buf)
    def get_success_rate_pose(self) -> float: return self._buf_rate(self._sr_pose_buf)

    # ── 커리큘럼 ─────────────────────────────────────────────────

    def _curriculum_t(self, episodes: int) -> float:
        return min(self.episode_count / episodes, 1.0) if episodes > 0 else 1.0

    def _curriculum_radius(self) -> float:
        t = self._curriculum_t(self.curriculum_episodes)
        return (1.0 - t) * self.curriculum_r_start + t * self.curriculum_r_end

    def _curriculum_ori_angle(self) -> float:
        t = self._curriculum_t(self.ori_curriculum_episodes)
        return (1.0 - t) * self.ori_angle_start + t * self.ori_angle_end

    def _ori_reward_weight(self) -> float:
        """자세 보상 가중치: 커리큘럼 진행도 0→1 (초반 약하게)."""
        return self._curriculum_t(self.ori_curriculum_episodes)

    def _stage1_wrist_schedule(self) -> tuple[str, float]:
        """손목 커리큘럼 (total_steps 기준): fixed0 → ramp → full."""
        if self.reward_mode != "P":
            return "none", 0.0
        s = self.total_steps
        if s < self.stage1_wrist_fixed_steps:
            return "fixed0", 0.0
        s2 = s - self.stage1_wrist_fixed_steps
        if s2 < self.stage1_wrist_ramp_steps:
            t   = min(s2 / max(self.stage1_wrist_ramp_steps, 1), 1.0)
            amp = (1.0 - t) * self.stage1_wrist_amp_start + t * self.stage1_wrist_amp_end
            return "ramp", amp
        return "full", self.stage1_wrist_amp_end

    def _maybe_auto_advance(self):
        """성공률 임계값 달성 시 reward_mode 자동 전환."""
        if not self.auto_advance_mode:
            return
        if self.reward_mode == "P":
            sr = self.get_success_rate_pos()
            if sr >= self.sr_pos_advance_thresh:
                self.set_reward_mode("O")
                print(f"[AutoAdvance] P→O  (ep={self.episode_count}, sr_pos={sr:.2f})")
        elif self.reward_mode == "O":
            sr = self.get_success_rate_pose()
            if sr >= self.sr_ori_advance_thresh:
                self.set_reward_mode("FT")
                print(f"[AutoAdvance] O→FT (ep={self.episode_count}, sr_pose={sr:.2f})")

    # ── q0 안전 샘플 ─────────────────────────────────────────────

    def _compute_d_eff_only(self, q: np.ndarray) -> float:
        """q 한 개의 d_effective_min (q0 안전 판정용)."""
        _, joints, _, poly = fk_full(q)
        disc = discretize_from_fk_output(poly, joints, n_points=100)
        dmin = nearest_nonlocal_distance(
            pairwise_distance_matrix(disc), self.neighbor_exclusion
        )
        return float(np.nanmin(dmin) - self.thickness_margin)

    def _sample_q0_safe_with_wrist(
        self, wrist_mode: str, amp: float
    ) -> tuple[np.ndarray, float, bool]:
        """
        충돌 없는 안전한 초기 자세 q0 샘플.
        Returns: (q0, d_eff, threshold_met)
        """
        lo, hi = self.joint_limits[:, 0], self.joint_limits[:, 1]
        best_q, best_d = np.zeros(6, dtype=np.float32), -np.inf

        for _ in range(self.q0_sample_max_tries):
            q = np.empty(6, dtype=np.float32)
            q[:3] = self.np_random.uniform(lo[:3], hi[:3])
            if wrist_mode == "fixed0":
                q[3:] = 0.0
            elif wrist_mode in ("ramp", "full"):
                q[3:] = self.np_random.uniform(-amp, amp, size=3)
            else:
                q[3:] = self.np_random.uniform(lo[3:], hi[3:])

            q = self._clip_q(q)
            d = self._compute_d_eff_only(q)
            if d > best_d:
                best_d, best_q = d, q.copy()
            if d > self.q0_safe_threshold:
                return best_q, best_d, True

        return best_q, best_d, False

    # ── 목표 샘플 ────────────────────────────────────────────────

    def _clip_q(self, q: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(q, dtype=np.float32),
                       self.joint_limits[:, 0], self.joint_limits[:, 1])

    def _sample_goal_position(self, p_end: np.ndarray) -> np.ndarray:
        """커리큘럼 반경 + rb3 작업공간 조건 만족하는 목표 위치."""
        z_lo, z_hi = self._WS_Z
        r_lo, r_hi = self._WS_R
        r_max = self._curriculum_radius()

        for _ in range(self.goal_sample_max_tries):
            v = self.np_random.normal(size=3)
            n = np.linalg.norm(v)
            if n < 1e-9:
                continue
            g  = p_end + (v / n * self.np_random.uniform(0.0, r_max)).astype(np.float32)
            hd = np.linalg.norm(g[:2])
            if r_lo <= hd <= r_hi and z_lo <= g[2] <= z_hi:
                return g.astype(np.float32)

        # fallback: 작업공간에서 직접 샘플
        theta = self.np_random.uniform(0.0, _TWO_PI)
        hr    = self.np_random.uniform(r_lo, r_hi)
        return np.array([hr * np.cos(theta), hr * np.sin(theta),
                         self.np_random.uniform(z_lo, z_hi)], dtype=np.float32)

    def _sample_goal_orientation(self) -> np.ndarray:
        """커리큘럼 최대 각도 범위 내 랜덤 R_goal (3×3)."""
        angle_max = self._curriculum_ori_angle()
        if angle_max <= 1e-9:
            return np.eye(3, dtype=np.float32)
        axis = self.np_random.normal(size=3)
        n    = np.linalg.norm(axis)
        axis = axis / n if n > 1e-9 else np.array([1., 0., 0.])
        return _axis_angle_to_R(axis, self.np_random.uniform(0.0, angle_max)).astype(np.float32)

    # ── FK → obs 계산 ─────────────────────────────────────────────

    def _compute_all(self) -> np.ndarray:
        """self.q → FK → obs(28D) 계산 + _cache 저장."""
        T07, joints, _, poly = fk_full(self.q)
        p_end = T07[:3, 3].astype(np.float32)
        R_end = T07[:3, :3].astype(np.float32)

        e_pos = (self.p_goal - p_end).astype(np.float32)
        e_ori, ori_err = orientation_error_rotvec(R_end, self.R_goal)

        disc  = discretize_from_fk_output(poly, joints, n_points=100).astype(np.float32)
        dmin  = nearest_nonlocal_distance(
            pairwise_distance_matrix(disc), self.neighbor_exclusion
        )
        d_eff = float(np.nanmin(dmin) - self.thickness_margin)
        topk_d, topk_idx, topk_idx_norm = topk_dmin_with_indices(dmin, k=self.topk)

        obs = np.concatenate([
            self.q, e_pos, e_ori.astype(np.float32),
            topk_d, topk_idx_norm,
        ]).astype(np.float32)

        if self.obs_noise_std > 0.0:
            obs += self.np_random.normal(0, self.obs_noise_std, size=obs.shape).astype(np.float32)

        self._cache = dict(
            p_end=p_end, R_end=R_end,
            e_pos=e_pos, e_ori=e_ori.astype(np.float32), ori_err=float(ori_err),
            disc=disc, dmin=dmin, d_eff=d_eff,
            topk_d=topk_d, topk_idx=topk_idx, topk_idx_norm=topk_idx_norm,
        )
        return obs

    # ── 보상 구성요소 ─────────────────────────────────────────────

    def _r_safety(self) -> float:
        d = self._cache["d_eff"]
        if d <= 0.0:
            return -self.collision_penalty
        if d < self.safe_margin:
            return -self.safety_weight * (self.safe_margin - d) / max(self.safe_margin, 1e-6)
        return 0.0

    def _r_posture(self) -> tuple[float, dict]:
        """Z 낮음 / spread / 방향 편향 페널티."""
        pts = self._cache["disc"]
        z   = pts[:, 2]

        # Z 낮음
        z_min    = float(np.min(z))
        viol_min = max(0.0, self.z_floor - z_min)
        viol_mean= float(np.mean(np.maximum(0.0, self.z_floor - z)))
        r_z      = -(self.z_low_min_weight * viol_min + self.z_low_mean_weight * viol_mean)

        # Spread
        v      = pts[:, :2]
        spread = float(np.mean(np.linalg.norm(v - np.mean(v, axis=0), axis=1)))
        r_spread = -self.spread_weight * max(0.0, spread - self.spread_target)

        # 방향
        goal_xy   = self.p_goal[:2]
        g_norm    = float(np.linalg.norm(goal_xy))
        perp_mean = back_mean = r_dir = 0.0
        if g_norm >= self.goal_xy_min_norm:
            u         = goal_xy / g_norm
            proj      = v @ u
            perp_mean = float(np.mean(np.linalg.norm(v - proj[:, None] * u, axis=1)))
            back_mean = float(np.mean(np.maximum(0.0, -proj)))
            r_dir     = -(self.perp_weight * perp_mean + self.back_weight * back_mean)

        return float(r_z + r_spread + r_dir), dict(
            z_min=z_min, viol_z_min=viol_min, viol_z_mean=viol_mean, r_z=r_z,
            spread=spread, r_spread=r_spread,
            perp_mean=perp_mean, back_mean=back_mean, r_dir=r_dir,
        )

    def _compute_reward(
        self,
        prev_e_pos: float, prev_e_ori: float,
        e_pos_norm: float, e_ori_norm: float,
        success_pos: bool, success_ori: bool,
        common: float,
    ) -> tuple[float, dict]:
        """보상 모든 항목 계산. side effect: pos/ori 보너스 플래그 갱신."""
        gate  = e_pos_norm <= self.tol_gate
        ori_w = self._ori_reward_weight()

        # 위치 진척
        progress_pos   = prev_e_pos - e_pos_norm
        r_pos_progress = self.progress_weight_pos * progress_pos
        r_pos_success  = 0.0
        if not self.pos_bonus_given and success_pos:
            self.pos_bonus_given  = True
            self.pos_reached_once = True
            r_pos_success = self.success_bonus_pos

        # 자세 진척 (gate 이후 + ori_w)
        progress_ori   = prev_e_ori - e_ori_norm
        r_ori_progress = ori_w * self.progress_weight_ori * progress_ori if gate else 0.0
        r_ori_success  = 0.0
        if gate and not self.ori_bonus_given and success_ori:
            self.ori_bonus_given = True
            r_ori_success = ori_w * self.success_bonus_ori

        # 모드별
        r_pos_hold = 0.0
        if self.reward_mode in ("O", "FT"):
            r_pos_hold = -self.pos_hold_weight * max(0.0, e_pos_norm - self.tol_gate)

        r_pos_in_O = 0.0
        if self.reward_mode == "O":
            if self.steps == 1:
                self._pos_bonus_in_O_given = False
            if not self._pos_bonus_in_O_given and success_pos:
                self._pos_bonus_in_O_given = True
                r_pos_in_O = self.pos_success_bonus_in_O

        pos_lock = self.reward_mode == "FT" and self.pos_reached_once
        if pos_lock:
            r_pos_progress *= 0.25

        R_P = r_pos_progress + r_pos_success + common
        R_O = r_ori_progress + r_ori_success + r_pos_hold + r_pos_in_O + common

        if self.reward_mode == "P":
            reward = R_P
        elif self.reward_mode == "O":
            reward = R_O
        else:
            reward = R_P + r_ori_progress + r_ori_success + r_pos_hold

        return float(np.clip(reward, -8.0, 8.0)), dict(
            gate=gate, ori_w=ori_w,
            R_P=R_P, R_O=R_O,
            progress_pos=progress_pos, r_pos_progress=r_pos_progress, r_pos_success=r_pos_success,
            progress_ori=progress_ori, r_ori_progress=r_ori_progress, r_ori_success=r_ori_success,
            r_pos_hold=r_pos_hold,
            pos_reached_once=self.pos_reached_once,
            pos_lock_active=pos_lock,
        )

    # ── 액션 처리 ────────────────────────────────────────────────

    def _process_action(self, action: np.ndarray) -> tuple[np.ndarray, float, float]:
        """action clip → velocity/acceleration 제한 → (dq, r_l2, r_smooth)."""
        a  = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        dq = np.empty(6, dtype=np.float32)
        dq[:3] = a[:3] * self.dq_max_pos
        dq[3:] = a[3:] * self.dq_max_ori

        # acceleration hard limit
        if self.ddq_max_pos > 0:
            dq[:3] = self._dq_prev[:3] + np.clip(
                dq[:3] - self._dq_prev[:3], -self.ddq_max_pos, self.ddq_max_pos
            )
        if self.ddq_max_ori > 0:
            dq[3:] = self._dq_prev[3:] + np.clip(
                dq[3:] - self._dq_prev[3:], -self.ddq_max_ori, self.ddq_max_ori
            )

        r_l2     = -self.action_l2_weight * float(np.dot(dq, dq))
        r_smooth = -self.action_smoothing_weight * float(np.dot(a - self.a_prev, a - self.a_prev))

        self.a_prev   = a
        self._dq_prev = dq.copy()
        return dq, r_l2, r_smooth

    def _execute_action(self, dq: np.ndarray):
        """dq 적용 → ROS2 전송 또는 시뮬 직접 갱신."""
        new_q = self._clip_q(self.q + dq)
        if self.ros is not None:
            self.ros.send_joint_positions(new_q, duration_sec=self.step_sleep)
            time.sleep(self.step_sleep)
            self.q = self.ros.get_joint_pos()
        else:
            self.q = new_q

    # ── Gym API ──────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            super().reset(seed=seed)
            self.np_random = np.random.default_rng(seed)
        elif self.np_random is None:
            self.np_random = np.random.default_rng(self._seed)

        # 에피소드 플래그 초기화
        self.steps      = 0
        self.a_prev     = np.zeros(6, dtype=np.float32)
        self._dq_prev   = np.zeros(6, dtype=np.float32)
        self.pos_bonus_given = self.ori_bonus_given = self.pos_reached_once = False
        self._pos_bonus_in_O_given = False
        self.episode_count += 1

        # 손목 커리큘럼 스케줄
        self._wrist_mode, self._wrist_amp = self._stage1_wrist_schedule()

        # 초기 자세 설정: 항상 홈 복귀 후 필요시 q0로 이동
        if self.ros is not None:
            self.ros.episode_reset()

        if self.randomize_start:
            q0, _, _ = self._sample_q0_safe_with_wrist(self._wrist_mode, self._wrist_amp)
            if self.ros is not None:
                self.ros.send_joint_positions(q0, duration_sec=self.q0_move_duration)
                time.sleep(self.q0_move_duration + 0.3)
                self.q = self.ros.get_joint_pos()
            else:
                self.q = q0
        else:
            self.q = self.ros.get_joint_pos() if self.ros is not None \
                else np.zeros(6, dtype=np.float32)
            # 손목 커리큘럼 강제 적용
            if self._wrist_mode == "fixed0":
                self.q[3:] = 0.0
            elif self._wrist_mode in ("ramp", "full"):
                self.q[3:] = np.clip(self.q[3:], -self._wrist_amp, self._wrist_amp)

        # FK로 현재 TCP
        T07_0, _, _, _ = fk_full(self.q)
        p_end_start = T07_0[:3, 3].astype(np.float32)
        R_end_start = T07_0[:3, :3].astype(np.float32)

        # 목표 샘플
        self.p_goal = self._sample_goal_position(p_end_start)
        if self.reward_mode == "P" and self._wrist_mode == "fixed0":
            self.R_goal, ori_goal_mode = R_end_start.copy(), "R_goal=R_end_start (denoise)"
        else:
            self.R_goal, ori_goal_mode = self._sample_goal_orientation(), "sampled"

        obs = self._compute_all()
        info = {
            "p_goal": self.p_goal.copy(), "R_goal": self.R_goal.copy(),
            "p_end" : self._cache["p_end"].copy(),
            "e_pos_norm"               : float(np.linalg.norm(self._cache["e_pos"])),
            "e_ori_norm"               : float(self._cache["ori_err"]),
            "d_effective_min"          : self._cache["d_eff"],
            "curriculum_r_max"         : self._curriculum_radius(),
            "curriculum_ori_angle_max" : float(np.rad2deg(self._curriculum_ori_angle())),
            "ori_goal_mode"            : ori_goal_mode,
            "episode_count"            : self.episode_count,
            "total_steps"              : self.total_steps,
            "reward_mode"              : self.reward_mode,
            "wrist_mode"               : self._wrist_mode,
            "wrist_amp_rad"            : self._wrist_amp,
            "sr_pos"                   : self.get_success_rate_pos(),
            "sr_pose"                  : self.get_success_rate_pose(),
            "randomize_start"          : self.randomize_start,
        }
        return obs, info

    def step(self, action: np.ndarray):
        self.steps       += 1
        self.total_steps += 1

        # 1. action → dq, 페널티 계산
        dq, r_l2, r_smooth = self._process_action(action)
        prev_e_pos = float(np.linalg.norm(self._cache.get("e_pos", np.zeros(3))))
        prev_e_ori = float(self._cache.get("ori_err", 0.0))

        # 2. 관절 이동
        self._execute_action(dq)

        # 3. FK → obs
        obs = self._compute_all()
        e_pos_norm = float(np.linalg.norm(self._cache["e_pos"]))
        e_ori_norm = float(self._cache["ori_err"])
        d_eff      = float(self._cache["d_eff"])

        # 4. 종료 판정
        collision    = d_eff <= 0.0
        success_pos  = e_pos_norm <= self.tol_pos
        success_ori  = e_ori_norm <= self.tol_ori
        success_pose = success_pos and success_ori
        truncated    = self.steps >= self.max_steps
        terminated   = bool(collision or (success_pos if self.reward_mode == "P" else success_pose))

        # 5. 보상
        r_safety              = self._r_safety()
        r_posture, posture_log = self._r_posture()
        common = r_safety + r_l2 + r_smooth - self.time_penalty_weight + r_posture

        reward, reward_log = self._compute_reward(
            prev_e_pos, prev_e_ori, e_pos_norm, e_ori_norm,
            success_pos, success_ori, common,
        )

        # 6. 성공률 업데이트
        if terminated or truncated:
            self._sr_pos_buf.append(int(success_pos))
            self._sr_pose_buf.append(int(success_pose))
            self._maybe_auto_advance()

        # 7. info
        info = {
            "p_goal": self.p_goal.copy(), "R_goal": self.R_goal.copy(),
            "p_end" : self._cache["p_end"].copy(),
            "e_pos_norm": e_pos_norm, "e_ori_norm": e_ori_norm,
            "d_effective_min": d_eff, "collision": collision,
            "success_pos": success_pos, "success_ori": success_ori, "success_pose": success_pose,
            "reward_mode": self.reward_mode, "reward_used": reward,
            "r_safety": r_safety, "r_l2": r_l2, "r_smooth": r_smooth,
            "r_time": -self.time_penalty_weight, "r_posture": r_posture,
            **reward_log, **posture_log,
            "steps": self.steps, "total_steps": self.total_steps,
            "truncated": truncated, "terminated": terminated,
            "wrist_mode": self._wrist_mode, "wrist_amp_rad": self._wrist_amp,
            "curriculum_r_max": self._curriculum_radius(),
            "curriculum_ori_deg": float(np.rad2deg(self._curriculum_ori_angle())),
            "episode_count": self.episode_count,
            "sr_pos": self.get_success_rate_pos(), "sr_pose": self.get_success_rate_pose(),
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        pass


# ── 동작 확인 (Gazebo 없이) ───────────────────────────────────────

if __name__ == "__main__":
    env = RobotReachEnv(
        ros=None,
        reward_mode="P",
        seed=0,
        stage1_wrist_fixed_steps=200_000,
        auto_advance_mode=True,
        sr_pos_advance_thresh=0.6,
        randomize_start=True,
        q0_sample_max_tries=200,
    )

    obs, info = env.reset()
    print(f"obs  : {obs.shape}  wrist={info['wrist_mode']}  ori_goal={info['ori_goal_mode']}")
    print(f"e_pos={info['e_pos_norm']:.4f}  d_eff={info['d_effective_min']:.4f}")

    ep_r = 0.0
    for t in range(50):
        obs, r, term, trunc, info = env.step(
            np.random.uniform(-0.3, 0.3, size=6).astype(np.float32)
        )
        ep_r += r
        if t < 3 or term or trunc:
            print(f"[{t:3d}] r={r:+.3f}  e_pos={info['e_pos_norm']:.4f}"
                  f"  e_ori={np.rad2deg(info['e_ori_norm']):.1f}°"
                  f"  d_eff={info['d_effective_min']:.4f}  ori_w={info['ori_w']:.2f}")
        if term or trunc:
            print(f"  → 종료 success_pos={info['success_pos']} collision={info['collision']}")
            break
    print(f"\n누적 보상: {ep_r:.3f}  sr_pos={env.get_success_rate_pos():.2f}")
