"""
test_env.py — RobotReachEnv 테스트 및 롤아웃 시각화

================================================================================
[사용법]
================================================================================

  # 1. 기능 테스트만 (Gazebo + ROS2 필요)
  ros2 run rbpodo_rl test_env

  # 2. 학습된 모델로 롤아웃 + 3D 애니메이션
  ros2 run rbpodo_rl test_env --model runs/sac_P_seed0_xxx/best_model/best_model.zip

  # 3. 애니메이션 파일로 저장
  ros2 run rbpodo_rl test_env --model ... --save rollout.mp4 --n-episodes 5

  # 4. 랜덤 액션 롤아웃 (모델 없이)
  ros2 run rbpodo_rl test_env --random --n-episodes 3
"""

from __future__ import annotations

import os
import sys
import argparse
import threading
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import rclpy
from stable_baselines3 import SAC, PPO

from rbpodo_rl.envs.ros2_interface import ROS2Interface
from rbpodo_rl.envs.single_robot_env import RobotReachEnv
from rbpodo_rl.envs.robot_fk_rb3 import fk_and_visualize


# ── 롤아웃 ────────────────────────────────────────────────────────

def rollout_policy(model, env: RobotReachEnv, max_steps: int = 200):
    """
    1 에피소드를 결정론적으로 실행하고 기록.

    Returns
        qs    : (T, 6) 관절 각도 시계열
        p_goal: (3,) 목표 위치
        R_goal: (3,3) 목표 자세
        infos : list of info dict
    """
    obs, info = env.reset()
    p_goal = info["p_goal"].copy()
    R_goal = info["R_goal"].copy()

    qs    = [env.q.copy()]
    infos = [info]

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        qs.append(env.q.copy())
        infos.append(info)

        if terminated or truncated:
            break

    return np.asarray(qs, dtype=float), p_goal, R_goal, infos


def rollout_random(env: RobotReachEnv, max_steps: int = 200):
    """랜덤 액션으로 1 에피소드 실행."""
    obs, info = env.reset()
    p_goal = info["p_goal"].copy()
    R_goal = info["R_goal"].copy()

    qs    = [env.q.copy()]
    infos = [info]

    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        qs.append(env.q.copy())
        infos.append(info)
        if terminated or truncated:
            break

    return np.asarray(qs, dtype=float), p_goal, R_goal, infos


# ── FK 프레임 계산 ─────────────────────────────────────────────────

def compute_frames_from_qs(qs: np.ndarray):
    """
    각 q에 대해 FK 수행 → 시각화용 joints, pts_tran 리스트 반환.
    """
    joints_list = []
    tran_list   = []
    for q in qs:
        _, joints, pts_tran, _ = fk_and_visualize(q, show=False)
        joints_list.append(np.asarray(joints,   dtype=float))
        tran_list.append(  np.asarray(pts_tran, dtype=float))
    return joints_list, tran_list


# ── 3D 애니메이션 ─────────────────────────────────────────────────

def animate_rollout(
    joints_list: list,
    tran_list:   list,
    p_goal:  np.ndarray,
    infos:   list,
    save_path: str | None = None,
    fps: int = 10,
    show: bool = True,
):
    """
    Matplotlib 3D 애니메이션:
      - 링크: 검정 폴리라인
      - 관절: 빨간 점
      - TCP  : 파란 점 (동적)
      - 목표 : 초록 별 (고정)
      - 텍스트: 위치 오차, 자세 오차, 충돌 마진
    """
    T = len(tran_list)
    goal_arr = np.asarray(p_goal, dtype=float).reshape(3,)

    fig = plt.figure(figsize=(9, 8))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_title("rb3_730es_u RL Rollout")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_zlim(0.0,  0.9)
    ax.view_init(elev=25, azim=45)

    pts0    = np.asarray(tran_list[0],   dtype=float)
    jts0    = np.asarray(joints_list[0], dtype=float)
    p_end0  = jts0[-1]

    (link_line,)  = ax.plot(
        pts0[:, 0], pts0[:, 1], pts0[:, 2],
        color="k", linewidth=3,
    )
    joints_sc = ax.scatter(jts0[:, 0], jts0[:, 1], jts0[:, 2], color="r", s=60)
    ee_sc     = ax.scatter([p_end0[0]], [p_end0[1]], [p_end0[2]], color="b", s=120, zorder=5)
    goal_sc   = ax.scatter(
        [goal_arr[0]], [goal_arr[1]], [goal_arr[2]],
        color="g", marker="*", s=300, zorder=5,
    )
    txt = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=8,
                    verticalalignment="top")

    def _update(i: int):
        pts   = np.asarray(tran_list[i],   dtype=float)
        jts   = np.asarray(joints_list[i], dtype=float)
        p_end = jts[-1]

        link_line.set_data(pts[:, 0], pts[:, 1])
        link_line.set_3d_properties(pts[:, 2])
        joints_sc._offsets3d = (jts[:, 0], jts[:, 1], jts[:, 2])
        ee_sc._offsets3d     = ([p_end[0]], [p_end[1]], [p_end[2]])

        info = infos[i] if i < len(infos) else {}
        e_p   = info.get("e_pos_norm",      float(np.linalg.norm(goal_arr - p_end)))
        e_o   = info.get("e_ori_norm",      0.0)
        d_e   = info.get("d_effective_min", 0.0)
        mode  = info.get("reward_mode",     "?")
        wm    = info.get("wrist_mode",      "?")
        sr_p  = info.get("sr_pos",          0.0)
        ori_w = info.get("ori_w",           0.0)

        txt.set_text(
            f"Frame {i:3d}/{T-1}  mode={mode}  wrist={wm}  sr_pos={sr_p:.2f}\n"
            f"e_pos={e_p:.4f}m  e_ori={np.rad2deg(e_o):.1f}°  d_eff={d_e:.3f}m  ori_w={ori_w:.2f}"
        )
        return link_line, joints_sc, ee_sc, goal_sc, txt

    ani = animation.FuncAnimation(
        fig, _update, frames=T,
        interval=int(1000 / fps), blit=False,
    )

    if save_path:
        save_path = os.path.abspath(save_path)
        ext = os.path.splitext(save_path)[1].lower()
        if ext == ".mp4":
            ani.save(save_path, writer=animation.FFMpegWriter(fps=fps))
        elif ext == ".gif":
            ani.save(save_path, writer=animation.PillowWriter(fps=fps))
        else:
            raise ValueError("save_path 는 .mp4 또는 .gif 여야 합니다.")
        print(f"[INFO] 저장 완료: {save_path}")

    if show:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)


# ── 단독 기능 테스트 ──────────────────────────────────────────────

def run_feature_tests(env: RobotReachEnv):
    """reset/step/관측 차원 확인."""
    print("\n[TEST 1] 관측 공간 확인")
    obs_dim = env.observation_space.shape[0]
    print(f"  obs_dim = {obs_dim}  (기대: 28)")
    assert obs_dim == 28, f"관측 차원 오류: {obs_dim}"
    print("  PASS")

    print("\n[TEST 2] reset()")
    obs, info = env.reset()
    assert obs.shape == (28,), f"obs shape: {obs.shape}"
    q    = obs[:6]
    e_p  = obs[6:9]
    e_o  = obs[9:12]
    td   = obs[12:20]
    tn   = obs[20:28]
    print(f"  q         = {np.round(q, 3)}")
    print(f"  e_pos     = {np.round(e_p, 4)}")
    print(f"  e_ori     = {np.round(e_o, 4)}")
    print(f"  topk_d    = {np.round(td, 4)}")
    print(f"  topk_idx_n= {np.round(tn, 4)}")
    print(f"  p_goal    = {np.round(info['p_goal'], 4)}")
    print(f"  d_eff_min = {info['d_effective_min']:.4f}")
    print(f"  wrist_mode= {info['wrist_mode']}")
    print(f"  ori_goal  = {info['ori_goal_mode']}")
    print("  PASS")

    print("\n[TEST 3] step() 10 회")
    ep_reward = 0.0
    for t in range(10):
        a = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(a)
        ep_reward += r
        assert obs.shape == (28,)
        print(
            f"  [{t:2d}] r={r:+.3f}  e_pos={info['e_pos_norm']:.4f}"
            f"  e_ori={np.rad2deg(info['e_ori_norm']):.1f}°"
            f"  d_eff={info['d_effective_min']:.4f}"
            f"  ori_w={info.get('ori_w',0.):.2f}"
            f"  col={info['collision']}  done={terminated or truncated}"
        )
        if terminated or truncated:
            print("      → 에피소드 종료")
            break
    print(f"  누적 보상: {ep_reward:.3f}  PASS")

    print("\n[TEST 4] 에피소드 루프 (max_steps 까지)")
    obs, info = env.reset()
    done = False
    total_r = 0.0
    n_steps = 0
    while not done:
        a = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(a)
        total_r += r
        n_steps += 1
        done = terminated or truncated
    print(f"  steps={n_steps}  total_r={total_r:.3f}"
          f"  collision={info['collision']}  success_pose={info['success_pose']}")
    print("  PASS")


def load_model(path: str):
    try:
        return SAC.load(path)
    except Exception:
        return PPO.load(path)


# ── 엔트리포인트 ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RobotReachEnv 테스트")
    parser.add_argument("--model",      type=str,  default=None,
                        help="학습된 모델 .zip 경로")
    parser.add_argument("--random",     action="store_true",
                        help="랜덤 액션으로 롤아웃")
    parser.add_argument("--n-episodes", type=int,  default=1)
    parser.add_argument("--max-steps",  type=int,  default=200)
    parser.add_argument("--save",       type=str,  default=None,
                        help="애니메이션 저장 경로 (.mp4 또는 .gif)")
    parser.add_argument("--fps",        type=int,  default=10)
    parser.add_argument("--no-show",    action="store_true",
                        help="화면 표시 없이 파일만 저장")
    parser.add_argument("--tests-only", action="store_true",
                        help="기능 테스트만 실행 (모델 불필요)")
    parser.add_argument("--reward-mode",type=str,  default="P",
                        choices=["P", "O", "FT"])
    args = parser.parse_args()

    # ── ROS2 초기화 ────────────────────────────────────────────────
    rclpy.init()
    ros = ROS2Interface()
    spin_thread = threading.Thread(target=rclpy.spin, args=(ros,), daemon=True)
    spin_thread.start()

    print("[INFO] joint_states 대기...")
    if not ros.wait_for_joint_states(timeout=30.0):
        print("[ERROR] joint_states 수신 실패.")
        rclpy.shutdown()
        sys.exit(1)
    print("[INFO] 준비 완료.")

    env = RobotReachEnv(ros=ros, reward_mode=args.reward_mode, seed=42)

    # ── 기능 테스트 ────────────────────────────────────────────────
    if args.tests_only or (args.model is None and not args.random):
        run_feature_tests(env)
        rclpy.shutdown()
        spin_thread.join(timeout=3.0)
        return

    # ── 모델 로드 ─────────────────────────────────────────────────
    model = None
    if args.model:
        print(f"[INFO] 모델 로드: {args.model}")
        model = load_model(args.model)

    # ── 롤아웃 + 애니메이션 ──────────────────────────────────────
    for ep in range(args.n_episodes):
        print(f"\n[INFO] Episode {ep+1}/{args.n_episodes}")

        if model is not None:
            qs, p_goal, R_goal, infos = rollout_policy(
                model, env, max_steps=args.max_steps
            )
        else:
            qs, p_goal, R_goal, infos = rollout_random(
                env, max_steps=args.max_steps
            )

        n_steps   = len(qs) - 1
        last_info = infos[-1]
        print(f"  steps={n_steps}"
              f"  e_pos={last_info.get('e_pos_norm', 0.):.4f}m"
              f"  e_ori={np.rad2deg(last_info.get('e_ori_norm', 0.)):.1f}°"
              f"  d_eff={last_info.get('d_effective_min', 0.):.4f}m"
              f"  success={last_info.get('success_pose', False)}")

        # FK 프레임 계산
        joints_list, tran_list = compute_frames_from_qs(qs)

        # 저장 경로: 여러 에피소드면 파일명에 번호 붙임
        save_path = None
        if args.save:
            base, ext = os.path.splitext(args.save)
            save_path = f"{base}_{ep:03d}{ext}" if args.n_episodes > 1 else args.save

        animate_rollout(
            joints_list=joints_list,
            tran_list=tran_list,
            p_goal=p_goal,
            infos=infos,
            save_path=save_path,
            fps=args.fps,
            show=not args.no_show,
        )

    # ── 정리 ─────────────────────────────────────────────────────
    rclpy.shutdown()
    spin_thread.join(timeout=3.0)


if __name__ == "__main__":
    main()
