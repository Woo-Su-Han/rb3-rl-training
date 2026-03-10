"""
train.py — rb3_730es_u Gazebo RL 학습 스크립트 (Stable-Baselines3)

================================================================================
[이 파일이 하는 일]
================================================================================
  RobotReachEnv(Gazebo+ROS2) 와 상호작용하며 정책(신경망)을 학습시킴.
  - 알고리즘: SAC 또는 PPO (--algo로 선택)
  - reward_mode: P→O→FT 단계적 전환 지원 (--reward-mode)
  - VecNormalize: obs 정규화 선택 (--normalize)
  - EvalCallback: best 모델 자동 저장
  - CheckpointCallback: 주기적 체크포인트 저장
  - --resume --ckpt: 이어학습 지원

  실행 예:
    # Stage 1 (위치 학습)
    python train.py --reward-mode P --total-steps 300000

    # Stage 2 (자세 학습, Stage1 이어서)
    python train.py --reward-mode O --resume \\
        --run-path runs/sac_P_seed0_xxx \\
        --ckpt    runs/sac_P_seed0_xxx/checkpoints/model_300000_steps.zip

    # TensorBoard
    tensorboard --logdir runs
"""

from __future__ import annotations

import os
import sys
import argparse
import time
import threading
from datetime import datetime

import numpy as np
import torch as th

import rclpy
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed

from rbpodo_rl.envs.ros2_interface import ROS2Interface
from rbpodo_rl.envs.single_robot_env import RobotReachEnv


# ── 강제 이어학습 (Spyder 등 IDE에서 CLI 없이 쓸 때) ──────────────
FORCE_RESUME     = False
FORCE_RUN_PATH   = r"runs/sac_P_seed0_20260101_000000"
FORCE_CKPT_PATH  = r"runs/sac_P_seed0_20260101_000000/checkpoints/model_300000_steps.zip"
FORCE_NORMALIZE  = False
FORCE_RESET_TS   = False


def make_env(ros, reward_mode: str, seed: int, auto_advance: bool = False):
    """SB3 DummyVecEnv 용 환경 생성 함수. ros=None 이면 Gazebo 없이 FK 시뮬레이션."""
    def _init():
        env = RobotReachEnv(
            ros=ros,
            reward_mode=reward_mode,
            seed=seed,
            randomize_start=True,
            auto_advance_mode=auto_advance,
        )
        return env
    return _init


def _find_existing(candidates: list[str]) -> str | None:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def main():
    parser = argparse.ArgumentParser(
        description="rb3_730es_u Gazebo RL 학습"
    )

    # ── 학습 ──────────────────────────────────────────────────────
    parser.add_argument("--algo",         type=str, default="sac",
                        choices=["sac", "ppo"])
    parser.add_argument("--reward-mode",  type=str, default="P",
                        choices=["P", "O", "FT"],
                        help="보상 모드 시작점: P=위치, O=자세, FT=둘 다")
    parser.add_argument("--auto-advance", action="store_true",
                        help="성공률 임계값 달성 시 P→O→FT 자동 전환")
    parser.add_argument("--total-steps",  type=int, default=300_000)
    parser.add_argument("--seed",         type=int, default=0)
    parser.add_argument("--log-dir",      type=str, default="runs")
    parser.add_argument("--eval-freq",    type=int, default=10_000)
    parser.add_argument("--eval-episodes",type=int, default=10)
    parser.add_argument("--save-freq",    type=int, default=50_000)
    parser.add_argument("--normalize",    action="store_true",
                        help="VecNormalize로 obs 정규화")

    # ── 이어학습 ──────────────────────────────────────────────────
    parser.add_argument("--resume",    action="store_true")
    parser.add_argument("--run-path",  type=str, default=None,
                        help="기존 런 폴더 경로")
    parser.add_argument("--ckpt",      type=str, default=None,
                        help="체크포인트 .zip 경로")
    parser.add_argument("--reset-timesteps", action="store_true",
                        help="TensorBoard 타임스텝 0부터 재시작")
    parser.add_argument("--no-ros", action="store_true",
                        help="Gazebo 없이 FK 시뮬레이션만으로 학습 (테스트용)")

    args = parser.parse_args()

    # Spyder 등에서 FORCE_RESUME=True 이면 위 값으로 오버라이드
    if FORCE_RESUME:
        args.algo         = "sac"
        args.resume       = True
        args.run_path     = FORCE_RUN_PATH
        args.ckpt         = FORCE_CKPT_PATH
        args.normalize    = FORCE_NORMALIZE
        args.reset_timesteps = FORCE_RESET_TS

    set_random_seed(args.seed)

    # ── 런 폴더 설정 ───────────────────────────────────────────────
    if args.run_path:
        run_path = args.run_path
        run_name = os.path.basename(run_path.rstrip("/\\"))
    else:
        run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.algo}_{args.reward_mode}_seed{args.seed}_{run_id}"
        run_path = os.path.join(args.log_dir, run_name)

    tb_path   = os.path.join(run_path, "tb")
    ckpt_path = os.path.join(run_path, "checkpoints")
    best_path = os.path.join(run_path, "best_model")
    for p in (run_path, tb_path, ckpt_path, best_path):
        os.makedirs(p, exist_ok=True)

    # ── ROS2 초기화 (--no-ros 이면 건너뜀) ────────────────────────
    ros         = None
    spin_thread = None

    if not args.no_ros:
        rclpy.init()
        ros = ROS2Interface()

        spin_thread = threading.Thread(target=rclpy.spin, args=(ros,), daemon=True)
        spin_thread.start()

        print("[INFO] Gazebo joint_states 대기 중...")
        if not ros.wait_for_joint_states(timeout=30.0):
            print("[ERROR] joint_states 수신 실패. Gazebo 실행 확인.")
            rclpy.shutdown()
            sys.exit(1)
        print("[INFO] joint_states 수신 완료.")
    else:
        print("[INFO] --no-ros 모드: Gazebo 없이 FK 시뮬레이션으로 학습합니다.")

    # ── 환경 구성 ─────────────────────────────────────────────────
    # ROS2 환경은 단일 인스턴스만 가능 (Gazebo 연결)
    train_env = DummyVecEnv([make_env(ros, args.reward_mode, args.seed,
                                      auto_advance=args.auto_advance)])
    train_env = VecMonitor(train_env)

    # eval 환경은 auto_advance 끔 (평가는 고정 모드에서 수행)
    eval_env  = DummyVecEnv([make_env(ros, args.reward_mode, args.seed + 10_000,
                                      auto_advance=False)])
    eval_env  = VecMonitor(eval_env)

    # ── VecNormalize ──────────────────────────────────────────────
    if args.normalize:
        if args.resume:
            vn_path = _find_existing([
                os.path.join(ckpt_path, "vecnormalize.pkl"),
                os.path.join(run_path,  "vecnormalize.pkl"),
            ])
            if vn_path is None:
                raise FileNotFoundError("VecNormalize 활성화됐으나 vecnormalize.pkl 없음.")
            train_env = VecNormalize.load(vn_path, train_env)
            train_env.training = True
            train_env.norm_reward = False
            eval_env = VecNormalize.load(vn_path, eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
        else:
            train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
            eval_env  = VecNormalize(eval_env,  norm_obs=True, norm_reward=False, clip_obs=10.0)
            eval_env.training  = False
            eval_env.norm_reward = False

    # ── 콜백 ─────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.save_freq, 1),
        save_path=ckpt_path,
        name_prefix="model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_path,
        log_path=os.path.join(run_path, "eval"),
        eval_freq=max(args.eval_freq, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    # ── 모델 생성 또는 로드 ────────────────────────────────────────
    algo_cls = SAC if args.algo == "sac" else PPO

    if args.resume:
        if not args.ckpt:
            raise ValueError("--resume 에는 --ckpt 가 필요합니다.")
        if not os.path.exists(args.ckpt):
            raise FileNotFoundError(f"체크포인트 없음: {args.ckpt}")

        print(f"[INFO] 이어학습: {args.ckpt}")
        model = algo_cls.load(
            args.ckpt,
            env=train_env,
            verbose=1,
            tensorboard_log=tb_path,
            seed=args.seed,
        )

        # SAC 이어학습: entropy 계수 고정 (탐험 안정화)
        if args.algo == "sac":
            fixed_ent = 0.01
            model.ent_coef = fixed_ent
            model.log_ent_coef = None
            model.ent_coef_optimizer = None
            model.ent_coef_tensor = th.tensor(fixed_ent, device=model.device, dtype=th.float32)

            rb_path = args.ckpt.replace(".zip", "_replay_buffer.pkl")
            if os.path.exists(rb_path):
                print(f"[INFO] 재생 버퍼 로드: {rb_path}")
                model.load_replay_buffer(rb_path)
            else:
                print("[WARN] 재생 버퍼 없음. 이어학습은 가능하지만 수렴이 느릴 수 있습니다.")
    else:
        if args.algo == "sac":
            model = SAC(
                policy="MlpPolicy",
                env=train_env,
                verbose=1,
                tensorboard_log=tb_path,
                seed=args.seed,
                learning_rate=3e-4,
                buffer_size=300_000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                learning_starts=10_000,
                ent_coef=0.005,
                policy_kwargs=dict(net_arch=[256, 256]),
            )
        else:
            model = PPO(
                policy="MlpPolicy",
                env=train_env,
                verbose=1,
                tensorboard_log=tb_path,
                seed=args.seed,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=256,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.0,
                policy_kwargs=dict(net_arch=[256, 256]),
            )

    # ── 학습 ─────────────────────────────────────────────────────
    print(f"[INFO] Run   : {run_name}")
    print(f"[INFO] Logs  : {run_path}")
    print(f"[INFO] Mode  : reward_mode={args.reward_mode}, algo={args.algo}")
    t0 = time.time()

    try:
        model.learn(
            total_timesteps=args.total_steps,
            callback=[checkpoint_cb, eval_cb],
            progress_bar=True,
            tb_log_name=run_name,
            reset_num_timesteps=args.reset_timesteps,
        )
    finally:
        # 학습 중단/완료 모두 모델 저장 보장
        final_path = os.path.join(run_path, "final_model.zip")
        model.save(final_path)
        print(f"[INFO] 최종 모델: {final_path}")

        if args.normalize:
            vn_out = os.path.join(run_path, "vecnormalize.pkl")
            train_env.save(vn_out)
            print(f"[INFO] VecNormalize: {vn_out}")

        elapsed = time.time() - t0
        print(f"[INFO] 완료. 경과 시간: {elapsed:.1f}s")

        # ROS2 정리
        if not args.no_ros:
            rclpy.shutdown()
            if spin_thread is not None:
                spin_thread.join(timeout=3.0)


if __name__ == "__main__":
    main()
