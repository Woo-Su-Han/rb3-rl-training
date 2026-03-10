"""
robot_fk_rb3_3dof.py — rb3_730es_u 3-DOF 정기구학 및 셀프 충돌 유틸리티

================================================================================
[개요]
================================================================================
  rb3_730es_u는 6-DOF 로봇이지만, 여기서는 앞 3개 관절(q1~q3)만 제어하고
  뒤 3개 손목 관절(q4~q6)은 항상 0(홈 자세)으로 고정한다.

  이 방식의 장점:
    - 학습 난이도 낮음: 3D 위치 도달 태스크만 학습
    - 실제 로봇(Gazebo)에서도 동일하게 q[3:]=0으로 전송
    - FK 결과(TCP 위치, 링크 충돌 거리)가 rb3의 실제 물리 치수를 그대로 사용

  single_robot_env_3dof.py 가 이 파일을 FK 백엔드로 사용한다.

================================================================================
[FK 체인] joint.yaml (rb3_730es_u) 기반 — 손목 고정
================================================================================
  base   : Tz(0.1453) @ Rz(q1)         [axis z]
  shoulder: Ry(q2)                      [axis y, origin 0]
  elbow  : T(0,-0.00645,0.286) @ Ry(q3) [axis y]
  wrist1 : Rz(0)     ← 고정            [axis z]
  wrist2 : Tz(0.344) @ Ry(0) ← 고정   [axis y]
  wrist3 : Rz(0)     ← 고정            [axis z]

[폴리라인 뼈대] (4 꼭짓점)
  P0=[0,0,0] → P1=[0,0,0.1453] → P2=elbow → P3=TCP
"""

from __future__ import annotations
import numpy as np


# ── 관절 한계 (3-DOF) ────────────────────────────────────────────
JOINT_LIMITS_3DOF = np.array(
    [[-np.pi, np.pi]] * 3, dtype=np.float32
)


# ── 기본 변환 행렬 ───────────────────────────────────────────────

def _Rz(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array(
        [[c, -s, 0, 0],
         [s,  c, 0, 0],
         [0,  0, 1, 0],
         [0,  0, 0, 1]], dtype=float
    )


def _Ry(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array(
        [[ c, 0, s, 0],
         [ 0, 1, 0, 0],
         [-s, 0, c, 0],
         [ 0, 0, 0, 1]], dtype=float
    )


def _T(x: float, y: float, z: float) -> np.ndarray:
    return np.array(
        [[1, 0, 0, x],
         [0, 1, 0, y],
         [0, 0, 1, z],
         [0, 0, 0, 1]], dtype=float
    )


# ── 3-DOF 정기구학 ───────────────────────────────────────────────

def fk_3dof(q: np.ndarray) -> tuple:
    """
    rb3_730es_u 3-DOF 정기구학. q4~q6 = 0 고정.

    Args
        q: (3,) 관절 각도 [rad] — [q1_base, q2_shoulder, q3_elbow]

    Returns
        T_tcp      : (4,4) 월드→TCP 변환 행렬 (회전 포함이지만 위치만 사용)
        joints     : (4,3) 관절 위치 [P_base, P_shoulder, P_elbow, P_tcp]
        pts_tran   : (4,3) 폴리라인 꼭짓점 (이산화·시각화용)
        polyline_pts: pts_tran 과 동일
    """
    q = np.asarray(q, dtype=float).flatten()
    if q.size != 3:
        raise ValueError("q must have 3 elements: [q1, q2, q3] in radians")
    q1, q2, q3 = q

    T = np.eye(4, dtype=float)
    P0 = T[:3, 3].copy()               # [0,0,0]  세계 원점

    # base: Tz(0.1453) @ Rz(q1)
    T = T @ _T(0, 0, 0.1453) @ _Rz(q1)
    P1 = T[:3, 3].copy()               # [0,0,0.1453]  어깨 관절

    # shoulder: Ry(q2)  (위치 변화 없음)
    T = T @ _Ry(q2)

    # elbow: T(0,-0.00645,0.286) @ Ry(q3)
    T = T @ _T(0, -0.00645, 0.286) @ _Ry(q3)
    P2 = T[:3, 3].copy()               # 팔꿈치 관절

    # wrist1: Rz(0) — 고정
    T = T @ _Rz(0.0)

    # wrist2: Tz(0.344) @ Ry(0) — 고정
    T = T @ _T(0, 0, 0.344) @ _Ry(0.0)
    P3 = T[:3, 3].copy()               # TCP 위치

    # wrist3: Rz(0) — 고정
    T = T @ _Rz(0.0)

    T_tcp        = T.astype(np.float32)
    joints       = np.array([P0, P1, P2, P3], dtype=np.float32)
    polyline_pts = np.array([P0, P1, P2, P3], dtype=np.float32)
    pts_tran     = polyline_pts.copy()

    return T_tcp, joints, pts_tran, polyline_pts


# ── 이산화 ───────────────────────────────────────────────────────

def _discretize_polyline(poly_pts: np.ndarray, n_points: int = 100) -> np.ndarray:
    """폴리라인 꼭짓점 → 호길이 균등 n_points 개 점."""
    P = np.asarray(poly_pts, dtype=float)
    if P.ndim != 2 or P.shape[0] < 2 or P.shape[1] != 3:
        raise ValueError("poly_pts must be (M,3) with M>=2")

    keep = [0]
    for i in range(1, len(P)):
        if np.linalg.norm(P[i] - P[keep[-1]]) > 1e-12:
            keep.append(i)
    P = P[keep]
    if len(P) < 2:
        return np.repeat(P[:1], n_points, axis=0)

    seg_len = np.linalg.norm(P[1:] - P[:-1], axis=1)
    cum     = np.concatenate([[0.0], np.cumsum(seg_len)])
    total   = cum[-1]
    if total <= 1e-12:
        return np.repeat(P[:1], n_points, axis=0)

    s_targets = np.linspace(0.0, total, n_points)
    out = np.zeros((n_points, 3), dtype=float)
    j = 0
    for i, s in enumerate(s_targets):
        while j < len(seg_len) - 1 and s > cum[j + 1]:
            j += 1
        s0, s1 = cum[j], cum[j + 1]
        t = 0.0 if (s1 - s0) <= 1e-12 else (s - s0) / (s1 - s0)
        out[i] = P[j] + t * (P[j + 1] - P[j])
    return out


def discretize_from_fk_output(
    polyline_pts: np.ndarray,
    joints: np.ndarray,
    n_points: int = 100,
) -> np.ndarray:
    """
    FK 출력으로 폴리라인을 n_points 개로 이산화.
    양 끝은 joints[0](베이스)와 joints[-1](TCP)로 보정.
    """
    disc       = _discretize_polyline(polyline_pts, n_points=n_points)
    disc[0]    = joints[0]
    disc[-1]   = joints[-1]
    return disc.astype(np.float32)


# ── 거리 계산 ────────────────────────────────────────────────────

def pairwise_distance_matrix(pts: np.ndarray) -> np.ndarray:
    """(N,3) → (N,N) 유클리드 거리 행렬."""
    diff = pts[:, None, :] - pts[None, :, :]   # (N,N,3)
    return np.sqrt(np.sum(diff ** 2, axis=-1)).astype(np.float32)


def nearest_nonlocal_distance(
    D: np.ndarray,
    neighbor_exclusion: int = 20,
) -> np.ndarray:
    """
    (N,N) 거리 행렬 → 각 점 i의 비인접(|i-j|>exclusion) 최소 거리.
    반환: dmin (N,)  NaN 포함 가능.
    """
    D   = np.asarray(D, dtype=float)
    n   = D.shape[0]
    idx = np.arange(n)
    dmin = np.full(n, np.nan, dtype=float)
    for i in range(n):
        valid = np.abs(idx - i) > neighbor_exclusion
        if np.any(valid):
            dmin[i] = np.min(D[i, valid])
    return dmin.astype(np.float32)


def topk_dmin_with_indices(
    dmin: np.ndarray,
    k: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    dmin (N,) 에서 가장 작은 k 개 항목 추출.

    Returns
        topk_d        : (k,) 거리
        topk_idx      : (k,) 원본 인덱스
        topk_idx_norm : (k,) 정규화 인덱스 (0~1)
    """
    dmin       = np.asarray(dmin, dtype=float)
    n          = len(dmin)
    valid_idx  = np.where(~np.isnan(dmin))[0]

    if len(valid_idx) == 0:
        pad = np.zeros(k, dtype=np.int32)
        return np.ones(k, dtype=np.float32) * 999.0, pad, np.zeros(k, dtype=np.float32)

    sorted_i   = np.argsort(dmin[valid_idx])
    picked_idx = valid_idx[sorted_i[:k]]

    if len(picked_idx) < k:
        last       = picked_idx[-1] if len(picked_idx) > 0 else 0
        picked_idx = np.concatenate(
            [picked_idx, np.full(k - len(picked_idx), last, dtype=np.int32)]
        )

    topk_d        = dmin[picked_idx].astype(np.float32)
    topk_idx      = picked_idx.astype(np.int32)
    topk_idx_norm = (picked_idx.astype(np.float32) / max(n - 1, 1)).astype(np.float32)
    return topk_d, topk_idx, topk_idx_norm


# ── 시각화 ───────────────────────────────────────────────────────

def fk_and_visualize(
    q: list | np.ndarray,
    show: bool = True,
) -> tuple:
    """
    q(3)에 대한 FK 수행 + matplotlib 3D 시각화.
    Returns T_tcp, joints, pts_tran, polyline_pts
    """
    T_tcp, joints, pts_tran, polyline_pts = fk_3dof(q)

    if show:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(7, 7))
        ax  = fig.add_subplot(111, projection="3d")

        for i in range(len(pts_tran) - 1):
            p0, p1 = pts_tran[i], pts_tran[i + 1]
            ax.plot(
                [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                color="k", linewidth=3,
            )

        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color="r", s=80)
        for i, p in enumerate(joints):
            ax.text(p[0], p[1], p[2], f"P{i}", fontsize=9)

        ax.set_title("rb3_730es_u FK (3-DOF) – links=black, joints=red")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_xlim(-0.7, 0.7); ax.set_ylim(-0.7, 0.7); ax.set_zlim(0, 0.9)
        ax.view_init(elev=25, azim=45)
        plt.tight_layout()
        plt.show()

    return T_tcp, joints, pts_tran, polyline_pts


# ── 단독 실행 확인 ───────────────────────────────────────────────

if __name__ == "__main__":
    import time

    q_test = [0.5, -0.3, 0.8]
    t0 = time.time()

    T_tcp, joints, pts_tran, polyline_pts = fk_3dof(q_test)
    disc   = discretize_from_fk_output(polyline_pts, joints, n_points=100)
    D      = pairwise_distance_matrix(disc)
    dmin   = nearest_nonlocal_distance(D, neighbor_exclusion=20)
    topk_d, topk_idx, topk_idx_norm = topk_dmin_with_indices(dmin, k=8)

    print(f"Elapsed  : {(time.time()-t0)*1000:.1f} ms")
    print(f"TCP pos  : {T_tcp[:3,3]}")
    print(f"d_eff min: {float(np.nanmin(dmin)):.4f} m")
    print(f"topk_d   : {topk_d}")
    print(f"joints   :")
    for i, p in enumerate(joints):
        print(f"  P{i} = {p}")

    fk_and_visualize(q_test, show=True)
