# rbpodo_rl — Docker로 학습 환경 구성

## 사전 조건

- Docker 20.10+
- Docker Compose v2+
- (Gazebo GUI 사용 시) `xhost +local:docker` 실행

---

## 빠른 시작

### 1. 이미지 빌드

```bash
cd rbpodo_ws
docker compose build
```

> 처음 빌드는 10~20분 소요 (ROS2 + Gazebo 다운로드)

---

### 2-A. Gazebo 없이 학습 (FK 시뮬레이션, 권장)

```bash
docker compose run --rm train-noros
```

학습 결과는 호스트의 `runs/` 폴더에 저장됩니다.

---

### 2-B. Gazebo 포함 학습

```bash
# X11 소켓 허용 (Gazebo GUI 표시용)
xhost +local:docker

# Gazebo + 학습 동시 실행
docker compose up
```

---

### 3. TensorBoard로 학습 확인

```bash
docker compose up tensorboard
# 브라우저에서 http://localhost:6006
```

---

## 이어학습

```bash
docker compose run --rm train-noros \
  python3 /ros2_ws/src/rbpodo_rl/rbpodo_rl/train.py \
  --no-ros \
  --resume \
  --run-path runs/sac_P_seed0_YYYYMMDD_HHMMSS \
  --ckpt    runs/sac_P_seed0_YYYYMMDD_HHMMSS/checkpoints/model_300000_steps.zip
```

---

## 커스텀 학습 옵션

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--reward-mode P/O/FT` | 보상 모드 시작점 | `P` |
| `--auto-advance` | P→O→FT 자동 전환 | 꺼짐 |
| `--total-steps N` | 총 학습 스텝 수 | 300,000 |
| `--algo sac/ppo` | 알고리즘 선택 | `sac` |
| `--normalize` | obs 정규화 (VecNormalize) | 꺼짐 |
| `--no-ros` | Gazebo 없이 FK 시뮬 | 꺼짐 |

---

## 폴더 구조

```
rbpodo_ws/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── docker/
│   └── entrypoint.sh
└── src/
    ├── rbpodo_rl/          ← 학습 코드
    ├── rbpodo_gazebo/      ← Gazebo launch
    ├── rbpodo_ros2/        ← URDF, 컨트롤러
    └── ...
```

---

## GPU 학습 (선택)

`docker-compose.yml`의 `train-noros` 서비스에 아래 추가:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

그리고 `requirements.txt`에서 torch를 GPU 버전으로 변경:

```
# CPU
torch>=2.0.0

# GPU (CUDA 12.1)
# --index-url https://download.pytorch.org/whl/cu121
# torch>=2.0.0
```
