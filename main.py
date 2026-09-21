# -*- coding: utf-8 -*-
"""진화 / 강화학습 방향만 여기서 수정한다."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# 알고리즘
#   ga    : 유전 알고리즘만
#   rl    : 강화학습만
#   ga_rl : GA 후 엘리트를 RL 로 미세조정
# ---------------------------------------------------------------------------
ALGORITHM = "ga"  # "ga" | "rl" | "ga_rl"

# 무엇을 더 좋은 안테나로 볼지
#   gain       : 지향성 × 정합효율 (실현이득 근사). 최종 목표
#   band_worst : 목표 대역에서 가장 나쁜 S11 만
#   f0         : 중심 주파수 S11 만
#   bandwidth  : 대역 안에서 S11 < S11_TARGET_DB 인 비율
FITNESS = "gain"  # "gain" | "band_worst" | "f0" | "bandwidth"
W_DIRECTIVITY = 1.0  # 한 방향 집중. 근거리 peak/평균 쓰지 않음(점광원 버그)
W_EFFICIENCY = 1.0
MIN_COPPER = 80      # 이보다 작으면 안테나로 안 봄 (3px 급전만 고득점 방지)
BAND_HZ = (7.00e9, 7.50e9)
F0_HZ = 7.25e9
S11_TARGET_DB = -10.0
F_START = 6.0e9
F_STOP = 8.5e9
N_FREQ = 51

# GA
POPULATION = 8
GENERATIONS = 10
MUT_PIXELS = 48
TOURNAMENT = 3
SYMMETRY_MODES = [-1, 1, 4]  # -1 없음, 1 직교 4방, 4 x축 반사
DIFFUSE_STEPS = 2   # 3x3 확산 행렬을 몇 번 적용할지
DIFFUSE_RATE = 0.45 # 이웃으로 구리가 번질 확률 스케일
# 다음 세대 JPG/점수는 이전보다 반드시 높을 때만 기록한다.
# 자손이 못 이기면 버리고 다시 만든다. 한 세대당 이 횟수만큼 돌려도 갱신 없으면 중단.
MAX_OFFSPRING_BATCHES = 40
# 자손마다 가장자리 국소상승. 배치당 GA 외에 RL5(지표별 맵) + 디퓨전B 5.
EXTRA_RL = 5
EXTRA_DIFFUSE = 5
LOCAL_CLIMB_STEPS = 8
LOCAL_TRY_PER_STEP = 4
DIFFUSE_RANK = 8  # 디퓨전 B: 자손−엘리트 PCA 축 수. 3×3은 국소 성장만.
RANDOM_START = 1.0  # gen1 자손 100% 랜덤. 세대 갈수록 감소. circle 예열이면 예열 이후부터.
# 온도는 샘플 수·정렬로 내리지 않음. 정렬은 수렴 지표로만 찍음.

# 짧은 원 예열 → 그 디스크 꼭대기들로 GA 시작. 원 안 최고 ≠ 전역.
CIRCLE_WARMUP = True
CIRCLE_R_MODE = "interval"  # "linear" | "discrete" | "interval"
CIRCLE_R_MAX = 24.0
CIRCLE_R_STEP = 4.0       # linear 일 때
CIRCLE_R_INTERVAL = 12.0  # interval 일 때
CIRCLE_RINGS = [8.0, 16.0, 24.0]  # discrete 일 때
CIRCLE_ANGLES = 4
CIRCLE_SEEDS = 3
CIRCLE_FAIL_ANGLES = 2  # 엘리트 못 넘길 때 원 테두리 추가 각 수

# RL (ALGORITHM 이 rl / ga_rl 일 때)
RL_EPISODES = 20
RL_STEPS_PER_GEN = 0
RL_LR = 0.08
RL_BASELINE_MOMENTUM = 0.9

FEED = (25, 25)
SEED = 0
BACKEND = "cuda"  # 연산만. "cpu" | "cuda"
OUT_DIR = Path(__file__).resolve().parent / "run_out"
BYPRODUCT_DIR = Path(__file__).resolve().parent / "run_byproduct"
SAVE_BYPRODUCT = True  # True 면 run_byproduct 에 자손 마스크/CSV 저장 (디버깅용)


@dataclass
class Config:
    backend: str
    algorithm: str
    population: int
    generations: int
    mut_pixels: int
    tournament: int
    symmetry_modes: list
    fitness: str
    band_hz: tuple
    f0_hz: float
    s11_target_db: float
    w_directivity: float
    w_efficiency: float
    min_copper: int
    f_start: float
    f_stop: float
    n_freq: int
    rl_episodes: int
    rl_steps_per_gen: int
    rl_lr: float
    rl_baseline_momentum: float
    feed: tuple
    seed: int
    diffuse_steps: int = 2
    diffuse_rate: float = 0.45
    offspring_retries: int = 40
    extra_rl: int = 5
    extra_diffuse: int = 5
    local_climb_steps: int = 8
    local_try_per_step: int = 4
    diffuse_rank: int = 8
    random_start: float = 1.0
    circle_warmup: bool = True
    circle_r_mode: str = "interval"
    circle_r_max: float = 24.0
    circle_r_step: float = 4.0
    circle_r_interval: float = 12.0
    circle_rings: list = field(default_factory=lambda: [8.0, 16.0, 24.0])
    circle_angles: int = 4
    circle_seeds: int = 3
    circle_fail_angles: int = 2
    out_dir: Path = field(default_factory=lambda: OUT_DIR)
    byproduct_dir: Path = field(default_factory=lambda: BYPRODUCT_DIR)
    save_byproduct: bool = False


def make_config() -> Config:
    return Config(
        backend=BACKEND,
        algorithm=ALGORITHM,
        population=POPULATION,
        generations=GENERATIONS,
        mut_pixels=MUT_PIXELS,
        tournament=TOURNAMENT,
        symmetry_modes=list(SYMMETRY_MODES),
        fitness=FITNESS,
        band_hz=BAND_HZ,
        f0_hz=F0_HZ,
        s11_target_db=S11_TARGET_DB,
        w_directivity=W_DIRECTIVITY,
        w_efficiency=W_EFFICIENCY,
        min_copper=MIN_COPPER,
        f_start=F_START,
        f_stop=F_STOP,
        n_freq=N_FREQ,
        rl_episodes=RL_EPISODES,
        rl_steps_per_gen=RL_STEPS_PER_GEN,
        rl_lr=RL_LR,
        rl_baseline_momentum=RL_BASELINE_MOMENTUM,
        feed=FEED,
        seed=SEED,
        diffuse_steps=DIFFUSE_STEPS,
        diffuse_rate=DIFFUSE_RATE,
        offspring_retries=MAX_OFFSPRING_BATCHES,
        extra_rl=EXTRA_RL,
        extra_diffuse=EXTRA_DIFFUSE,
        local_climb_steps=LOCAL_CLIMB_STEPS,
        local_try_per_step=LOCAL_TRY_PER_STEP,
        diffuse_rank=DIFFUSE_RANK,
        random_start=RANDOM_START,
        circle_warmup=CIRCLE_WARMUP,
        circle_r_mode=CIRCLE_R_MODE,
        circle_r_max=CIRCLE_R_MAX,
        circle_r_step=CIRCLE_R_STEP,
        circle_r_interval=CIRCLE_R_INTERVAL,
        circle_rings=list(CIRCLE_RINGS),
        circle_angles=CIRCLE_ANGLES,
        circle_seeds=CIRCLE_SEEDS,
        circle_fail_angles=CIRCLE_FAIL_ANGLES,
        out_dir=OUT_DIR,
        byproduct_dir=BYPRODUCT_DIR,
        save_byproduct=SAVE_BYPRODUCT,
    )


def make_solver(backend: str):
    backend = backend.lower().strip()
    if backend == "cuda":
        from solver_cuda import VoxelFDTD
    elif backend == "cpu":
        from solver_cpu import VoxelFDTD
    else:
        raise ValueError(f"BACKEND 는 'cpu' 또는 'cuda' 여야 합니다: {backend!r}")
    return VoxelFDTD()


def main():
    cfg = make_config()
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 8) - 1)))
    print(
        f"config  algorithm={cfg.algorithm}  fitness={cfg.fitness}  "
        f"band={cfg.band_hz[0]/1e9:.2f}-{cfg.band_hz[1]/1e9:.2f} GHz  backend={cfg.backend}"
    )
    from openems.gate import ensure_before_optimize

    ensure_before_optimize(cfg.backend)
    solver = make_solver(cfg.backend)
    try:
        algo = cfg.algorithm.lower().strip()
        if algo == "ga":
            from opt_ga import run_ga

            run_ga(solver, cfg)
        elif algo == "rl":
            from opt_rl import run_rl

            run_rl(solver, cfg)
        elif algo == "ga_rl":
            from opt_ga import run_ga
            from opt_rl import refine_individual, run_rl

            best = run_ga(solver, cfg, rl_refine=refine_individual)
            print("GA 종료 → RL 미세조정")
            run_rl(solver, cfg, start=best)
        else:
            raise ValueError(f"ALGORITHM 은 'ga' | 'rl' | 'ga_rl' 이어야 합니다: {algo!r}")
    finally:
        solver.close()
    print("done", cfg.out_dir)


if __name__ == "__main__":
    main()
