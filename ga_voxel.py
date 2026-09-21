# -*- coding: utf-8 -*-
"""
51x51 픽셀 패치 GA. 평가는 고정 메쉬 voxel FDTD (금속 비트맵만 교체).

기존 simule_mini.py 에서 고친 점
- repair_chromosome 이 잘못된 사분면을 복사하며 모든 대칭 모드를 Mode1 로 덮어쓰던 것 삭제
- 급전점에서 4-연결이 아닌 구리는 제거 (소금-후추 금지)
- 돌연변이 5%(~130비트) → 소수 픽셀만
- 점수: 1.5 GHz 전체 max(S11) 대신 7.00–7.50 GHz 최악 S11
- 세대마다 ParaView/openEMS 덤프 제거
- Windows Pool+stdout dup2 대신, FDTD 내부 OpenMP (프로세스 1개)
- 검증된 사각 패치를 엘리트로 시드
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from voxel_fdtd import VoxelFDTD, rectangle_mask

PIX = 51
CENTER = PIX // 2
OUT = Path(__file__).resolve().parent / "ga_out"

# 7.25 GHz ± 250 MHz. 예전 6.5–8.0 전대역 -10 dB 는 패치에 비현실적.
BAND = (7.00e9, 7.50e9)
F0 = 7.25e9
N_FREQ = 51
POP = 6
GENS = 6
MUT_PIXELS = 12
TOURNEY = 3


@dataclass
class Individual:
    metal: np.ndarray  # (51,51) uint8
    mode: int  # -1: 대칭 강제 없음, 0..6: D4 부분군


def apply_symmetry(core_src: np.ndarray, mode: int) -> np.ndarray:
    """mode 에 따라 코어를 펼친다. mode<0 이면 입력 복사만."""
    ind = np.zeros((PIX, PIX), dtype=np.uint8)
    if mode < 0:
        ind[:] = core_src
        return ind
    half = CENTER + 1
    if mode == 1:
        for i in range(half):
            for j in range(half):
                v = int(core_src[CENTER + i, CENTER + j])
                ind[CENTER + i, CENTER + j] = v
                ind[CENTER - i, CENTER + j] = v
                ind[CENTER + i, CENTER - j] = v
                ind[CENTER - i, CENTER - j] = v
    elif mode == 4:
        for i in range(PIX):
            for j in range(half):
                v = int(core_src[i, CENTER + j])
                ind[i, CENTER + j] = v
                ind[i, CENTER - j] = v
    elif mode == 3:
        for i in range(half):
            for j in range(PIX):
                v = int(core_src[CENTER + i, j])
                ind[CENTER + i, j] = v
                ind[CENTER - i, j] = v
    else:
        # 0,2,5,6 은 일단 Mode1 과 같은 직교 반사 (구현 단순화)
        return apply_symmetry(core_src, 1)
    return ind


def keep_connected(metal: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    """급전 픽셀에서 4-연결로 닿는 구리만 남김."""
    out = np.zeros_like(metal)
    if metal[seed] == 0:
        metal = metal.copy()
        metal[seed] = 1
    stack = [seed]
    seen = np.zeros_like(metal, dtype=bool)
    seen[seed] = True
    out[seed] = 1
    while stack:
        i, j = stack.pop()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < PIX and 0 <= nj < PIX and not seen[ni, nj] and metal[ni, nj]:
                seen[ni, nj] = True
                out[ni, nj] = 1
                stack.append((ni, nj))
    return out


def repair(ind: Individual, feed: tuple[int, int]) -> Individual:
    metal = keep_connected(ind.metal.astype(np.uint8), feed)
    if ind.mode >= 0:
        metal = apply_symmetry(metal, ind.mode)
        metal = keep_connected(metal, feed)
    metal[feed] = 1
    return Individual(metal=metal, mode=ind.mode)


def seed_rectangle(feed: tuple[int, int]) -> Individual:
    g = rectangle_mask(PIX, 9.0, 11.7)
    g[feed] = 1
    return repair(Individual(g, mode=-1), feed)


def seed_blob(rng: np.random.Generator, feed: tuple[int, int], mode: int) -> Individual:
    g = np.zeros((PIX, PIX), dtype=np.uint8)
    # 급전 주변 작은 타원 덩어리. 예전 50% 랜덤 점과 달리 처음부터 연결됨.
    rr, cc = np.ogrid[:PIX, :PIX]
    ax = rng.integers(4, 8)
    ay = rng.integers(5, 9)
    g[((rr - feed[0]) / ax) ** 2 + ((cc - feed[1]) / ay) ** 2 <= 1.0] = 1
    noise = rng.random((PIX, PIX)) < 0.08
    g[noise] = 1
    return repair(Individual(g, mode=mode), feed)


def crossover(a: Individual, b: Individual, rng: np.random.Generator, feed) -> Individual:
    mask = rng.random((PIX, PIX)) < 0.5
    child = np.where(mask, a.metal, b.metal).astype(np.uint8)
    mode = a.mode if rng.random() < 0.5 else b.mode
    return repair(Individual(child, mode), feed)


def mutate(ind: Individual, rng: np.random.Generator, feed) -> Individual:
    g = ind.metal.copy()
    ys, xs = np.nonzero(g)
    if len(xs) == 0:
        g[feed] = 1
        return Individual(g, ind.mode)
    for _ in range(MUT_PIXELS):
        if rng.random() < 0.5 and len(xs):
            k = int(rng.integers(0, len(xs)))
            i, j = int(ys[k]), int(xs[k])
            # 가장자리만 뒤집기: 이웃 중 빈칸으로 확장하거나 내부 구멍
            if rng.random() < 0.5:
                g[i, j] = 0
            else:
                di, dj = int(rng.integers(-1, 2)), int(rng.integers(-1, 2))
                ni, nj = np.clip(i + di, 0, PIX - 1), np.clip(j + dj, 0, PIX - 1)
                g[ni, nj] = 1
        else:
            i = int(rng.integers(0, PIX))
            j = int(rng.integers(0, PIX))
            g[i, j] = 1 - g[i, j]
    g[feed] = 1
    return repair(Individual(g, ind.mode), feed)


def score_s11(s11: np.ndarray, freqs: np.ndarray) -> tuple[float, float]:
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    worst = float(np.max(s11[band]))
    at_f0 = float(s11[np.argmin(np.abs(freqs - F0))])
    # 최대화: 대역 최악 S11이 더 음수일수록 좋음. f0 도 약간 가중.
    score = -0.7 * worst - 0.3 * at_f0
    return score, worst


def save_plot(gen: int, metal, s11, freqs, score, worst, feed):
    OUT.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(metal.T, origin="lower", cmap="Greys", extent=[-25.5, 25.5, -25.5, 25.5])
    ax1.plot(feed[0] - 25, feed[1] - 25, "rx", ms=8, mew=2)
    ax1.set_title(f"Gen {gen} best  pixels={int(metal.sum())}")
    ax1.set_xlabel("x (mm)")
    ax1.set_ylabel("y (mm)")
    ax2.plot(freqs / 1e9, s11, "b-")
    ax2.axhline(-10, color="r", ls="--")
    ax2.axvline(7.25, color="gray", ls=":")
    ax2.set_ylim([-40, 0])
    ax2.grid(True, ls=":")
    ax2.set_title(f"score={score:.2f}  worst in band={worst:.1f} dB")
    ax2.set_xlabel("GHz")
    ax2.set_ylabel("S11 (dB)")
    fig.tight_layout()
    fig.savefig(OUT / f"Generation_{gen:03d}_Best.jpg", dpi=130)
    plt.close(fig)


def run_ga():
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 8) - 1)))
    rng = np.random.default_rng(0)
    feed = (23, 25)  # 검증 패치와 동일 오프셋 급전
    freqs = np.linspace(6.0e9, 8.5e9, N_FREQ)
    solver = VoxelFDTD()
    solver.set_feed(*feed)

    pop = [seed_rectangle(feed)]
    while len(pop) < POP:
        mode = int(rng.choice([-1, 1, 4]))
        pop.append(seed_blob(rng, feed, mode))

    print(f"GA start  pop={POP} gens={GENS}  voxel FDTD  feed={feed}")
    best_hist = []

    for gen in range(1, GENS + 1):
        scores = []
        details = []
        for n, ind in enumerate(pop):
            s11, mindb, minhz, nused = solver.run(ind.metal, freqs)
            sc, worst = score_s11(s11, freqs)
            scores.append(sc)
            details.append((s11, worst, mindb, minhz, nused))
            print(
                f"  gen{gen} ind{n+1}  score={sc:.2f}  band_worst={worst:.1f}dB  "
                f"min={mindb:.1f}dB@{minhz/1e9:.2f}GHz  steps={nused}  copper={int(ind.metal.sum())}"
            )

        bi = int(np.argmax(scores))
        best_hist.append(scores[bi])
        s11, worst, *_ = details[bi]
        save_plot(gen, pop[bi].metal, s11, freqs, scores[bi], worst, feed)
        print(f"  ★ gen{gen} best score={scores[bi]:.2f}  worst={worst:.1f} dB")

        new_pop = [Individual(pop[bi].metal.copy(), pop[bi].mode)]
        while len(new_pop) < POP:
            def pick():
                ix = rng.choice(POP, size=TOURNEY, replace=False)
                return pop[int(ix[np.argmax([scores[i] for i in ix])])]

            child = crossover(pick(), pick(), rng, feed)
            child = mutate(child, rng, feed)
            new_pop.append(child)
        pop = new_pop

    np.savetxt(OUT / "score_history.csv", np.array(best_hist), delimiter=",")
    solver.close()
    print("done", OUT)


if __name__ == "__main__":
    run_ga()
