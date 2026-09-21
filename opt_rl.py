# -*- coding: utf-8 -*-
"""픽셀 베르누이 정책 경사(REINFORCE). 강화학습을 쓸지는 main 에서 고른다."""
from __future__ import annotations

import numpy as np

from opt_ga import Individual, evaluate, keep_connected, repair, save_plot, seed_rectangle


def _bernoulli_sample(logits: np.ndarray, rng: np.random.Generator, feed) -> np.ndarray:
    prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -12.0, 12.0)))
    metal = (rng.random(logits.shape) < prob).astype(np.uint8)
    metal = keep_connected(metal, feed)
    metal[feed] = 1
    return metal, prob


def refine_individual(ind: Individual, solver, cfg, rng, freqs, n_steps: int) -> Individual:
    """엘리트 주변에서 짧은 정책 경사 미세조정 (ga_rl 의 세대 사이 RL)."""
    logits = np.where(ind.metal > 0, 2.2, -2.2).astype(np.float64)
    baseline = 0.0
    best = Individual(ind.metal.copy(), ind.mode)
    best_sc, *_ = evaluate(solver, best.metal, freqs, cfg)[:2]
    init = True
    for t in range(n_steps):
        metal, prob = _bernoulli_sample(logits, rng, cfg.feed)
        cand = repair(Individual(metal, ind.mode), cfg.feed)
        sc, worst, s11, mindb, minhz, nused, air = evaluate(solver, cand.metal, freqs, cfg)
        if init:
            baseline = sc
            init = False
        adv = sc - baseline
        baseline = cfg.rl_baseline_momentum * baseline + (1.0 - cfg.rl_baseline_momentum) * sc
        logits += cfg.rl_lr * adv * (cand.metal.astype(np.float64) - prob)
        logits = np.clip(logits, -6.0, 6.0)
        print(
            f"    RL step {t+1}/{n_steps}  score={sc:.2f}  worst={worst:.1f}dB  "
            f"min={mindb:.1f}@{minhz/1e9:.2f}GHz  copper={int(cand.metal.sum())}"
        )
        if sc >= best_sc:
            best_sc = sc
            best = cand
    return best


def run_rl(solver, cfg, start: Individual | None = None) -> Individual:
    rng = np.random.default_rng(cfg.seed + 17)
    feed = cfg.feed
    freqs = np.linspace(cfg.f_start, cfg.f_stop, cfg.n_freq)
    solver.set_feed(*feed)
    start = start or seed_rectangle(feed)
    logits = np.where(start.metal > 0, 2.2, -2.2).astype(np.float64)

    print(
        f"RL start  episodes={cfg.rl_episodes}  lr={cfg.rl_lr}  "
        f"backend={getattr(solver, 'backend', '?')}  fitness={cfg.fitness}"
    )
    baseline = 0.0
    best = Individual(start.metal.copy(), start.mode)
    best_sc, best_worst, best_s11, *_ = evaluate(solver, best.metal, freqs, cfg)
    print(f"  seed score={best_sc:.2f}  worst={best_worst:.1f} dB")
    hist = [best_sc]
    init = True

    for ep in range(1, cfg.rl_episodes + 1):
        metal, prob = _bernoulli_sample(logits, rng, feed)
        cand = repair(Individual(metal, start.mode), feed)
        sc, worst, s11, mindb, minhz, nused, air = evaluate(solver, cand.metal, freqs, cfg)
        if init:
            baseline = sc
            init = False
        adv = sc - baseline
        baseline = cfg.rl_baseline_momentum * baseline + (1.0 - cfg.rl_baseline_momentum) * sc
        logits += cfg.rl_lr * adv * (cand.metal.astype(np.float64) - prob)
        logits = np.clip(logits, -6.0, 6.0)
        hist.append(sc)
        print(
            f"  ep{ep}  score={sc:.2f}  band_worst={worst:.1f}dB  "
            f"min={mindb:.1f}dB@{minhz/1e9:.2f}GHz  steps={nused}  copper={int(cand.metal.sum())}"
        )
        if sc >= best_sc:
            best_sc = sc
            best_worst = worst
            best_s11 = s11
            best = cand
            save_plot(
                cfg.out_dir,
                f"RL_{ep:03d}_Best",
                best.metal,
                best_s11,
                freqs,
                best_sc,
                best_worst,
                feed,
                air=air,
                cfg=cfg,
            )
            print(f"  ★ ep{ep} new best score={best_sc:.2f}  worst={best_worst:.1f} dB")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(cfg.out_dir / "rl_score_history.csv", np.array(hist), delimiter=",")
    save_plot(cfg.out_dir, "RL_final_best", best.metal, best_s11, freqs, best_sc, best_worst, feed, air=air, cfg=cfg)
    return best
