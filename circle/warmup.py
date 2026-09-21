# -*- coding: utf-8 -*-
"""짧은 원 예열. 디스크 안 꼭대기 몇 개를 GA 초기집단으로 넘김."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from circle.chart import sample_on_ring
from circle.schedule import RadiusSchedule, make_schedule
from core import Individual, clone_ind, hamming, seed_random, seed_rectangle
from learn.viz import save_reveal


def snap_reveal(cfg, model, tag: str, log=None, note: str = "", r=None) -> float | None:
    if model is None:
        return None
    if r is not None:
        model.chart.set_r_probed(float(r))
    det = save_reveal(cfg.out_dir, tag, model, note=note)
    if log is not None:
        log(f"    reveal {tag}  n={int(det)}")
    return det


def _diverse_pick(peaks: list[Individual], n: int, min_hamming: int = 40) -> list[Individual]:
    scored = [p for p in peaks if p.score is not None]
    scored.sort(key=lambda x: x.score if x.score is not None else -1e18, reverse=True)
    picked: list[Individual] = []
    for p in scored:
        if len(picked) >= n:
            break
        if all(hamming(p.metal, q.metal) >= min_hamming for q in picked):
            picked.append(clone_ind(p))
    for p in scored:
        if len(picked) >= n:
            break
        if not any(np.array_equal(p.metal, q.metal) for q in picked):
            picked.append(clone_ind(p))
    return picked


def schedule_from_cfg(cfg) -> RadiusSchedule:
    return make_schedule(
        getattr(cfg, "circle_r_mode", "interval"),
        r_max=float(getattr(cfg, "circle_r_max", 24.0)),
        step=float(getattr(cfg, "circle_r_step", 4.0)),
        interval=float(getattr(cfg, "circle_r_interval", 12.0)),
        rings=list(getattr(cfg, "circle_rings", [8.0, 16.0, 24.0])),
    )


def probe_after_fail(
    center: Individual,
    cfg,
    rng: np.random.Generator,
    climb: Callable[[Individual], Individual],
    model=None,
    attempt: int = 0,
) -> tuple[float, list[Individual]]:
    """엘리트를 못 넘긴 뒤, 갱신된 맵/B 로 원 테두리를 조금 더 밝힌다."""
    radii = schedule_from_cfg(cfg).radii()
    r = float(radii[min(max(int(attempt), 0), len(radii) - 1)])
    n_ang = max(1, int(getattr(cfg, "circle_fail_angles", 2)))
    offset = float(rng.random() * 2.0 * np.pi)
    b = None if model is None else getattr(model, "b", None)
    peaks = []
    for i in range(n_ang):
        th = offset + 2.0 * np.pi * i / n_ang
        kid = sample_on_ring(
            center.metal, r, th, rng, cfg.feed, mode=center.mode, b=b
        )
        peaks.append(climb(kid))
    if model is not None:
        model.chart.set_center(center.metal)
        model.chart.set_r_probed(r)
    return r, peaks


def inject_peak(
    pop: list[Individual], elite: Individual, peak: Individual, n: int
) -> list[Individual]:
    """엘리트는 유지하고, 추가탐색 최고를 집단에 넣는다."""
    out = [clone_ind(elite)]
    rest = [p for p in pop if not np.array_equal(p.metal, elite.metal)]
    if peak.score is None or np.array_equal(peak.metal, elite.metal):
        return (out + rest)[:n]
    rest = [p for p in rest if not np.array_equal(p.metal, peak.metal)]
    rest.append(clone_ind(peak))
    rest.sort(key=lambda x: x.score if x.score is not None else -1e18, reverse=True)
    return (out + rest)[:n]


def make_seeds(rng: np.random.Generator, feed, n: int, modes: list) -> list[Individual]:
    seeds = [seed_rectangle(feed)]
    while len(seeds) < max(int(n), 1):
        mode = int(rng.choice(modes if modes else [-1]))
        seeds.append(seed_random(rng, feed, mode))
    return seeds[:n]


def run_circle_warmup(
    cfg,
    rng: np.random.Generator,
    climb: Callable[[Individual], Individual],
    model=None,
    log=None,
    progress=None,
) -> list[Individual]:
    """
    시드 몇 개 주위를 스케줄 r 로만 짧게 밝힌다.
    climb(ind) 는 평가+국소상승을 끝낸 Individual 을 돌려야 한다.
    """
    feed = cfg.feed
    modes = list(getattr(cfg, "symmetry_modes", [-1]))
    n_seed = int(getattr(cfg, "circle_seeds", 3))
    n_ang = int(getattr(cfg, "circle_angles", 4))
    n_pop = int(cfg.population)
    sched: RadiusSchedule = schedule_from_cfg(cfg)
    radii = sched.radii()
    n_jobs = max(n_seed * (1 + len(radii) * n_ang), 1)
    done = 0

    def say(msg: str):
        if log is not None:
            log(msg)
        else:
            print(msg, flush=True)

    def tick(msg: str, inc: bool = False):
        nonlocal done
        if inc:
            done += 1
        if progress is not None:
            progress(min(done / n_jobs, 1.0), msg)

    say(
        f"  circle warmup  seeds={n_seed}  schedule={sched.name}  "
        f"r={radii}  angles={n_ang}"
    )
    peaks: list[Individual] = []
    seeds = make_seeds(rng, feed, n_seed, modes)
    b = None if model is None else getattr(model, "b", None)

    for si, seed in enumerate(seeds):
        seed.src = f"cseed{si+1}"
        if model is not None:
            model.chart.set_center(seed.metal)
        tick(f"seed {si+1}/{n_seed} climb")
        seed = climb(seed)
        tick(f"seed {si+1}/{n_seed}", inc=True)
        peaks.append(seed)
        center = seed.metal
        if model is not None:
            model.chart.set_center(center)
        say(
            f"    seed{si+1}/{n_seed}  score={seed.score:.2f}  "
            f"copper={int(seed.metal.sum())}"
        )
        snap_reveal(cfg, model, f"Reveal_w_s{si+1:02d}", log=log, note=f"seed{si+1}", r=0)
        for r in radii:
            thetas = np.linspace(0.0, 2.0 * np.pi, n_ang, endpoint=False)
            for ai, theta in enumerate(thetas):
                tick(f"seed {si+1}/{n_seed} r={r:g} ang={ai+1}/{n_ang}")
                kid = sample_on_ring(
                    center, r, float(theta), rng, feed, mode=seed.mode, b=b
                )
                kid = climb(kid)
                tick(f"seed {si+1}/{n_seed} r={r:g} ang={ai+1}/{n_ang}", inc=True)
                peaks.append(kid)
            snap_reveal(
                cfg,
                model,
                f"Reveal_w_s{si+1:02d}_r{int(round(r)):03d}",
                log=log,
                note=f"r={r:g}",
                r=r,
            )
            if log is not None:
                say(f"    r={r:g}  last={peaks[-1].score:.2f}  n_peaks={len(peaks)}")

    picked = _diverse_pick(peaks, n_pop)
    while len(picked) < n_pop:
        extra = seed_random(rng, feed, int(rng.choice(modes)))
        extra = climb(extra)
        picked.append(extra)
    best = max(picked, key=lambda x: x.score or -1e18)
    say(
        f"  circle done  peaks={len(peaks)}  init_pop={len(picked)}  "
        f"disk_best={best.score:.2f}  (원 안 최고 ≠ 전역)"
    )
    return picked
