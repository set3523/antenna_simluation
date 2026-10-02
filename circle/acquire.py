# -*- coding: utf-8 -*-
"""Next cell: only angles with large fog x promise, not the whole ring."""
from __future__ import annotations

import numpy as np

from learn.infer import predict


def ang_diff(a: float, b: float) -> float:
    return float(abs(np.arctan2(np.sin(a - b), np.cos(a - b))))


def acquire_field(
    uu: np.ndarray,
    vv: np.ndarray,
    su: np.ndarray,
    sv: np.ndarray,
    ss: np.ndarray,
    ell: float,
    *,
    bal: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A = (1-c) * (0.35+0.65 hope) * (balance)
    -1 if a direct FDTD point is already nearby or it is a confidently low valley.
    """
    uu = np.asarray(uu, dtype=np.float64).ravel()
    vv = np.asarray(vv, dtype=np.float64).ravel()
    su = np.asarray(su, dtype=np.float64)
    sv = np.asarray(sv, dtype=np.float64)
    ss = np.asarray(ss, dtype=np.float64)
    n = int(uu.size)
    if n == 0:
        z = np.zeros(0, dtype=np.float64)
        return z, z, z
    if ss.size == 0:
        return np.ones(n), np.zeros(n), np.zeros(n)

    pred, conf = predict(uu, vv, su, sv, ss, ell)
    s_best = float(np.max(ss))
    s_low = float(np.percentile(ss, 25))
    span = abs(s_best - s_low) + 1e-6
    exist_r = np.hypot(su, sv)
    exist_th = np.arctan2(sv, su)
    n_sec = 8
    if exist_th.size:
        sec = np.floor((exist_th + np.pi) / (2.0 * np.pi) * n_sec) % n_sec
        cnt = np.bincount(sec.astype(int), minlength=n_sec).astype(np.float64)
        cmax = float(np.max(cnt)) + 1.0
    else:
        cnt = np.zeros(n_sec)
        cmax = 1.0

    acq = np.zeros(n, dtype=np.float64)
    for i in range(n):
        d = np.hypot(uu[i] - su, vv[i] - sv)
        if float(np.min(d)) < 0.40 * ell:
            acq[i] = -1.0
            continue
        r_i = float(np.hypot(uu[i], vv[i]))
        th_i = float(np.arctan2(vv[i], uu[i]))
        near_ring = np.abs(exist_r - r_i) < 0.45 * ell
        if np.any(near_ring):
            dth = np.abs(
                np.arctan2(
                    np.sin(th_i - exist_th[near_ring]),
                    np.cos(th_i - exist_th[near_ring]),
                )
            )
            if float(np.min(dth)) * r_i < 0.35 * ell:
                acq[i] = -1.0
                continue
        fog = 1.0 - float(conf[i])
        if float(conf[i]) < 1e-6:
            hope = 1.0
        else:
            if float(conf[i]) > 0.55 and float(pred[i]) < s_low - 0.10 * span:
                acq[i] = -1.0
                continue
            hope = float(np.clip((pred[i] - s_low) / span, 0.0, 1.0))
        balf = 1.0
        if bal and exist_th.size:
            si = int(np.floor((th_i + np.pi) / (2.0 * np.pi) * n_sec) % n_sec)
            balf = 1.0 - cnt[si] / cmax
        acq[i] = fog * (0.35 + 0.65 * hope) * (0.40 + 0.60 * balf)
    return acq, pred, conf


def pick_cone_probe(
    r: float,
    heading: float,
    su: np.ndarray,
    sv: np.ndarray,
    ss: np.ndarray,
    *,
    ell: float,
    n_cand: int,
    half: float,
    rng: np.random.Generator,
    avoid: list[float] | None = None,
    min_sep: float = 0.45,
) -> tuple[float, float]:
    """Angle with the largest A inside the cone along the tip heading. acq<0 if none."""
    r = float(r)
    n_cand = max(int(n_cand), 3)
    half = max(float(half), 0.15)
    jitter = float(rng.uniform(-0.08, 0.08))
    thetas = heading + jitter + np.linspace(-half, half, n_cand)
    uu = r * np.cos(thetas)
    vv = r * np.sin(thetas)
    acq, _, _ = acquire_field(uu, vv, su, sv, ss, ell, bal=False)
    order = np.argsort(-acq)
    avoid = list(avoid or [])
    for i in order:
        if acq[i] < 0.0:
            continue
        th = float(thetas[i])
        if any(ang_diff(th, a) < min_sep for a in avoid):
            continue
        return th, float(acq[i])
    i = int(order[0])
    return float(thetas[i]), float(acq[i])


def pick_inertial_probe(
    r: float,
    heading: float,
    omega: float,
    su: np.ndarray,
    sv: np.ndarray,
    ss: np.ndarray,
    *,
    ell: float,
    n_cand: int,
    half: float,
    rng: np.random.Generator,
    avoid: list[float] | None = None,
    min_sep: float = 0.45,
) -> tuple[float, float]:
    """Cone biased toward the predicted angle (heading+ω). A weighted by alignment."""
    r = float(r)
    n_cand = max(int(n_cand), 3)
    half = max(float(half), 0.12)
    pred = float(heading + omega)
    thetas = pred + np.linspace(-half, half, n_cand)
    thetas = thetas + float(rng.uniform(-0.03, 0.03))
    uu = r * np.cos(thetas)
    vv = r * np.sin(thetas)
    acq, _, _ = acquire_field(uu, vv, su, sv, ss, ell, bal=False)
    align = np.exp(-np.array([ang_diff(float(t), pred) for t in thetas]) ** 2 / (2.0 * half * half + 1e-9))
    score = acq * (0.30 + 0.70 * align)
    score = np.where(acq < 0.0, -1.0, score)
    order = np.argsort(-score)
    avoid = list(avoid or [])
    for i in order:
        if score[i] < 0.0:
            continue
        th = float(thetas[i])
        if any(ang_diff(th, a) < min_sep for a in avoid):
            continue
        return th, float(acq[i])
    i = int(order[0])
    return float(thetas[i]), float(acq[i])


def pick_heading_split(
    r: float,
    heading: float,
    su: np.ndarray,
    sv: np.ndarray,
    ss: np.ndarray,
    *,
    ell: float,
    owned: list[float],
    sep: float = 0.52,
    min_acq: float = 0.28,
    offsets: tuple[float, ...] = (0.52, -0.52),
) -> tuple[float, float, float] | None:
    """
    Branch if the same metric does not have this angle yet.
    Default offset ±30°. Skip if a trend line for the same metric already has that angle.
    """
    r = float(r)
    best: tuple[float, float, float] | None = None
    for off in offsets:
        th = float(heading + off)
        if any(ang_diff(th, h) < sep for h in owned):
            continue
        uu = np.array([r * np.cos(th)])
        vv = np.array([r * np.sin(th)])
        acq, _, _ = acquire_field(uu, vv, su, sv, ss, ell, bal=False)
        a = float(acq[0])
        if a < float(min_acq):
            continue
        if best is None or a > best[1]:
            best = (th, a, float(np.sign(off) or 1.0))
    return best


def pick_fork_theta(
    r: float,
    su: np.ndarray,
    sv: np.ndarray,
    ss: np.ndarray,
    *,
    ell: float,
    n_cand: int,
    rng: np.random.Generator,
    used: list[float],
    min_sep: float,
    min_acq: float,
) -> tuple[float, float] | None:
    """One angle on the full circle with max fog x promise, away from existing trend lines."""
    r = float(r)
    n_cand = max(int(n_cand), 8)
    offset = float(rng.random() * 2.0 * np.pi)
    thetas = offset + np.linspace(0.0, 2.0 * np.pi, n_cand, endpoint=False)
    uu = r * np.cos(thetas)
    vv = r * np.sin(thetas)
    acq, _, _ = acquire_field(uu, vv, su, sv, ss, ell, bal=True)
    order = np.argsort(-acq)
    for i in order:
        if float(acq[i]) < float(min_acq):
            break
        th = float(thetas[i])
        if any(ang_diff(th, u) < min_sep for u in used):
            continue
        return th, float(acq[i])
    return None


def assign_metric_key(
    u: float,
    v: float,
    su: np.ndarray,
    sv: np.ndarray,
    series: dict[str, np.ndarray],
    ell: float,
    keys: tuple[str, ...] = ("beam", "eta", "leave"),
) -> str:
    """Metric with the largest acquisition at that cell."""
    best_k = "beam"
    best_a = -1e18
    uu = np.array([u], dtype=np.float64)
    vv = np.array([v], dtype=np.float64)
    for key in keys:
        ss = series.get(key)
        if ss is None or ss.size == 0:
            continue
        acq, _, _ = acquire_field(uu, vv, su, sv, ss, ell, bal=False)
        if float(acq[0]) > best_a:
            best_a = float(acq[0])
            best_k = key
    return best_k


def pick_ring_probes(
    r: float,
    su: np.ndarray,
    sv: np.ndarray,
    ss: np.ndarray,
    *,
    ell: float,
    n_probe: int,
    n_cand: int,
    rng: np.random.Generator,
) -> list[float]:
    """
    Only candidate angles on the r circle with large acquisition.
    A(θ) = (1-c) * hope
    Skip if a direct FDTD point is already nearby.
    """
    r = float(r)
    n_probe = max(int(n_probe), 1)
    n_cand = max(int(n_cand), n_probe)
    offset = float(rng.random() * 2.0 * np.pi)
    thetas = offset + np.linspace(0.0, 2.0 * np.pi, n_cand, endpoint=False)
    uu = r * np.cos(thetas)
    vv = r * np.sin(thetas)
    if np.asarray(ss).size == 0:
        pick = np.linspace(0, n_cand, n_probe, endpoint=False).astype(int)
        return [float(thetas[i]) for i in pick]
    acq, _, conf = acquire_field(uu, vv, su, sv, ss, ell, bal=True)
    order = np.argsort(-acq)
    min_sep = 0.70 * (2.0 * np.pi) / float(max(n_probe, 2))
    picked: list[float] = []

    def far_enough(th: float) -> bool:
        return all(ang_diff(th, p) >= min_sep for p in picked)

    for i in order:
        if acq[i] < 0.0:
            continue
        if not far_enough(float(thetas[i])):
            continue
        picked.append(float(thetas[i]))
        if len(picked) >= n_probe:
            break
    if len(picked) < n_probe:
        for i in order:
            if acq[i] < 0.0 or float(thetas[i]) in picked:
                continue
            picked.append(float(thetas[i]))
            if len(picked) >= n_probe:
                break
    if not picked:
        picked.append(float(thetas[int(np.argmax(1.0 - conf))]))
    return picked
