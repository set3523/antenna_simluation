# -*- coding: utf-8 -*-
"""Chart kernel inference. Not a GP (kriging); weights only nearby measured points."""
from __future__ import annotations

import numpy as np

# Gaussian kernel cutoff in units of length scale ℓ
K_CUT = 2.2
# Confidence >= this -> orange (inferred). Below -> gray (unexplored)
TAU_ORANGE = 0.18


def disk_rings(r_disk: float, r_step: float) -> list[float]:
    """Fill the disk r<=r_disk at Δr spacing with no empty rings."""
    step = max(float(r_step), 1e-6)
    r_out = max(float(r_disk), step)
    n = max(int(round(r_out / step)), 1)
    return [step * float(i) for i in range(1, n + 1)]


def n_ang_to_cover(
    r: float,
    *,
    r_step: float = 4.0,
    tau: float = TAU_ORANGE,
    safety: float = 0.88,
) -> int:
    """
    Number of angles needed on the circle |x|=r so the gaps between them are orange (c≥τ).
    Uses c=1-exp(-Σw)≥τ, not the 2.2ℓ kernel cutoff. Two neighboring points:
    2 exp(-(d/ℓ)²) ≥ -ln(1-τ),  d=2 r sin(π/(2N)).
    ℓ shrinks for inner points, so only the 0.55 Δr floor is used. For r≤8, 6 angles fill it.
    """
    r = max(float(r), 1e-6)
    if r <= 8.0 + 1e-9:
        return 6
    ell = 0.55 * max(float(r_step), 1e-6)
    ell = float(np.clip(ell, 1.0, 8.0))
    wneed = -np.log(max(1.0 - float(tau), 1e-9))
    d_max = float(safety) * ell * float(np.sqrt(-np.log(0.5 * wneed)))
    ratio = min(d_max / (2.0 * r), 0.999999)
    n = int(np.ceil(np.pi / (2.0 * np.arcsin(ratio))))
    return int(np.clip(n, 6, 48))


def length_scale(
    su: np.ndarray,
    sv: np.ndarray,
    r_step: float,
    r_seed: float = 8.0,
    n_seed_ang: int = 6,
) -> float:
    """Kernel length ℓ. The smaller of neighbor spacing on the seed circle and the r step."""
    step = max(float(r_step), 1e-6)
    n_ang = max(int(n_seed_ang), 3)
    arc = float(r_seed) * (2.0 * np.pi / n_ang)
    ell = 0.55 * min(step, arc)
    if su.size >= 2:
        n = int(su.size)
        if n > 512:
            # with thousands of chart points an O(n²) Python loop looks frozen
            j = min(512, n)
            ix = np.linspace(0, n - 1, j, dtype=np.intp)
            su_u, sv_u = su[ix], sv[ix]
        else:
            su_u, sv_u = su, sv
        du = su_u[:, None] - su_u[None, :]
        dv = sv_u[:, None] - sv_u[None, :]
        d2 = du * du + dv * dv
        np.fill_diagonal(d2, np.inf)
        med = float(np.median(np.sqrt(np.min(d2, axis=1))))
        if np.isfinite(med) and med > 1e-6:
            ell = max(ell, 0.50 * med)
    return float(np.clip(ell, 1.0, 8.0))


def _weights(
    u: np.ndarray, v: np.ndarray, su: np.ndarray, sv: np.ndarray, ell: float, k_cut: float = K_CUT
) -> np.ndarray:
    """w_i = exp(-||x-x_i||^2 / ℓ^2), zero when ||x-x_i|| > kℓ.  shape (..., n)."""
    ell2 = max(float(ell) * float(ell), 1e-12)
    d2 = (u[..., None] - su) ** 2 + (v[..., None] - sv) ** 2
    w = np.exp(-d2 / ell2)
    reach2 = (float(k_cut) * float(ell)) ** 2
    return np.where(d2 <= reach2, w, 0.0)


def predict(
    u: np.ndarray,
    v: np.ndarray,
    su: np.ndarray,
    sv: np.ndarray,
    ss: np.ndarray,
    ell: float,
    k_cut: float = K_CUT,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predicted height and confidence.
    ŷ = Σ w_i s_i / Σ w_i
    c = 1 - exp(-Σ w_i)   (0 if no points)
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    su = np.asarray(su, dtype=np.float64)
    sv = np.asarray(sv, dtype=np.float64)
    ss = np.asarray(ss, dtype=np.float64)
    if ss.size == 0:
        z = np.zeros_like(u, dtype=np.float64)
        return z, z
    w = _weights(u, v, su, sv, ell, k_cut=k_cut)
    wsum = np.sum(w, axis=-1)
    pred = np.sum(w * ss, axis=-1) / np.maximum(wsum, 1e-30)
    pred = np.where(wsum > 1e-12, pred, 0.0)
    conf = 1.0 - np.exp(-wsum)
    conf = np.where(wsum > 1e-12, conf, 0.0)
    return pred, conf


def disk_holes(
    su: np.ndarray,
    sv: np.ndarray,
    ss: np.ndarray,
    r_disk: float,
    ell: float,
    *,
    n_max: int = 12,
    r_step: float = 4.0,
    tau: float = TAU_ORANGE,
) -> list[tuple[float, float]]:
    """Gray (c<τ) cells inside the disk. Returns local (Δu,Δv) coordinates."""
    if int(n_max) <= 0 or su.size == 0:
        return []
    r_disk = float(r_disk)
    step = max(float(r_step), 1e-6)
    n_rad = max(int(round(r_disk / (0.5 * step))), 2)
    n_th = max(24, int(n_ang_to_cover(r_disk, r_step=step) * 2))
    rs = np.linspace(0.5 * step, r_disk, n_rad)
    ths = np.linspace(0.0, 2.0 * np.pi, n_th, endpoint=False)
    rr, tt = np.meshgrid(rs, ths)
    cu = (rr * np.cos(tt)).ravel()
    cv = (rr * np.sin(tt)).ravel()
    _p, conf = predict(cu, cv, su, sv, ss, ell)
    ok = (conf < float(tau)) & (np.hypot(cu, cv) <= r_disk + 1e-9)
    if not np.any(ok):
        return []
    cu, cv = cu[ok], cv[ok]
    ou = np.asarray(su, dtype=np.float64).copy()
    ov = np.asarray(sv, dtype=np.float64).copy()
    picked: list[tuple[float, float]] = []
    min_d2 = (0.45 * step) ** 2
    sep2 = (0.80 * step) ** 2
    for _ in range(int(n_max)):
        if cu.size == 0:
            break
        dmin = np.min((cu[:, None] - ou[None, :]) ** 2 + (cv[:, None] - ov[None, :]) ** 2, axis=1)
        j = int(np.argmax(dmin))
        if dmin[j] < min_d2:
            break
        picked.append((float(cu[j]), float(cv[j])))
        ou = np.append(ou, cu[j])
        ov = np.append(ov, cv[j])
        keep = (cu - cu[j]) ** 2 + (cv - cv[j]) ** 2 > sep2
        cu, cv = cu[keep], cv[keep]
    return picked


def _polar_grid(r_lim: float, r_step: float) -> tuple[np.ndarray, np.ndarray]:
    """Local (Δu, Δv) relative to the anchor."""
    r_lim = max(float(r_lim), 1e-6)
    step = max(float(r_step), 1e-6)
    n_rad = max(int(round(r_lim / (0.5 * step))), 3)
    n_th = max(32, int(n_ang_to_cover(r_lim, r_step=step) * 2))
    rs = np.linspace(0.5 * step, r_lim, n_rad)
    ths = np.linspace(0.0, 2.0 * np.pi, n_th, endpoint=False)
    rr, tt = np.meshgrid(rs, ths)
    return (rr * np.cos(tt)).ravel(), (rr * np.sin(tt)).ravel()


def _yw(w: np.ndarray, wsum: np.ndarray, ss: np.ndarray) -> np.ndarray:
    ss = np.asarray(ss, dtype=np.float64)
    pred = np.sum(w * ss, axis=-1) / np.maximum(wsum, 1e-30)
    return np.where(wsum > 1e-12, pred, 0.0)


def predict_parts(
    u: np.ndarray,
    v: np.ndarray,
    su: np.ndarray,
    sv: np.ndarray,
    ell: float,
    scores: np.ndarray,
    d_peak: np.ndarray | None = None,
    beam: np.ndarray | None = None,
    eta: np.ndarray | None = None,
    leave: np.ndarray | None = None,
    k_cut: float = K_CUT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Interpolate the three metrics separately. Combined ŷ = cone 60° + η_dB + total efficiency."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    su = np.asarray(su, dtype=np.float64)
    sv = np.asarray(sv, dtype=np.float64)
    if su.size == 0:
        z = np.zeros_like(u, dtype=np.float64)
        return z, z, None, None, None
    w = _weights(u, v, su, sv, ell, k_cut=k_cut)
    wsum = np.sum(w, axis=-1)
    conf = np.where(wsum > 1e-12, 1.0 - np.exp(-wsum), 0.0)
    yl = leave if leave is not None else d_peak
    yp = _yw(w, wsum, yl) if yl is not None and np.asarray(yl).size == su.size else None
    yb = _yw(w, wsum, beam) if beam is not None and np.asarray(beam).size == su.size else None
    ye = _yw(w, wsum, eta) if eta is not None and np.asarray(eta).size == su.size else None
    if yp is not None and yb is not None and ye is not None:
        yhat = yb + 10.0 * np.log10(np.clip(ye, 1e-15, 1.0)) + yp
    else:
        yhat = _yw(w, wsum, scores)
    return yhat, conf, yp, yb, ye


def orange_good_sites(
    su: np.ndarray,
    sv: np.ndarray,
    ell: float,
    champ_score: float,
    r_lim: float,
    *,
    scores: np.ndarray,
    d_peak: np.ndarray | None = None,
    beam: np.ndarray | None = None,
    eta: np.ndarray | None = None,
    leave: np.ndarray | None = None,
    n_max: int = 8,
    slack: float = 0.35,
    slack_eta: float = 0.02,
    r_step: float = 4.0,
    tau: float = TAU_ORANGE,
) -> list[tuple[float, float, float]]:
    """Orange ŷ candidates as local (Δu, Δv, ŷ). If n_max<=0, return everything that passes the filter (sep spacing only)."""
    if su.size == 0:
        return []
    cap = int(n_max)
    unlimited = cap <= 0
    step = max(float(r_step), 1e-6)
    cu, cv = _polar_grid(r_lim, step)
    yscore, conf = predict(cu, cv, su, sv, scores, ell)
    _yp, _conf2, yp, yb, ye = predict_parts(
        cu, cv, su, sv, ell, scores, d_peak=d_peak, beam=beam, eta=eta, leave=leave
    )
    d2 = np.min((cu[:, None] - su[None, :]) ** 2 + (cv[:, None] - sv[None, :]) ** 2, axis=1)
    orange = (
        (conf >= float(tau))
        & (np.hypot(cu, cv) <= float(r_lim) + 1e-9)
        & (d2 >= (0.45 * step) ** 2)
    )
    good = yscore >= float(champ_score) - float(slack)
    if yp is not None and d_peak is not None:
        fin = np.asarray(d_peak, dtype=np.float64)
        fin = fin[np.isfinite(fin)]
        if fin.size:
            good = good | (yp >= float(np.max(fin)) - float(slack))
    if yb is not None and beam is not None:
        fin = np.asarray(beam, dtype=np.float64)
        fin = fin[np.isfinite(fin)]
        if fin.size:
            good = good | (yb >= float(np.max(fin)) - float(slack))
    if ye is not None and eta is not None:
        fin = np.asarray(eta, dtype=np.float64)
        fin = fin[np.isfinite(fin)]
        if fin.size:
            good = good | (ye >= float(np.max(fin)) - float(slack_eta))
    ok = orange & good
    if not np.any(ok):
        return []
    cu, cv, yscore = cu[ok], cv[ok], yscore[ok]
    picked: list[tuple[float, float, float]] = []
    sep2 = (0.80 * step) ** 2
    while cu.size > 0:
        if not unlimited and len(picked) >= cap:
            break
        j = int(np.argmax(yscore))
        picked.append((float(cu[j]), float(cv[j]), float(yscore[j])))
        keep = (cu - cu[j]) ** 2 + (cv - cv[j]) ** 2 > sep2
        cu, cv, yscore = cu[keep], cv[keep], yscore[keep]
    return picked


def orange_score_sites(
    su: np.ndarray,
    sv: np.ndarray,
    ss: np.ndarray,
    ell: float,
    champ: float,
    r_lim: float,
    *,
    n_max: int = 8,
    slack: float = 0.35,
    r_step: float = 4.0,
    tau: float = TAU_ORANGE,
    d_peak: np.ndarray | None = None,
    beam: np.ndarray | None = None,
    eta: np.ndarray | None = None,
) -> list[tuple[float, float, float]]:
    return orange_good_sites(
        su,
        sv,
        ell,
        champ,
        r_lim,
        scores=ss,
        d_peak=d_peak,
        beam=beam,
        eta=eta,
        n_max=n_max,
        slack=slack,
        r_step=r_step,
        tau=tau,
    )


def inferred_combined_max(
    su: np.ndarray,
    sv: np.ndarray,
    ell: float,
    r_lim: float,
    scores: np.ndarray,
    *,
    d_peak: np.ndarray | None = None,
    beam: np.ndarray | None = None,
    eta: np.ndarray | None = None,
    leave: np.ndarray | None = None,
    r_step: float = 4.0,
    tau: float = TAU_ORANGE,
) -> tuple[float, float, float] | None:
    """Extrema of the combined ŷ over the orange area, as local (Δu, Δv, ŷ)."""
    if su.size == 0:
        return None
    cu, cv = _polar_grid(r_lim, r_step)
    yhat, conf, _yp, _yb, _ye = predict_parts(
        cu, cv, su, sv, ell, scores, d_peak=d_peak, beam=beam, eta=eta, leave=leave
    )
    ok = (conf >= float(tau)) & (np.hypot(cu, cv) <= float(r_lim) + 1e-9)
    if not np.any(ok):
        return None
    j = int(np.argmax(np.where(ok, yhat, -1e30)))
    return float(cu[j]), float(cv[j]), float(yhat[j])


NEIGH8 = (
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (1.0, 1.0),
    (1.0, -1.0),
    (-1.0, 1.0),
    (-1.0, -1.0),
)


def chart_neighbor_uv1(u0: int | float, v0: int | float) -> list[tuple[int, int]]:
    """Legacy flat-grid ±1 (8 cells). Exploration extrema use `chart_neighbor_rs8` with (r,s)±1."""
    u0, v0 = int(round(float(u0))), int(round(float(v0)))
    return [(u0 + int(du), v0 + int(dv)) for du, dv in NEIGH8]


def chart_neighbor8(
    u0: float,
    v0: float,
    step: float,
    su: np.ndarray | None = None,
    sv: np.ndarray | None = None,
    min_frac: float = 0.45,
) -> list[tuple[float, float]]:
    """8-direction neighbors in chart (u,v) at spacing step. Drops those already close to green."""
    step = max(float(step), 1e-6)
    min2 = (float(min_frac) * step) ** 2
    out: list[tuple[float, float]] = []
    su = np.asarray(su, dtype=np.float64) if su is not None else np.zeros(0)
    sv = np.asarray(sv, dtype=np.float64) if sv is not None else np.zeros(0)
    for du, dv in NEIGH8:
        u = float(u0) + du * step
        v = float(v0) + dv * step
        if su.size:
            if float(np.min((su - u) ** 2 + (sv - v) ** 2)) < min2:
                continue
        out.append((u, v))
    return out
