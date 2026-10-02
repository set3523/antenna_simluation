# -*- coding: utf-8 -*-
"""Per-metric trend lines. Branch when the same metric gets a new angle. Links are regrouped by the angle rule."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core import Individual, clone_ind, hamming, is_feasible


KEYS = ("beam", "eta", "leave")


@dataclass
class Root:
    rid: int
    key: str
    theta: float
    r: float
    tip: Individual
    best: float
    stall: int = 0
    alive: bool = True
    omega: float = 0.0
    speed: float = 1.0
    path: list[tuple[float, float]] = field(default_factory=list)


INERTIA = 0.70
SPEED_UP = 1.16
SPEED_DN = 0.84
SPEED_MIN = 0.55
SPEED_MAX = 2.2
HEADING_SEP = 0.52  # ≈30°. Same metric closer than this counts as the same direction


def wrap_ang(d: float) -> float:
    return float(np.arctan2(np.sin(d), np.cos(d)))


def heading_of(rt: Root) -> float:
    return float(rt.theta + rt.omega)


def cone_half(rt: Root, base: float = 0.52) -> float:
    """The faster it goes, the narrower the cone, to keep inertia."""
    return float(np.clip(base / (0.60 + 0.50 * rt.speed), 0.16, 0.62))


def same_key_headings(roots: list[Root], key: str) -> list[float]:
    return [heading_of(rt) for rt in roots if rt.key == key]


def is_new_heading(th: float, headings: list[float], sep: float = HEADING_SEP) -> bool:
    return all(abs(wrap_ang(th - h)) >= sep for h in headings)


def apply_steer(rt: Root, new_theta: float, improved: bool):
    turn = wrap_ang(new_theta - rt.theta)
    rt.omega = INERTIA * rt.omega + (1.0 - INERTIA) * turn
    if improved:
        rt.speed = float(np.clip(rt.speed * SPEED_UP, SPEED_MIN, SPEED_MAX))
    else:
        rt.speed = float(np.clip(rt.speed * SPEED_DN, SPEED_MIN, SPEED_MAX))
    rt.theta = float(new_theta)


def metric_val(ind: Individual, key: str) -> float:
    if key in ("leave", "leave_db") and getattr(ind, "leave_db", None) is not None:
        return float(ind.leave_db)
    if key in ("d_peak", "d_db") and getattr(ind, "d_peak", None) is not None:
        return float(ind.d_peak)
    if key == "d_ap" and getattr(ind, "d_ap", None) is not None:
        return float(ind.d_ap)
    if key in ("beam", "beam_db") and getattr(ind, "beam_db", None) is not None:
        return float(ind.beam_db)
    if key == "d_use" and ind.d_use is not None:
        return float(ind.d_use)
    if key == "eta" and ind.eta is not None:
        return float(ind.eta)
    if ind.score is not None:
        return float(ind.score)
    return -1e18


def metric_champs(pool: list[Individual], cfg=None) -> dict[str, Individual]:
    """Score #1 plus #1 per metric. FDTD-evaluated only."""
    scored = [
        p
        for p in pool
        if p.score is not None and (cfg is None or is_feasible(p, cfg))
    ]
    out: dict[str, Individual] = {}
    if not scored:
        return out
    out["score"] = max(scored, key=lambda x: x.score or -1e18)
    for key in KEYS:
        out[key] = max(scored, key=lambda x: metric_val(x, key))
    return out


def keep_metric_parents(
    pool: list[Individual],
    n: int,
    elite: Individual | None = None,
    min_hamming: int = 40,
    cfg=None,
) -> list[Individual]:
    """Elite (score) + metric tops + far-by-Hamming in score order."""
    scored = [
        p
        for p in pool
        if p.score is not None and (cfg is None or is_feasible(p, cfg))
    ]
    if not scored:
        return [clone_ind(p) for p in pool[: max(int(n), 0)]]
    if elite is None or elite.score is None:
        elite = max(scored, key=lambda x: x.score or -1e18)
    picked: list[Individual] = []

    def add(p: Individual | None) -> bool:
        if p is None or p.score is None or len(picked) >= n:
            return False
        if any(np.array_equal(p.metal, q.metal) for q in picked):
            return False
        picked.append(clone_ind(p))
        return True

    add(elite)
    champs = metric_champs(scored + [elite], cfg=cfg)
    for key in KEYS:
        add(champs.get(key))
    scored_s = sorted(scored, key=lambda x: x.score or -1e18, reverse=True)
    for p in scored_s:
        if len(picked) >= n:
            break
        if all(hamming(p.metal, q.metal) >= min_hamming for q in picked):
            add(p)
    for p in scored_s:
        if len(picked) >= n:
            break
        add(p)
    return picked


def elite_metric_deficits(
    elite: Individual, pool: list[Individual]
) -> tuple[bool, bool, float, float]:
    """Whether the score #1 cone (beam_db) or η falls below the pool best."""
    scored = [p for p in pool if p.score is not None]
    if not scored:
        return False, False, 0.0, 0.0
    ch = metric_champs(scored)
    cone_e = metric_val(elite, "beam")
    eta_e = metric_val(elite, "eta")
    cone_best = metric_val(ch["beam"], "beam") if "beam" in ch else cone_e
    eta_best = metric_val(ch["eta"], "eta") if "eta" in ch else eta_e
    cone_low = cone_e < cone_best - 0.25
    eta_low = eta_e < eta_best - 0.005
    return cone_low, eta_low, cone_best - cone_e, eta_best - eta_e


def pick_breed_pair(
    pool: list[Individual],
    rng,
    elite: Individual | None = None,
    min_hamming: int = 40,
) -> tuple[Individual, Individual] | None:
    """
    If the score #1 lacks cone or η, cross that metric's #1s, or score #1 x metric #1.
    Otherwise pick_metric_pair.
    """
    scored = [p for p in pool if p.score is not None]
    if not scored:
        return None
    if elite is None or elite.score is None:
        elite = max(scored, key=lambda x: x.score or -1e18)
    ch = metric_champs(scored)
    cone_low, eta_low, _, _ = elite_metric_deficits(elite, scored)

    def ok(a: Individual | None, b: Individual | None) -> bool:
        return (
            a is not None
            and b is not None
            and a.score is not None
            and b.score is not None
            and not np.array_equal(a.metal, b.metal)
        )

    beam_p = ch.get("beam")
    eta_p = ch.get("eta")
    score_p = ch.get("score", elite)

    if cone_low and eta_low and ok(beam_p, eta_p):
        return clone_ind(beam_p), clone_ind(eta_p)
    if cone_low and ok(score_p, beam_p):
        return clone_ind(score_p), clone_ind(beam_p)
    if eta_low and ok(score_p, eta_p):
        return clone_ind(score_p), clone_ind(eta_p)
    if (cone_low or eta_low) and ok(beam_p, eta_p):
        return clone_ind(beam_p), clone_ind(eta_p)
    return pick_metric_pair(scored, rng, min_hamming=min_hamming)


def pick_metric_pair(
    pool: list[Individual],
    rng,
    min_hamming: int = 40,
) -> tuple[Individual, Individual] | None:
    """Cross tops of different metrics. If one individual dominates, pick a partner far by Hamming."""
    scored = [p for p in pool if p.score is not None]
    if len(scored) < 2:
        return None
    champs = metric_champs(scored)
    keys = [k for k in KEYS if k in champs]
    rng.shuffle(keys)
    for i, ka in enumerate(keys):
        a = champs[ka]
        for kb in keys[i + 1 :] + keys[:i]:
            b = champs[kb]
            if not np.array_equal(a.metal, b.metal):
                return a, b
    a = champs.get("score") or scored[0]
    others = [p for p in scored if not np.array_equal(p.metal, a.metal)]
    if not others:
        return None
    far = [p for p in others if hamming(p.metal, a.metal) >= min_hamming]
    pool_b = far or others
    return a, max(pool_b, key=lambda p: p.score or -1e18)


def _uv_theta(ind: Individual, anchor: tuple[int, int] | None = None) -> float | None:
    uv = getattr(ind, "chart_uv", None)
    if uv is None:
        return None
    au, av = (0.0, 0.0) if anchor is None else (float(anchor[0]), float(anchor[1]))
    du, dv = float(uv[0]) - au, float(uv[1]) - av
    if du == 0 and dv == 0:
        return None
    return float(np.arctan2(dv, du))


def label_disk_points(peaks: list[Individual], chart=None, n_sec: int = 8):
    """Mark at most 2 good angles per metric. Only points at those angles form lines."""
    pool = [p for p in peaks if p.score is not None and getattr(p, "chart_uv", None)]
    if len(pool) < 3:
        return
    au = av = 0
    if chart is not None and hasattr(chart, "origin_uv"):
        au, av = chart.origin_uv()
    for p in pool:
        p.chart_kind = "seed"
        if chart is not None and hasattr(chart, "mark_kind"):
            chart.mark_kind(p.metal, "seed")

    def sector(p: Individual) -> int:
        u, v = p.chart_uv
        th = float(np.arctan2(float(v) - av, float(u) - au))
        return int(np.floor((th + np.pi) / (2.0 * np.pi) * n_sec)) % n_sec

    claimed: dict[int, tuple[str, float]] = {}
    for key in KEYS:
        vals = np.array([metric_val(p, key) for p in pool], dtype=np.float64)
        med = float(np.median(vals))
        best: dict[int, tuple[float, Individual]] = {}
        for p, val in zip(pool, vals):
            if val < med:
                continue
            sec = sector(p)
            if sec not in best or val > best[sec][0]:
                best[sec] = (float(val), p)
        tops = sorted(best.items(), key=lambda kv: -kv[1][0])[:2]
        win = {sec for sec, _ in tops}
        for p, val in zip(pool, vals):
            if sector(p) not in win or val < med:
                continue
            i = id(p)
            if i not in claimed or val > claimed[i][1]:
                claimed[i] = (key, float(val))
    for p in pool:
        got = claimed.get(id(p))
        if got is None:
            continue
        p.chart_kind = got[0]
        if chart is not None and hasattr(chart, "mark_kind"):
            chart.mark_kind(p.metal, got[0])


def start_from_chains(
    chains,
    peaks: list[Individual],
    chart,
    max_roots: int,
) -> list[Root]:
    """Turn angle, r, score lines found on the disk into trend lines. Not limited to one angle per metric."""
    if chart is None or not chains:
        return []
    from circle.index_map import uv_local

    u, v, _, rr = chart.points()
    rows = (
        chart.point_rows()
        if hasattr(chart, "point_rows")
        else np.arange(u.size, dtype=np.intp)
    )
    metals = getattr(chart, "metals", [])
    uvs = getattr(chart, "uvs", [])
    ou, ov = chart.origin_uv() if hasattr(chart, "origin_uv") else (0, 0)
    roots: list[Root] = []
    used_h: dict[str, list[float]] = {k: [] for k in KEYS}

    def match(pi: int) -> Individual | None:
        if pi < 0 or pi >= int(rows.size):
            return None
        ri = int(rows[pi])
        metal = metals[ri] if ri < len(metals) else None
        uv = uvs[ri] if ri < len(uvs) else None
        if metal is not None:
            raw = np.asarray(metal, dtype=np.uint8).tobytes()
            for p in peaks:
                if p.score is None:
                    continue
                if np.asarray(p.metal, dtype=np.uint8).tobytes() == raw:
                    return p
        if uv is None:
            return None
        best = None
        bd = 1e18
        for p in peaks:
            if p.score is None or p.chart_uv is None:
                continue
            du = float(p.chart_uv[0]) - float(uv[0])
            dv = float(p.chart_uv[1]) - float(uv[1])
            d = float(np.hypot(du, dv))
            if d < bd:
                bd, best = d, p
        return best

    def nearest_peak_local(lu: float, lv: float) -> Individual | None:
        best = None
        bd = 1e18
        for p in peaks:
            if p.score is None or p.chart_uv is None:
                continue
            plu, plv = uv_local(p.chart_uv[0], p.chart_uv[1], (ou, ov))
            d = float(np.hypot(plu - lu, plv - lv))
            if d < bd:
                bd, best = d, p
        return best

    ranked = sorted(chains, key=lambda c: (-len(c.idxs), -float(getattr(c, "r_tip", 0.0))))
    next_id = 1
    for ch in ranked:
        if len(roots) >= max_roots:
            continue
        th = float(getattr(ch, "theta0", 0.0))
        if not is_new_heading(th, used_h[ch.key]):
            continue
        peak = None
        tip_local: tuple[float, float] | None = None
        r_root = float(getattr(ch, "r_tip", 0.0) or 0.0)
        if getattr(ch, "mode", "radial") == "chord":
            tip_local = (float(ch.tipu), float(ch.tipv))
        tip_i = 0
        if ch.idxs:
            tip_i = int(max(ch.idxs, key=lambda i: float(rr[i]) if i < rr.size else -1.0))
            if tip_i < u.size:
                peak = match(tip_i)
                tip_local = (float(u[tip_i]), float(v[tip_i]))
                r_root = float(rr[tip_i])
        if peak is None and tip_local is not None:
            peak = nearest_peak_local(tip_local[0], tip_local[1])
        if peak is None:
            r_tip = float(getattr(ch, "r_tip", 0.0) or 0.0)
            om = float(getattr(ch, "omega", 0.0) or 0.0)
            r0 = float(getattr(ch, "r0", 4.0) or 4.0)
            th_t = th + om * (r_tip - r0)
            tip_local = (r_tip * float(np.cos(th_t)), r_tip * float(np.sin(th_t)))
            r_root = r_tip
            peak = nearest_peak_local(tip_local[0], tip_local[1])
        if peak is None:
            continue
        used_h[ch.key].append(th)
        tip = clone_ind(peak)
        tip.chart_kind = ch.key
        tip.chart_root = next_id
        path: list[tuple[float, float]] = []
        for key_p, rid_p, poly in getattr(chart, "root_paths", []) or []:
            if int(rid_p) == int(getattr(ch, "rid", 0)) and key_p == ch.key and len(poly) >= 2:
                path = [(float(a), float(b)) for a, b in poly]
                break
        if len(path) < 2 and tip_local is not None:
            path = [tip_local]
        roots.append(
            Root(
                rid=next_id,
                key=ch.key,
                theta=th,
                r=r_root,
                tip=tip,
                best=metric_val(peak, ch.key),
                omega=float(getattr(ch, "dth", 0.0) or 0.0),
                path=path,
            )
        )
        next_id += 1
    return roots


def start_from_ring(
    peaks: list[Individual],
    r_seed: float,
    max_roots: int,
    anchor: tuple[float, float] | None = None,
) -> list[Root]:
    """Best distinct angles per metric on the seed circle → initial trend lines."""
    ring = []
    for p in peaks:
        if p.score is None:
            continue
        uv = getattr(p, "chart_uv", None)
        if uv is None:
            continue
        from circle.index_map import chart_dist

        if chart_dist(uv, anchor or (0, 0)) < 3.0:
            continue
        ring.append(p)
    roots: list[Root] = []
    next_id = 1
    for key in KEYS:
        if len(roots) >= max_roots or not ring:
            break
        best = max(ring, key=lambda x: metric_val(x, key))
        th = _uv_theta(best, anchor)
        if th is None:
            continue
        uv = best.chart_uv
        from circle.index_map import uv_local

        au, av = (0.0, 0.0) if anchor is None else (float(anchor[0]), float(anchor[1]))
        plu, plv = uv_local(int(uv[0]), int(uv[1]), (au, av))
        tip = clone_ind(best)
        tip.chart_kind = key
        tip.chart_root = next_id
        roots.append(
            Root(
                rid=next_id,
                key=key,
                theta=th,
                r=float(r_seed),
                tip=tip,
                best=metric_val(best, key),
                path=[(0.0, 0.0), (plu, plv)],
            )
        )
        next_id += 1
    if not roots and ring:
        from circle.index_map import uv_local

        best = max(ring, key=lambda x: x.score or -1e18)
        th = _uv_theta(best, anchor) or 0.0
        tip = clone_ind(best)
        tip.chart_kind = "leave"
        tip.chart_root = 1
        uv = best.chart_uv
        if uv is not None:
            au, av = (0.0, 0.0) if anchor is None else (float(anchor[0]), float(anchor[1]))
            plu, plv = uv_local(int(uv[0]), int(uv[1]), (au, av))
            path_end = (plu, plv)
        else:
            path_end = (float(r_seed), 0.0)
        roots.append(
            Root(
                rid=1,
                key="leave",
                theta=th,
                r=float(r_seed),
                tip=tip,
                best=metric_val(best, "leave"),
                path=[(0.0, 0.0), path_end],
            )
        )
    return roots


def rematch_live_roots_to_chains(
    roots: list[Root],
    chains,
    *,
    min_gain: float = 0.12,
    log=None,
) -> int:
    """If the ŷ ridge is stronger, switch the live root's key, θ and ω."""
    switched = 0
    for rt in [x for x in roots if x.alive]:
        cur_q = 0.0
        for ch in chains:
            if ch.key != rt.key:
                continue
            dh = abs(wrap_ang(float(ch.theta0) - heading_of(rt)))
            if dh < 0.55:
                cur_q = max(cur_q, float(getattr(ch, "qual", len(ch.idxs))))
        best_ch = None
        best_q = -1e18
        h = heading_of(rt)
        for ch in chains:
            q = float(getattr(ch, "qual", len(ch.idxs)))
            if q <= 0.0:
                continue
            dh = abs(wrap_ang(float(ch.theta0) - h))
            if dh > 0.55:
                continue
            bonus = 1.0 if ch.key == rt.key else 1.06
            sc = q * bonus
            if sc > best_q:
                best_q, best_ch = sc, ch
        if best_ch is None:
            continue
        if cur_q > 0 and best_q < cur_q * (1.0 + float(min_gain)):
            continue
        if (
            best_ch.key == rt.key
            and abs(wrap_ang(float(best_ch.theta0) - rt.theta)) < 0.12
        ):
            continue
        old_k = rt.key
        rt.key = str(best_ch.key)
        rt.theta = float(best_ch.theta0)
        rt.omega = float(getattr(best_ch, "omega", 0.0))
        rt.stall = 0
        switched += 1
        if log is not None:
            log(
                f"    root{rt.rid} ridge→{rt.key} (was {old_k})  "
                f"θ={rt.theta:.2f}  qual {cur_q:.2f}→{best_q:.2f}"
            )
    return switched
