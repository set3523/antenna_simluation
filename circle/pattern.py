# -*- coding: utf-8 -*-
"""Angle and r trends. Finds lines like 0° r1 → 30° r3 → 60° r7 and predicts the next cell."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from circle.acquire import ang_diff
from circle.roots import HEADING_SEP, KEYS


@dataclass
class Chain:
    key: str
    idxs: list[int]
    dth: float
    rho: float
    resid: float
    rid: int
    theta0: float = 0.0
    omega: float = 0.0
    r0: float = 4.0
    r_tip: float = 0.0
    qual: float = 0.0
    mode: str = "radial"
    dirx: float = 0.0
    diry: float = 0.0
    tipu: float = 0.0
    tipv: float = 0.0

    def predict_next(
        self,
        u: np.ndarray,
        v: np.ndarray,
        step: float = 4.0,
    ) -> tuple[float, float] | None:
        """Step along the tangent from the end of a direct FDTD line. Does not pass through the origin."""
        if not self.idxs:
            return None
        idxs = self.idxs
        if len(idxs) >= 2:
            a, b = int(idxs[-2]), int(idxs[-1])
            du = float(u[b] - u[a])
            dv = float(v[b] - v[a])
            ln = float(np.hypot(du, dv))
            if ln > 1e-9:
                s = float(step) / ln
                return float(u[b] + du * s), float(v[b] + dv * s)
        b = int(idxs[-1])
        r = float(np.hypot(float(u[b]), float(v[b])))
        if r < 1e-9:
            return float(u[b]), float(v[b])
        st = float(step) / r
        return float(u[b] + u[b] * st), float(v[b] + v[b] * st)

    def predict(self, th: np.ndarray, rr: np.ndarray) -> tuple[float, float] | None:
        _ = th, rr
        return None


def _wrap(d: float) -> float:
    return float(np.arctan2(np.sin(d), np.cos(d)))


TURN_EPS = 0.08  # ≈4.6°. Below this, straight
TURN_MAX = 0.70  # ≈40°. Turning more than this in one step is not a trend


def _turn_sign(d: float, eps: float = TURN_EPS) -> int:
    if abs(d) < eps:
        return 0
    return 1 if d > 0.0 else -1


def _same_way(prev: float, nxt: float, eps: float = TURN_EPS) -> bool:
    a, b = _turn_sign(prev, eps), _turn_sign(nxt, eps)
    if a == 0 or b == 0:
        return True
    return a == b


def _grow(
    i: int,
    j: int,
    th: np.ndarray,
    rr: np.ndarray,
    pool: list[int],
) -> tuple[list[int], float, float, float] | None:
    """Curves one way only. Cut at an inflection (reversed turn)."""
    if rr[j] <= rr[i] * 1.08:
        return None
    dth = _wrap(float(th[j] - th[i]))
    if abs(dth) > TURN_MAX:
        return None
    rho = float(rr[j] / max(rr[i], 1e-6))
    if rho < 1.08 or rho > 4.8:
        return None
    dr = float(rr[j] - rr[i])
    b = float(rr[j] - 2.0 * rr[i])
    chain = [i, j]
    used = {i, j}
    resid = 0.0
    n_err = 0
    way = _turn_sign(dth)
    while True:
        last = chain[-1]
        pred_th = float(th[last] + dth)
        r_opts = (
            float(rr[last] * rho),
            float(rr[last] + dr),
            float(2.0 * rr[last] + b),
        )
        best_k = None
        best_c = 1e9
        best_d = 0.0
        for k in pool:
            if k in used or rr[k] <= rr[last] * 1.04:
                continue
            d_new = _wrap(float(th[k] - th[last]))
            if abs(d_new) > TURN_MAX:
                continue
            if not _same_way(dth if way != 0 else d_new, d_new):
                continue
            e_th = ang_diff(float(th[k]), pred_th)
            e_r = min(abs(float(rr[k]) - p) / max(p, 1e-6) for p in r_opts)
            cost = e_th + 0.85 * e_r
            if cost < best_c:
                best_c = cost
                best_k = k
                best_d = d_new
        if best_k is None or best_c > 0.55:
            break
        chain.append(best_k)
        used.add(best_k)
        resid += best_c
        n_err += 1
        if way == 0:
            way = _turn_sign(best_d)
        if way != 0:
            dth = 0.70 * dth + 0.30 * best_d
    if n_err:
        resid /= n_err
    return chain, dth, rho, resid


def _split_one_way(
    th: np.ndarray, idxs: list[int]
) -> list[list[int]]:
    """Split at the inflection into two trend lines. Each piece keeps a single turn sign."""
    if len(idxs) < 3:
        return [idxs] if len(idxs) >= 2 else []
    parts: list[list[int]] = []
    cur = [idxs[0], idxs[1]]
    way = _turn_sign(_wrap(float(th[idxs[1]] - th[idxs[0]])))
    for k in range(2, len(idxs)):
        d = _wrap(float(th[idxs[k]] - th[idxs[k - 1]]))
        s = _turn_sign(d)
        if s != 0 and way != 0 and s != way:
            if len(cur) >= 2:
                parts.append(cur)
            cur = [idxs[k - 1], idxs[k]]
            way = s
            continue
        cur.append(idxs[k])
        if way == 0 and s != 0:
            way = s
    if len(cur) >= 2:
        parts.append(cur)
    return parts


def _subsample_pool(idxs: list[int], rr: np.ndarray, cap: int) -> list[int]:
    """Prevent fit_chains O(n²) blowup: sample evenly in r order."""
    if len(idxs) <= cap:
        return idxs
    order = sorted(idxs, key=lambda i: float(rr[i]))
    k = len(order)
    cap = max(int(cap), 2)
    picks = [order[int(round(j * (k - 1) / max(cap - 1, 1)))] for j in range(cap)]
    return list(dict.fromkeys(picks))


def fit_chains(
    th: np.ndarray,
    rr: np.ndarray,
    idxs: list[int],
    *,
    max_pool: int = 96,
) -> list[tuple[list[int], float, float, float]]:
    if len(idxs) < 2:
        return []
    idxs = _subsample_pool(idxs, rr, max_pool)
    cands: list[tuple[list[int], float, float, float]] = []
    for a in range(len(idxs)):
        for b in range(len(idxs)):
            if a == b:
                continue
            i, j = idxs[a], idxs[b]
            got = _grow(i, j, th, rr, idxs)
            if got is None or len(got[0]) < 2:
                continue
            cands.append(got)
    split: list[tuple[list[int], float, float, float]] = []
    for ch, dth, rho, resid in cands:
        for part in _split_one_way(th, ch):
            if len(part) < 2:
                continue
            d0 = _wrap(float(th[part[1]] - th[part[0]]))
            split.append((part, d0, rho, resid))
    split.sort(key=lambda x: (-len(x[0]), x[3]))
    picked: list[tuple[list[int], float, float, float]] = []
    taken: set[int] = set()
    for ch, dth, rho, resid in split:
        if any(k in taken for k in ch):
            continue
        picked.append((ch, dth, rho, resid))
        taken.update(ch)
    return picked


def _metric_profile(t: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """That metric only, vs r: constant slope, no inflection. (qual, slope) or None."""
    ok = np.isfinite(y)
    if int(np.sum(ok)) < 3:
        return None
    t_use = t[ok]
    y_use = y[ok]
    dy = np.diff(y_use)
    strong = np.abs(dy) > 0.05 * (float(np.std(y_use)) + 0.25)
    if int(np.sum(strong)) >= 2:
        sg = np.sign(dy[strong])
        if float(np.min(sg) * np.max(sg)) < 0.0:
            return None
    A = np.column_stack([t_use, np.ones(t_use.size)])
    coef, *_ = np.linalg.lstsq(A, y_use, rcond=None)
    slope, a0 = float(coef[0]), float(coef[1])
    fit = a0 + slope * t_use
    resid = float(np.mean((fit - y_use) ** 2))
    span = float(np.std(y_use)) + 1e-6
    if resid > 0.50 * span * span:
        return None
    return abs(slope) * float(t_use.size) / (1.0 + resid), slope


def _add_chord_rays(
    key: str,
    ss: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    n: int,
    r_disk: float,
    r_step: float,
    ell: float,
    sep: float,
    new_roots: list[int],
    paths: list,
    trends: list,
    chains: list[Chain],
    next_rid: int,
) -> int:
    """Straight trend that need not pass through the center. Any direction on the disk."""
    from learn.infer import TAU_ORANGE, predict

    phis = np.linspace(0.0, np.pi, 16, endpoint=False)
    offs = (-0.55, -0.38, 0.38, 0.55)
    raw: list[tuple[float, float, float, float, float, np.ndarray, np.ndarray]] = []
    for phi in phis:
        dx, dy = float(np.cos(phi)), float(np.sin(phi))
        nx, ny = -dy, dx
        for ocoef in offs:
            off = float(ocoef) * r_disk
            half = float(np.sqrt(max(r_disk * r_disk - off * off, 0.0)))
            if half < 2.0 * r_step:
                continue
            ts = np.arange(-half, half + 0.51 * r_step, r_step, dtype=np.float64)
            ys = []
            xs = []
            zs = []
            for t in ts:
                x = off * nx + float(t) * dx
                y = off * ny + float(t) * dy
                pr, cf = predict(np.array([x]), np.array([y]), u, v, ss, ell)
                ys.append(float(pr[0]) if float(cf[0]) >= 0.5 * TAU_ORANGE else np.nan)
                xs.append(x)
                zs.append(y)
            got = _metric_profile(ts, np.asarray(ys, dtype=np.float64))
            if got is None:
                continue
            qual, slope = got
            raw.append((qual, phi, off, dx, dy, np.asarray(xs), np.asarray(zs)))
    raw.sort(key=lambda r: -r[0])
    kept: list[tuple[float, float]] = []
    for qual, phi, off, dx, dy, xs, zs in raw:
        if any(
            abs(_wrap(phi - p0)) < sep and abs(off - o0) < 0.22 * r_disk
            for p0, o0 in kept
        ):
            continue
        kept.append((phi, off))
        nx, ny = -dy, dx
        idxs: list[int] = []
        for i in range(n):
            px, py = float(u[i]), float(v[i])
            if abs(px * nx + py * ny - off) > 2.4:
                continue
            t = (px - off * nx) * dx + (py - off * ny) * dy
            if abs(t) > r_disk + r_step:
                continue
            idxs.append(i)
            new_roots[i] = next_rid
        order = np.argsort(xs * dx + zs * dy)
        poly = [(float(xs[k]), float(zs[k])) for k in order]
        if len(idxs) >= 2:
            gxy = sorted(
                [(float(u[i]), float(v[i])) for i in idxs],
                key=lambda p: p[0] * dx + p[1] * dy,
            )
            poly = gxy
        if len(poly) < 2:
            continue
        # One gray cell toward increasing score
        t_end = poly[-1][0] * dx + poly[-1][1] * dy
        t_beg = poly[0][0] * dx + poly[0][1] * dy
        step = r_step if t_end >= t_beg else -r_step
        xu, xv = poly[-1] if step > 0 else poly[0]
        for _ in range(10):
            xu = xu + step * dx
            xv = xv + step * dy
            _pr, cf = predict(np.array([xu]), np.array([xv]), u, v, ss, ell)
            if float(cf[0]) < TAU_ORANGE:
                break
        trends.append((next_rid, xu, xv, key))
        paths.append((key, next_rid, poly))
        chains.append(
            Chain(
                key=key,
                idxs=idxs,
                dth=0.0,
                rho=1.0,
                resid=0.0,
                rid=next_rid,
                theta0=float(np.arctan2(dy, dx)),
                omega=0.0,
                r0=r_step,
                r_tip=float(np.hypot(xu, xv)),
                mode="chord",
                dirx=float(np.sign(step) * dx),
                diry=float(np.sign(step) * dy),
                tipu=float(xu),
                tipv=float(xv),
            )
        )
        next_rid += 1
    return next_rid


def reconnect_by_pattern(chart, sep: float = HEADING_SEP) -> list[Chain]:
    """Explored FDTD points: straight or one-way turning lines where only the key metric follows the trend."""
    return find_field_rays(chart, sep=sep)


def _chain_omega(th: np.ndarray, rr: np.ndarray, idxs: list[int]) -> float:
    om: list[float] = []
    for a, b in zip(idxs[:-1], idxs[1:]):
        dr = float(rr[b] - rr[a])
        if dr < 1e-6:
            continue
        om.append(_wrap(float(th[b] - th[a])) / dr)
    if not om:
        return 0.0
    return float(np.median(om))


def _monotonic_metric(y: np.ndarray, tol: float = 0.22) -> bool:
    """Metric changes in one direction only (r order for both straight and one-way turning lines)."""
    if y.size < 2:
        return False
    dy = np.diff(y)
    scale = float(np.std(y)) + 0.20
    strong = np.abs(dy) > tol * scale
    if int(np.sum(strong)) < 1:
        return True
    sg = np.sign(dy[strong])
    return float(np.min(sg) * np.max(sg)) >= 0.0


def _chain_passes_metric(r: np.ndarray, m: np.ndarray) -> bool:
    """Not the overall score; only when this key's metric matches the r (or line order) trend."""
    if r.size < 2 or not np.all(np.isfinite(m)):
        return False
    if r.size == 2:
        return float(m[1]) >= float(m[0]) - 0.35 * (abs(float(m[0])) + 0.15)
    if _metric_profile(r, m) is not None:
        return True
    order = np.argsort(r)
    return _monotonic_metric(m[order])


def find_field_rays(
    chart,
    sep: float = HEADING_SEP,
    n_head: int = 36,
) -> list[Chain]:
    _ = n_head
    u, v, _, rr = chart.points()
    n = int(u.size)
    th = np.arctan2(v, u)
    chains: list[Chain] = []
    if n == 0:
        chart.root_paths = []
        chart.trends = []
        chart.inferred = []
        return chains

    r_step = 4.0
    new_roots = [0] * n
    paths: list[tuple[str, int, list[tuple[float, float]]]] = []
    trends: list[tuple[int, float, float, float, float, str]] = []
    next_rid = 1

    for key in KEYS:
        ss = chart.series_at_points(key)
        pool = [
            i
            for i in range(n)
            if float(rr[i]) > 0.5 and i < ss.size and np.isfinite(ss[i])
        ]
        if len(pool) < 2:
            continue
        picked = fit_chains(th, rr, pool)
        picked_headings: list[float] = []
        for ch, dth, rho, resid in sorted(picked, key=lambda x: (-len(x[0]), x[3])):
            if len(ch) < 2:
                continue
            order = np.argsort([float(rr[i]) for i in ch])
            ch_ord = [ch[int(j)] for j in order]
            r_sorted = np.asarray([float(rr[i]) for i in ch_ord], dtype=np.float64)
            s_sorted = np.asarray([float(ss[i]) for i in ch_ord], dtype=np.float64)
            if not _chain_passes_metric(r_sorted, s_sorted):
                continue
            th0 = float(th[ch_ord[0]])
            if any(abs(_wrap(th0 - t)) < sep for t in picked_headings):
                continue
            picked_headings.append(th0)
            for i in ch_ord:
                new_roots[i] = next_rid
            path = [(float(u[i]), float(v[i])) for i in ch_ord]
            tmp = Chain(key=key, idxs=ch_ord, dth=0.0, rho=1.0, resid=0.0, rid=0)
            tip_u = float(u[ch_ord[-1]])
            tip_v = float(v[ch_ord[-1]])
            nxt = tmp.predict_next(u, v, step=r_step)
            if nxt is not None:
                trends.append(
                    (
                        next_rid,
                        tip_u,
                        tip_v,
                        float(nxt[0]),
                        float(nxt[1]),
                        key,
                    )
                )
            prof = _metric_profile(r_sorted, s_sorted)
            qual = float(prof[0]) if prof is not None else float(len(ch_ord))
            omega = _chain_omega(th, rr, ch_ord)
            chains.append(
                Chain(
                    key=key,
                    idxs=ch_ord,
                    dth=float(dth),
                    rho=float(rho),
                    resid=float(resid),
                    rid=next_rid,
                    theta0=th0,
                    omega=omega,
                    r0=float(rr[ch_ord[0]]),
                    r_tip=float(rr[ch_ord[-1]]),
                    qual=qual,
                    mode="radial",
                    tipu=float(u[ch_ord[-1]]),
                    tipv=float(v[ch_ord[-1]]),
                )
            )
            paths.append((key, next_rid, path))
            next_rid += 1

    chart.roots = new_roots
    chart.trends = trends
    chart.root_paths = paths
    chart.inferred = []
    return chains


def _seg_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx, vy = bx - ax, by - ay
    t = ((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy + 1e-12)
    t = float(np.clip(t, 0.0, 1.0))
    return float(np.hypot(px - (ax + t * vx), py - (ay + t * vy)))


def _dir_fit(px: float, py: float, pts: list[tuple[float, float]]) -> tuple[bool, float]:
    """Angle and direction trend. (match, estimated step n)."""
    if not pts:
        return False, 0.0
    pth = float(np.arctan2(py, px))
    pr = float(np.hypot(px, py))
    heads = [float(np.arctan2(y, x)) for x, y in pts]
    rs = [float(np.hypot(x, y)) for x, y in pts]
    if len(pts) == 1:
        if ang_diff(pth, heads[0]) < HEADING_SEP:
            return True, max(pr / max(rs[0], 1e-6) - 1.0, 0.0)
        return False, 0.0
    dths = [_wrap(heads[i + 1] - heads[i]) for i in range(len(heads) - 1)]
    rhos = [rs[i + 1] / max(rs[i], 1e-6) for i in range(len(rs) - 1)]
    dth = float(np.median(dths))
    rho = float(np.clip(np.median(rhos), 1.05, 4.8))
    if ang_diff(pth, heads[-1]) < HEADING_SEP:
        n = np.log(max(pr, 1e-6) / max(rs[-1], 1e-6)) / np.log(rho) if rho > 1.02 else 0.0
        return True, float(n)
    for n in (0.5, 1.0, 2.0, 3.0):
        if ang_diff(pth, heads[-1] + n * dth) >= 0.40:
            continue
        pred_r = rs[-1] * (rho**n)
        if pred_r > 0.8 and abs(pr - pred_r) / pred_r < 0.45:
            return True, float(n)
    return False, 0.0


def _trend_fit(
    px: float,
    py: float,
    s_hat: float,
    pts: list[tuple[float, float, float]],
) -> bool:
    """Link orange (inferred) points only when angle, direction and score trends all match."""
    if not pts:
        return False
    ok, n = _dir_fit(px, py, [(x, y) for x, y, _s in pts])
    if not ok:
        return False
    scores = [float(s) for _x, _y, s in pts]
    if len(scores) == 1:
        span = abs(scores[0]) + 0.75
        return abs(s_hat - scores[0]) < 0.50 * span or s_hat >= scores[0] - 0.20
    dss = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]
    ds = float(np.median(dss))
    s_exp = scores[-1] + ds * max(n, 0.0)
    span = max(abs(ds), abs(scores[-1] - scores[0]) / max(len(scores) - 1, 1), 0.40)
    return abs(s_hat - s_exp) <= 0.60 * span


def attach_orange_nodes(chart, ell: float | None = None):
    """From orange (inferred) points, link only those matching angle, direction and score trends into virtual trend lines."""
    from learn.infer import TAU_ORANGE, length_scale, predict

    u, v, _, sr = chart.points()
    if u.size == 0:
        chart.inferred = []
        return
    kinds = chart.kinds_at_points()
    rids = chart.roots_at_points()
    r_disk = float(getattr(chart, "r_probed", 0.0) or 0.0)
    r_seed = float(getattr(chart, "r_seed", 8.0) or 8.0)
    if ell is None:
        ell = length_scale(u, v, r_step=4.0, r_seed=r_seed)
    r_zoom = max(r_disk, r_seed, 4.0) * 1.35
    nbin = 41
    axis = np.linspace(-r_zoom, r_zoom, nbin)
    uu, vv = np.meshgrid(axis, axis)
    inferred: list[tuple[float, float, str, int]] = []
    extra_trends: list[tuple[int, float, float, str]] = list(getattr(chart, "trends", []) or [])

    for key in KEYS:
        ss = chart.series_at_points(key)
        pred, conf = predict(uu, vv, u, v, ss, ell)
        orange = conf >= TAU_ORANGE
        sel = kinds == key
        pu = u[sel]
        pv = v[sel]
        pr = rids[sel] if rids.size == u.size else np.zeros(pu.size, dtype=np.int32)
        if pu.size == 0:
            continue
        by_rid: dict[int, list[tuple[float, float, float]]] = {}
        for x, y, rid, sv_ in zip(pu, pv, pr, ss[sel]):
            by_rid.setdefault(int(rid), []).append((float(x), float(y), float(sv_)))
        for rid, pts in by_rid.items():
            pts = sorted(pts, key=lambda p: float(np.hypot(p[0], p[1])))
            hits: list[tuple[float, float]] = []
            for i in range(nbin):
                for j in range(nbin):
                    if not orange[i, j]:
                        continue
                    px, py = float(uu[i, j]), float(vv[i, j])
                    if float(np.min(np.hypot(px - pu, py - pv))) < 0.40 * ell:
                        continue
                    if _trend_fit(px, py, float(pred[i, j]), pts):
                        hits.append((px, py))
            hits.sort(key=lambda p: float(np.hypot(p[0], p[1])))
            kept: list[tuple[float, float]] = []
            for p in hits:
                if kept and float(np.hypot(p[0] - kept[-1][0], p[1] - kept[-1][1])) < 0.50 * ell:
                    continue
                kept.append(p)
                inferred.append((p[0], p[1], key, int(rid)))

    chart.inferred = inferred
    chart.trends = extra_trends


def breaks_score_trend(
    chart, key: str, rid: int, uv: tuple[float, float], s_hat: float
) -> bool:
    """True if the point just computed does not match that trend line's (key metric) trend."""
    u, v, _, _ = chart.points()
    if u.size == 0:
        return False
    kinds = chart.kinds_at_points()
    rids = chart.roots_at_points()
    ss = chart.series_at_points(key)
    pts: list[tuple[float, float, float]] = []
    from circle.index_map import uv_local

    ou, ov = chart.origin_uv()
    tu, tv = uv_local(int(uv[0]), int(uv[1]), (ou, ov))
    for i in range(int(u.size)):
        if kinds[i] != key or int(rids[i]) != int(rid):
            continue
        if abs(float(u[i]) - tu) + abs(float(v[i]) - tv) < 1e-6:
            continue
        pts.append((float(u[i]), float(v[i]), float(ss[i])))
    if len(pts) < 1:
        return False
    pts.sort(key=lambda p: float(np.hypot(p[0], p[1])))
    return not _trend_fit(tu, tv, float(s_hat), pts)


def next_trend_probes(
    chains: list[Chain],
    u: np.ndarray,
    v: np.ndarray,
    rr: np.ndarray,
    r_max: float,
    r_step: float = 4.0,
) -> list[tuple[str, float, float, int]]:
    """Next (key, r, θ, rid) along the tangent of the direct FDTD line."""
    _ = rr
    out: list[tuple[str, float, float, int]] = []
    for ch in chains:
        pred = ch.predict_next(u, v, step=r_step)
        if pred is None:
            continue
        pu, pv = pred
        r = float(np.hypot(pu, pv))
        if r > float(r_max) + 1e-9 or r < 1.0:
            continue
        out.append((ch.key, r, float(np.arctan2(pv, pu)), ch.rid))
    return out
