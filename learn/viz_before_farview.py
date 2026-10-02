# -*- coding: utf-8 -*-
"""Reveal = chart of all cases. Orange (inferred) and per-metric trend lines."""
from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
import numpy as np

from core import PIX
from learn.infer import TAU_ORANGE, length_scale, predict

N_BIN = 81
R_FULL = float(PIX * PIX)
# Zoom near the feed even for large disks like r=240 (the old r≈24 view). Do not zoom out to the full r_disk.
R_REVEAL_ZOOM_CAP = 48.0
GRAY = np.array([0.22, 0.22, 0.24])
ORANGE = np.array([0.92, 0.48, 0.10])

# Direct-FDTD trend lines. Black dots get a white edge. Metric dots are larger than green and drawn on top.
ROOT_STYLE = {
    "seed": dict(c="#2ad866", ec="#042208", label="direct FDTD"),
    "leave": dict(c="#ff1a1a", ec="#fff0f0", halo="#4a0000", label="total efficiency"),
    "d_peak": dict(c="#ff1a1a", ec="#fff0f0", halo="#4a0000", label="total efficiency"),
    "beam": dict(c="#ffffff", ec="#0a0a0a", halo="#111111", label="cone 60°"),
    "eta": dict(c="#0a0a0a", ec="#ffffff", halo="#f5f5f5", label="η"),
}


def save_reveal(out: Path, tag: str, model, note: str = ""):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    chart = getattr(model, "chart", None)
    b = getattr(model, "b", None)
    if chart is None or chart.n() == 0:
        return 0.0

    su, sv, ss, sr = chart.points(b)
    if hasattr(chart, "kinds_at_points"):
        kinds = chart.kinds_at_points()
        rids = chart.roots_at_points()
    else:
        kinds = chart.kinds_arr() if hasattr(chart, "kinds_arr") else np.array(["seed"] * su.size)
        rids = chart.roots_arr() if hasattr(chart, "roots_arr") else np.zeros(su.size, dtype=np.int32)
    if kinds.size != su.size:
        kinds = np.asarray(["seed"] * su.size, dtype=object)
    if rids.size != su.size:
        rids = np.zeros(su.size, dtype=np.int32)
    r_disk = float(chart.r_probed)
    r_seed = float(getattr(chart, "r_seed", 0.0) or 0.0)
    # points() = local coordinates relative to the anchor
    ou, ov = 0.0, 0.0
    r_frame = max(r_disk, 4.0)
    r_step = 4.0
    if r_seed > r_frame + 1e-6:
        r_step = max(float(r_seed) - r_frame, 4.0)
    # Kernel ℓ: if the target r_seed (e.g. 520) is far off, base it on the r_disk explored so far
    r_ell = max(r_frame, 4.0) if r_seed > r_frame * 1.25 else max(r_seed, 4.0)
    ell = length_scale(su, sv, r_step=r_step, r_seed=r_ell)

    r_view = r_frame * 1.35
    local = sr <= r_view + 1e-6 if sr.size else np.zeros(0, dtype=bool)
    if local.size and np.any(local):
        su_l, sv_l, ss_l, k_l, rid_l = su[local], sv[local], ss[local], kinds[local], rids[local]
    else:
        su_l, sv_l, ss_l, k_l, rid_l = su, sv, ss, kinds, rids

    # Zoom window: based on r_probed, but only near the feed even on a large disk (yellow dashed circle stays at r_disk).
    r_zoom = min(r_frame * 1.40, R_REVEAL_ZOOM_CAP)
    if su_l.size:
        q = float(np.quantile(np.hypot(su_l, sv_l), 0.95)) * 1.25
        r_zoom = min(max(r_zoom, q), R_REVEAL_ZOOM_CAP)
    r_zoom = max(r_zoom, 6.0)
    # FDTD (green) samples rings only; draw smaller as r_zoom and point count grow so the orange area shows.
    ms_seed = float(np.clip(165.0 / r_zoom, 2.0, 20.0))
    if su_l.size:
        n_fdtd = int(np.sum(np.asarray(k_l, dtype=object) == "seed"))
        if n_fdtd > 80:
            ms_seed *= float(np.clip(80.0 / max(n_fdtd, 1), 0.30, 1.0))
    ms_seed = max(ms_seed, 1.2)
    seed_lw = float(np.clip(0.55 / max(ms_seed, 1.0), 0.08, 0.35))
    ms_root = float(np.clip(480.0 / r_zoom, 22.0, 52.0))

    axis = np.linspace(-r_zoom, r_zoom, N_BIN)
    uu, vv = np.meshgrid(axis, axis)
    pred, conf = predict(uu, vv, su_l, sv_l, ss_l, ell)

    vmin = float(np.min(ss)) if ss.size else 0.0
    vmax = float(np.max(ss)) if ss.size else 1.0
    if vmax - vmin < 1e-9:
        vmax = vmin + 1.0
    t = np.clip((pred - vmin) / (vmax - vmin), 0.0, 1.0)

    rgb = np.broadcast_to(GRAY, (N_BIN, N_BIN, 3)).copy()
    orange = conf >= TAU_ORANGE
    glow = 0.40 + 0.60 * t
    rgb[orange] = ORANGE * glow[orange][:, None]

    n_all = int(chart.n())
    n_orange = int(np.sum(orange))
    n_gray = int(np.sum(~orange))
    fig, (ax, ins) = plt.subplots(1, 2, figsize=(10.8, 5.6), facecolor="#07080a")
    for a in (ax, ins):
        a.set_facecolor("#07080a")

    extent = (-r_zoom, r_zoom, -r_zoom, r_zoom)
    ax.imshow(rgb, origin="lower", extent=extent, interpolation="nearest", aspect="equal")
    if r_seed > 0.4 and r_seed <= r_zoom * 1.08:
        ax.add_patch(Circle((0, 0), r_seed, fill=False, ec="#d8d8d8", lw=1.0, ls=":"))
    if r_disk > 0.4:
        ax.add_patch(Circle((0, 0), r_disk, fill=False, ec="#f2e38a", lw=1.3, ls="--"))

    paths = getattr(chart, "root_paths", None) or []

    def _draw_roots(a, uu_, vv_, kk_, rid_, lw, ms_seed, ms_root, seed_lw=0.25):
        a.plot(0, 0, marker="+", color="#ffffff", markersize=7, zorder=6)
        for key, _rid, poly in paths:
            st = ROOT_STYLE.get(key)
            if st is None or key == "seed" or len(poly) < 2:
                continue
            xs_l = [p[0] for p in poly]
            ys_l = [p[1] for p in poly]
            if st.get("halo"):
                a.plot(xs_l, ys_l, color=st["halo"], lw=lw + 2.6, alpha=0.95, zorder=4)
            a.plot(xs_l, ys_l, color=st["c"], lw=lw, alpha=1.0, zorder=5)
        if uu_.size:
            kinds = np.asarray(kk_, dtype=object)
            a.scatter(
                uu_,
                vv_,
                c=ROOT_STYLE["seed"]["c"],
                s=ms_seed,
                edgecolors=ROOT_STYLE["seed"]["ec"],
                linewidths=seed_lw,
                zorder=6,
            )
            for key in ("leave", "beam", "eta"):
                st = ROOT_STYLE[key]
                m = kinds == key
                if not np.any(m):
                    continue
                a.scatter(
                    uu_[m],
                    vv_[m],
                    c=st["c"],
                    s=ms_root,
                    edgecolors=st["ec"],
                    linewidths=1.15,
                    zorder=8,
                )

    if su_l.size:
        _draw_roots(
            ax,
            su_l,
            sv_l,
            k_l,
            rid_l,
            lw=2.5,
            ms_seed=ms_seed,
            ms_root=ms_root,
            seed_lw=seed_lw,
        )
    trends = getattr(chart, "trends", None) or []
    for item in trends:
        if len(item) >= 6:
            _rid, fu, fv, tu, tv, tkey = item[:6]
        else:
            _rid, tu, tv, tkey = item[:4]
            fu, fv = 0.0, 0.0
        if su.size and float(np.min(np.hypot(tu - su, tv - sv))) < 0.45 * ell:
            continue
        st = ROOT_STYLE.get(tkey, ROOT_STYLE["d_peak"])
        if st.get("halo"):
            ax.plot([fu, tu], [fv, tv], color=st["halo"], lw=2.2, ls="--", alpha=0.7, zorder=3)
        ax.plot([fu, tu], [fv, tv], color=st["c"], lw=1.3, ls="--", alpha=0.9, zorder=4)
        ax.scatter(
            [tu],
            [tv],
            c=st["c"],
            s=48,
            marker="x",
            linewidths=1.6,
            zorder=9,
        )
    ax.set_xlim(-r_zoom, r_zoom)
    ax.set_ylim(-r_zoom, r_zoom)
    ax.set_xlabel("u (horizontal)", color="#c8c8c8", fontsize=9)
    ax.set_ylabel("v (vertical)", color="#c8c8c8", fontsize=9)
    ax.tick_params(colors="#8a8a8a", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#2a2c34")
    ax.set_title(
        "Left = direct and inferred, zoomed (r≤yellow dashed)\n"
        "solid=trend line   dashed=next cell on trend",
        color="#e8e8e8",
        fontsize=10,
        loc="left",
    )
    handles = [
        Patch(facecolor=ORANGE, edgecolor="none", label="inferred"),
        Patch(facecolor=GRAY, edgecolor="none", label="unexplored"),
    ]
    for key in ("seed", "leave", "beam", "eta"):
        st = ROOT_STYLE[key]
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color=st["c"],
                markerfacecolor=st["c"],
                markeredgecolor=st["ec"],
                lw=1.2,
                label=st["label"],
            )
        )
    ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=8,
        facecolor="#12131a",
        edgecolor="#2a2c34",
        labelcolor="#d8d8d8",
    )

    ins.add_patch(Circle((0, 0), R_FULL, fill=True, fc="#14161c", ec="#3a3d48", lw=0.8))
    ins.add_patch(Circle((0, 0), r_zoom, fill=False, ec="#f2e38a", lw=1.0))
    if su.size:
        _draw_roots(ins, su, sv, kinds, rids, lw=1.2, ms_seed=2.5, ms_root=14)
    ins.set_xlim(-R_FULL, R_FULL)
    ins.set_ylim(-R_FULL, R_FULL)
    ins.set_aspect("equal")
    ins.set_xticks([])
    ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_color("#8a7a40")
    ins.set_title(
        f"Right = all possible cases  2^{PIX * PIX}\ntrend lines = direct FDTD per metric",
        color="#c8c8c8",
        fontsize=10,
        loc="left",
    )

    head = f"{tag}   {n_all} computed   r={r_disk:g}   ℓ={ell:.1f}"
    if note:
        head = f"{head}   {note}"
    fig.suptitle(head, color="#e8e8e8", fontsize=12, y=0.98)
    fig.text(
        0.5,
        0.015,
        f"unexplored {n_gray} cells, inferred {n_orange} cells, legend = per-metric trend lines (not copper)",
        ha="center",
        color="#9a9a9a",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.90))
    dest = out / f"{tag}.jpg"
    tmp = Path(tempfile.gettempdir()) / f"antenna_reveal_{tag.replace('/', '_')}.jpg"
    try:
        fig.savefig(tmp, dpi=120, facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)
    try:
        dest.write_bytes(tmp.read_bytes())
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    return float(n_all)
