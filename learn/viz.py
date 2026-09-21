# -*- coding: utf-8 -*-
"""PES 차트 Reveal.

전체 경우의 수는 약 2^2601. 그 지도에서는 초반 평가는 점조차 안 된다.
큰 칸은 시드 옆 Hamming 원을 확대한 것. 구석 미니맵이 전체 공간이다.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
import numpy as np

from core import PIX

N_BIN = 81
R_FULL = float(PIX * PIX)  # Hamming 지름. 경우의 수 2^2601 의 2D 껍질.


def _idw_halo(uu, vv, su, sv, ss, reach: float):
    """샘플 근처만 지형 추측. 원 전체를 물감으로 채우지 않음."""
    if len(ss) == 0:
        z = np.zeros_like(uu)
        return z, z
    d2 = (uu[:, :, None] - su[None, None, :]) ** 2 + (vv[:, :, None] - sv[None, None, :]) ** 2
    d = np.sqrt(d2)
    w = np.exp(-d2 / max(reach * reach, 1e-6))
    near = d <= (2.2 * reach)
    w = np.where(near, w, 0.0)
    wsum = np.sum(w, axis=2)
    pred = np.sum(w * ss[None, None, :], axis=2) / np.maximum(wsum, 1e-30)
    conf = np.clip(wsum / (np.max(wsum) + 1e-30), 0.0, 1.0)
    conf = np.where(wsum > 1e-12, conf, 0.0)
    return pred, conf


def save_reveal(out: Path, tag: str, model, note: str = ""):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    chart = getattr(model, "chart", None)
    b = getattr(model, "b", None)
    if chart is None or chart.n() == 0:
        return 0.0

    su, sv, ss, sr = chart.points(b)
    r_disk = float(chart.r_probed)
    local = sr <= max(r_disk, 4.0) * 1.35 if sr.size else np.zeros(0, dtype=bool)
    if local.size and np.any(local):
        su_l, sv_l, ss_l = su[local], sv[local], ss[local]
    else:
        su_l, sv_l, ss_l = su, sv, ss

    # 큰 그림 = 지금 원만 확대. 먼 시드는 미니맵에만.
    r_zoom = max(r_disk, 4.0) * 1.35
    if su_l.size:
        r_zoom = max(r_zoom, float(np.quantile(np.hypot(su_l, sv_l), 0.95)) * 1.25)
    r_zoom = max(r_zoom, 6.0)

    axis = np.linspace(-r_zoom, r_zoom, N_BIN)
    uu, vv = np.meshgrid(axis, axis)
    reach = max(r_zoom / 18.0, 0.8)
    pred, conf = _idw_halo(uu, vv, su_l, sv_l, ss_l, reach)

    vmin = float(np.min(ss)) if ss.size else 0.0
    vmax = float(np.max(ss)) if ss.size else 1.0
    if vmax - vmin < 1e-9:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("inferno")

    rgb = np.zeros((N_BIN, N_BIN, 3), dtype=np.float64)
    rgb[...] = 0.035
    glow = cmap(norm(pred))[..., :3]
    rgb = rgb * (1.0 - 0.85 * conf[..., None]) + glow * (0.85 * conf[..., None])

    n_local = int(np.sum(local)) if local.size else int(su.size)
    n_all = int(chart.n())

    fig, ax = plt.subplots(figsize=(6.6, 6.7), facecolor="#07080a")
    ax.set_facecolor("#07080a")
    extent = (-r_zoom, r_zoom, -r_zoom, r_zoom)
    ax.imshow(rgb, origin="lower", extent=extent, interpolation="nearest", aspect="equal")
    if r_disk > 0.4:
        ax.add_patch(
            Circle((0, 0), r_disk, fill=False, ec="#f2e38a", lw=1.2, ls="--", zorder=4)
        )
    if su_l.size:
        ax.scatter(
            su_l,
            sv_l,
            c=ss_l,
            cmap=cmap,
            norm=norm,
            s=14,
            edgecolors="#fff6c8",
            linewidths=0.35,
            zorder=5,
        )
    ax.plot(0, 0, marker="+", color="#ffffff", markersize=7, zorder=6)
    ax.set_xlim(-r_zoom, r_zoom)
    ax.set_ylim(-r_zoom, r_zoom)
    ax.set_xlabel("zoom u  (local Hamming / B₁)", color="#9a9a9a", fontsize=8)
    ax.set_ylabel("zoom v  (local Hamming / B₂)", color="#9a9a9a", fontsize=8)
    ax.tick_params(colors="#6a6a6a", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#2a2c34")

    zoom_x = R_FULL / max(r_zoom, 1e-9)
    title = (
        f"{tag}   n={n_all}   local={n_local}   "
        f"zoom ×{zoom_x:.0f}   r={r_disk:g}/{R_FULL:.0f}"
    )
    if note:
        title = f"{title}   {note}"
    ax.set_title(title, color="#e8e8e8", fontsize=9.5, pad=8)

    # 구석: 전체 Hamming 공. 우리가 본 것은 원점에 점 하나.
    ins = ax.inset_axes([0.66, 0.66, 0.31, 0.31])
    ins.set_facecolor("#0b0c10")
    ins.add_patch(Circle((0, 0), R_FULL, fill=True, fc="#14161c", ec="#3a3d48", lw=0.7))
    ins.add_patch(Circle((0, 0), r_zoom, fill=False, ec="#f2e38a", lw=0.8))
    if su.size:
        ins.scatter(su, sv, c="#ffd35a", s=2, zorder=5, linewidths=0)
    ins.plot(0, 0, marker="+", color="#ffffff", markersize=4)
    ins.set_xlim(-R_FULL, R_FULL)
    ins.set_ylim(-R_FULL, R_FULL)
    ins.set_aspect("equal")
    ins.set_xticks([])
    ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_color("#8a7a40")
        sp.set_linewidth(0.8)
    ins.set_title(f"full  2^{PIX * PIX}", color="#c8c8c8", fontsize=7, pad=2)
    ins.text(
        0.5,
        -0.08,
        "evals = 1 spec",
        transform=ins.transAxes,
        ha="center",
        va="top",
        color="#8a8a8a",
        fontsize=6.5,
    )

    fig.text(
        0.5,
        0.012,
        "main=zoomed neighbor disk   inset=all 2^2601 cases (points invisible at that scale)",
        ha="center",
        color="#8a8a8a",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / f"{tag}.jpg", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return float(n_all)
