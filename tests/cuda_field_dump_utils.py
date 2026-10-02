# -*- coding: utf-8 -*-
"""Analysis of CUDA field npz dumps (for test_best_plot_jpg)."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from core import Individual, PIX


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def individual_from_eval_tuple(metal, freqs, tup) -> Individual:
    sc, worst, s11, mindb, minhz, nused, air, cut, faces, leave, field_extra = tup
    ind = Individual(metal=np.asarray(metal, dtype=np.uint8), mode=-1)
    ind.score = sc
    ind.worst = worst
    ind.s11 = s11
    ind.air = air
    ind.cut = cut
    ind.field_extra = field_extra
    ind.faces = None if faces is None else np.asarray(faces, dtype=np.float64)
    return ind


def field_max_abs_diff(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return float("nan")
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape:
        return float("nan")
    return float(np.max(np.abs(aa - bb)))


def analyze_cuda_field_dir(cuda_root: Path) -> list[dict]:
    """Read byproduct/cuda_fields/*.npz and tabulate signs of overlap or cache contamination."""
    cuda_root = Path(cuda_root)
    if not cuda_root.is_dir():
        return []
    tags: dict[str, dict[str, Path]] = {}
    for p in sorted(cuda_root.glob("*.npz")):
        stem = p.stem
        if stem.endswith("_direct"):
            base = stem[: -len("_direct")]
            tags.setdefault(base, {})["direct"] = p
        else:
            tags.setdefault(stem, {})["fill"] = p

    rows: list[dict] = []
    prev_mid: np.ndarray | None = None
    prev_metal: np.ndarray | None = None
    for tag in sorted(tags.keys()):
        paths = tags[tag]
        row: dict = {"tag": tag}
        fill_mid = direct_mid = metal = None
        if "fill" in paths:
            z = np.load(paths["fill"], allow_pickle=False)
            fill_mid = z["mid_cropped"] if "mid_cropped" in z.files else z.get("mid")
            metal = z["metal"] if "metal" in z.files else None
            row["fill_score"] = float(z["score"]) if "score" in z.files else float("nan")
        if "direct" in paths:
            zd = np.load(paths["direct"], allow_pickle=False)
            direct_mid = (
                zd["mid_cropped"] if "mid_cropped" in zd.files else zd.get("mid")
            )
            row["direct_score"] = (
                float(zd["score"]) if "score" in zd.files else float("nan")
            )
        row["mid_fill_vs_direct_maxdiff"] = field_max_abs_diff(fill_mid, direct_mid)
        if metal is not None and fill_mid is not None:
            m = np.asarray(metal, dtype=np.float64)
            p = np.asarray(fill_mid, dtype=np.float64)
            if p.shape == (PIX, PIX):
                row["mid_metal_corr"] = _corr(m.T, p)
            else:
                row["mid_metal_corr"] = float("nan")
        else:
            row["mid_metal_corr"] = float("nan")
        if prev_mid is not None and fill_mid is not None:
            row["mid_vs_prev_fill_corr"] = _corr(prev_mid, fill_mid)
            if prev_metal is not None and metal is not None:
                row["metal_vs_prev_hamming"] = int(
                    np.count_nonzero(prev_metal != metal)
                )
            else:
                row["metal_vs_prev_hamming"] = -1
        else:
            row["mid_vs_prev_fill_corr"] = float("nan")
            row["metal_vs_prev_hamming"] = -1
        rows.append(row)
        if fill_mid is not None:
            prev_mid = np.asarray(fill_mid, dtype=np.float64)
        if metal is not None:
            prev_metal = np.asarray(metal, dtype=np.uint8)
    return rows


def write_diag_csv(cuda_root: Path, rows: list[dict]) -> Path:
    cuda_root = Path(cuda_root)
    cuda_root.mkdir(parents=True, exist_ok=True)
    out = cuda_root / "diag.csv"
    if not rows:
        return out
    fields = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return out


def print_diag_report(rows: list[dict]) -> None:
    if not rows:
        print("(no cuda_fields npz, run the test then check byproduct/cuda_fields)")
        return
    print("\n=== CUDA npz diagnostics (overlap suspect: high mid_vs_prev_fill_corr and large metal_vs_prev_hamming) ===")
    print(
        f"{'tag':<18} {'mid∧metal':>9} {'mid∧prev':>9} {'Δmetal':>7} "
        f"{'|fill-direct|':>14}"
    )
    for r in rows:
        print(
            f"{r['tag']:<18} "
            f"{r.get('mid_metal_corr', float('nan')):9.3f} "
            f"{r.get('mid_vs_prev_fill_corr', float('nan')):9.3f} "
            f"{r.get('metal_vs_prev_hamming', -1):7d} "
            f"{r.get('mid_fill_vs_direct_maxdiff', float('nan')):14.6g}"
        )
    warn = [
        r
        for r in rows
        if np.isfinite(r.get("mid_vs_prev_fill_corr", float("nan")))
        and r.get("metal_vs_prev_hamming", 0) > 100
        and r["mid_vs_prev_fill_corr"] > 0.55
    ]
    if warn:
        print(f"\n⚠ {len(warn)} overlap candidates (corr>0.55 with previous mid, copper differs): "
              + ", ".join(r["tag"] for r in warn))
