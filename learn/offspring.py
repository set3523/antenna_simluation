# -*- coding: utf-8 -*-
"""맵 + 디퓨전 B + (레거시) 3x3 커널을 한 인터페이스로."""
from __future__ import annotations

import numpy as np

from core import PIX, DIFFUSE_K, Individual, diffuse_grow, repair
from learn.maps import MetricMaps
from learn.diffuse_b import DiffuseB
from learn.chart_log import ChartLog


class OffspringModel:
    METRICS = MetricMaps.METRICS

    def __init__(self, pca_rank: int = 8, pca_buf: int = 96):
        self.maps = MetricMaps()
        self.b = DiffuseB(rank=pca_rank, buf=pca_buf)
        self.chart = ChartLog()
        self.kernel = DIFFUSE_K.copy()
        self.kernel_acc = np.zeros((3, 3), dtype=np.float64)
        self.n = 0
        self.ga_dir = np.zeros(PIX * PIX, dtype=np.float64)
        self.seen: set[bytes] = set()

    @property
    def pcs(self):
        return self.b.pcs

    @property
    def eigs(self):
        return self.b.eigs

    @property
    def best_parts(self):
        return self.maps.best_parts

    def temperature(self) -> float:
        return 1.0

    def annealed_kernel(self) -> np.ndarray:
        k = np.maximum(self.kernel, 0.0)
        s = float(k.sum())
        return k / s if s > 1e-12 else DIFFUSE_K.copy()

    def accuracy(self, key: str) -> float:
        return self.maps.accuracy(key)

    def prob_map(self, key: str = "score") -> np.ndarray:
        return self.maps.prob_map(key)

    def pick_metric(self, rng: np.random.Generator) -> str:
        return self.maps.pick_metric(rng)

    def _refit_pca(self):
        self.b.refit()

    def observe_eval(
        self,
        metal: np.ndarray,
        score: float,
        d_use: float | None = None,
        eta: float | None = None,
        elite: Individual | None = None,
    ):
        self.n += 1
        self.seen.add(np.asarray(metal, dtype=np.uint8).tobytes())
        parts = {"score": float(score)}
        if d_use is not None:
            parts["d_use"] = float(d_use)
        if eta is not None:
            parts["eta"] = float(eta)
        w_pca = self.maps.observe(metal, parts)
        self.chart.observe(metal, float(score))
        if elite is not None:
            resid = metal.astype(np.float64) - elite.metal.astype(np.float64)
            self.b.observe(resid, w_pca)
            if self.n % 4 == 0:
                self.b.refit()

    def observe_peak(self, metal: np.ndarray, score: float, **kw):
        self.observe_eval(metal, score, **kw)

    def observe_step(self, old: np.ndarray, new: np.ndarray, dscore: float):
        if dscore <= 0.0:
            return
        delta = new.astype(np.float64) - old.astype(np.float64)
        ys, xs = np.nonzero(np.abs(delta) > 0.5)
        if len(xs) == 0:
            return
        pad = np.pad(new.astype(np.float64), 1, mode="constant")
        loc = np.zeros((3, 3), dtype=np.float64)
        for i, j in zip(ys, xs):
            loc += pad[i : i + 3, j : j + 3]
        loc /= float(len(xs))
        self.kernel_acc += dscore * loc
        k = np.maximum(self.kernel_acc, 0.0)
        if k.sum() > 1e-12:
            self.kernel = k / k.sum()

    def observe_move(self, elite_metal: np.ndarray, kid_metal: np.ndarray, dscore: float):
        delta = (kid_metal.astype(np.float64) - elite_metal.astype(np.float64)).ravel()
        self.ga_dir = 0.88 * self.ga_dir + 0.12 * float(dscore) * delta

    def _kernel_field(self, elite_metal: np.ndarray) -> np.ndarray:
        k = self.annealed_kernel()
        g = np.asarray(elite_metal, dtype=np.float64)
        pad = np.pad(g, 1, mode="constant")
        p = np.zeros_like(g)
        for di in range(3):
            for dj in range(3):
                p += k[di, dj] * pad[di : di + PIX, dj : dj + PIX]
        return (p - g).ravel()

    def direction_vectors(self, elite: Individual) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        v_ga = self.ga_dir
        v_rl = (self.prob_map("score") - elite.metal.astype(np.float64)).ravel()
        if self.b.ready():
            v_b = self.b.pcs[0]
        else:
            v_b = self._kernel_field(elite.metal)
        return v_ga, v_rl, v_b

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        c = float(np.dot(a, b) / (na * nb))
        return c if np.isfinite(c) else 0.0

    def alignment(self, elite: Individual | None) -> tuple[float, tuple[float, float, float]]:
        if elite is None:
            return 0.0, (0.0, 0.0, 0.0)
        va, vr, vb = self.direction_vectors(elite)
        c_ar = self._cosine(va, vr)
        c_ab = self._cosine(va, vb)
        c_rb = self._cosine(vr, vb)
        return float((c_ar + c_ab + c_rb) / 3.0), (float(c_ar), float(c_ab), float(c_rb))

    def sample_rl(
        self,
        n: int,
        rng: np.random.Generator,
        feed,
        mode: int = -1,
        parent: Individual | None = None,
        metric: str | None = None,
    ) -> list[Individual]:
        kids = []
        for _ in range(n):
            key = metric or self.pick_metric(rng)
            p = self.prob_map(key)
            acc = self.accuracy(key)
            add_s = 0.10 + 0.28 * acc
            rem_s = 0.06 + 0.16 * acc
            if parent is not None:
                metal = parent.metal.copy().astype(np.uint8)
                u = rng.random((PIX, PIX))
                metal[(u < add_s * p) & (metal == 0)] = 1
                metal[(u < rem_s * (1.0 - p)) & (metal == 1)] = 0
            else:
                metal = (rng.random((PIX, PIX)) < p).astype(np.uint8)
            kid = repair(Individual(metal, mode, src=f"rl:{key}"), feed)
            kid.src = f"rl:{key}"
            kids.append(kid)
        return kids

    def sample_diffuse(
        self, n: int, parent: Individual, rng: np.random.Generator, feed, rate: float, steps: int
    ) -> list[Individual]:
        kids = []
        k = self.annealed_kernel()
        for _ in range(n):
            g = diffuse_grow(parent.metal, rng, rate, steps, kernel=k)
            kid = repair(Individual(g, parent.mode, src="diffA"), feed)
            kid.src = "diffA"
            kids.append(kid)
        return kids

    def sample_diffuse_b(
        self, n: int, parent: Individual, rng: np.random.Generator, feed
    ) -> list[Individual]:
        if not self.b.ready():
            return self.sample_diffuse(n, parent, rng, feed, 0.45, 2)
        kids = []
        metal0 = parent.metal.astype(np.float64)
        k = int(self.b.pcs.shape[0])
        scale = np.sqrt(np.maximum(self.b.eigs[:k], 1e-12))
        scale = scale / (float(np.max(scale)) + 1e-12)
        for _ in range(n):
            coeff = rng.normal(0.0, 1.0, size=k) * scale
            delta = (self.b.pcs[:k].T @ coeff).reshape(PIX, PIX)
            logits = 6.0 * (metal0 - 0.5) + 2.4 * delta
            p = 1.0 / (1.0 + np.exp(-np.clip(logits, -12.0, 12.0)))
            g = (rng.random((PIX, PIX)) < p).astype(np.uint8)
            kid = repair(Individual(g, parent.mode, src="diffB"), feed)
            kid.src = "diffB"
            kids.append(kid)
        return kids

    def basin_lines(self, elite: Individual | None) -> list[str]:
        align, (c_ar, c_ab, c_rb) = self.alignment(elite)
        acc_s = self.accuracy("score")
        acc_d = self.accuracy("d_use")
        acc_e = self.accuracy("eta")
        n_uniq = len(self.seen)
        lines = [
            f"align={align:.2f}  GA·RL={c_ar:.2f}  GA·B={c_ab:.2f}  RL·B={c_rb:.2f}  "
            f"acc(score/D/η)={acc_s:.2f}/{acc_d:.2f}/{acc_e:.2f}  unique={n_uniq}  pcs="
            f"{0 if self.pcs is None else self.pcs.shape[0]}"
        ]
        if elite is not None:
            eta = float(elite.eta or 0.0)
            d_use = float(elite.d_use or 0.0)
            room_eta = max(0.0, 1.0 - eta)
            best_d = self.best_parts["d_use"]
            room_d = 0.0 if best_d <= -1e8 else max(0.0, best_d - d_use)
            mix = self.maps.mix()
            lines.append(
                f"ceiling  η={100 * eta:.0f}%  room_η={100 * room_eta:.0f}%  "
                f"D_use={d_use:.2f}  bestD={best_d:.2f}  room_D={room_d:.2f}  "
                f"mix={100 * mix:.0f}%  (전역최적 증명은 아님)"
            )
        return lines
