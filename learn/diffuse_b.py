# -*- coding: utf-8 -*-
"""디퓨전 B: 잔차 PCA. 차트 축 + 테두리 허용 주름."""
from __future__ import annotations

import numpy as np

from core import PIX


class DiffuseB:
    def __init__(self, rank: int = 8, buf: int = 96):
        self.rank = max(int(rank), 1)
        self.buf = max(int(buf), 8)
        self.resid: list[tuple[np.ndarray, float]] = []
        self.pcs: np.ndarray | None = None
        self.eigs: np.ndarray | None = None

    def observe(self, residual: np.ndarray, weight: float = 1.0):
        self.resid.append((np.asarray(residual, dtype=np.float64).ravel(), max(float(weight), 1e-6)))
        if len(self.resid) > self.buf:
            self.resid = self.resid[-self.buf :]

    def refit(self):
        if len(self.resid) < 4:
            self.pcs = None
            self.eigs = None
            return
        x = np.stack([v for v, _ in self.resid], axis=0)
        w = np.asarray([wt for _, wt in self.resid], dtype=np.float64)
        w = w / (float(w.sum()) + 1e-30)
        mu = (w[:, None] * x).sum(axis=0)
        xw = (x - mu) * np.sqrt(w)[:, None]
        _u, s, vt = np.linalg.svd(xw, full_matrices=False)
        k = min(self.rank, int(vt.shape[0]))
        self.pcs = vt[:k]
        self.eigs = s[:k] ** 2

    def ready(self) -> bool:
        return self.pcs is not None and self.eigs is not None and self.pcs.shape[0] >= 1

    def chart_axes(self) -> tuple[np.ndarray, np.ndarray] | None:
        """내려다보는 2축. 없으면 1축만 또는 None."""
        if not self.ready():
            return None
        v0 = self.pcs[0]
        v1 = self.pcs[1] if self.pcs.shape[0] > 1 else None
        return v0, v1

    def ring_delta(self, r: float, theta: float) -> np.ndarray:
        """차트 원 위 한 점의 마스크 잔차. r 은 축 진폭 스케일."""
        if not self.ready():
            return np.zeros((PIX, PIX), dtype=np.float64)
        k = int(self.pcs.shape[0])
        scale = np.sqrt(np.maximum(self.eigs[:k], 1e-12))
        scale = scale / (float(np.max(scale)) + 1e-12)
        coeff = np.zeros(k, dtype=np.float64)
        coeff[0] = float(r) * float(np.cos(theta)) * scale[0]
        if k > 1:
            coeff[1] = float(r) * float(np.sin(theta)) * scale[1]
        if k > 2:
            coeff[2:] = 0.15 * float(r) * scale[2:]
        return (self.pcs[:k].T @ coeff).reshape(PIX, PIX)
