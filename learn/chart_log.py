# -*- coding: utf-8 -*-
"""PES 차트 로그. 칸은 구리 픽셀이 아니라 가까운 경우들의 통."""
from __future__ import annotations

import numpy as np

from core import PIX


class ChartLog:
    def __init__(self):
        self.metals: list[np.ndarray] = []
        self.scores: list[float] = []
        self.center: np.ndarray | None = None
        self.r_probed: float = 0.0

    def set_center(self, metal: np.ndarray):
        self.center = np.asarray(metal, dtype=np.float64)

    def set_r_probed(self, r: float):
        self.r_probed = max(self.r_probed, float(r))

    def observe(self, metal: np.ndarray, score: float):
        self.metals.append(np.asarray(metal, dtype=np.uint8).copy())
        self.scores.append(float(score))

    def n(self) -> int:
        return len(self.scores)

    def project(self, metal: np.ndarray, b=None) -> tuple[float, float]:
        """시드/엘리트 기준 (u,v). B 축이 있으면 그 평면, 없으면 해밍 극좌표."""
        m = np.asarray(metal, dtype=np.float64).ravel()
        if self.center is None:
            return 0.0, 0.0
        c = np.asarray(self.center, dtype=np.float64).ravel()
        resid = m - c
        if b is not None and b.ready() and b.pcs.shape[0] >= 1:
            scale = np.sqrt(np.maximum(b.eigs, 1e-12))
            scale = scale / (float(np.max(scale)) + 1e-12)
            u = float(resid @ b.pcs[0]) / (float(scale[0]) + 1e-12)
            if b.pcs.shape[0] > 1:
                v = float(resid @ b.pcs[1]) / (float(scale[1]) + 1e-12)
            else:
                v = 0.0
            return u, v
        h = float(np.sum(np.abs(np.round(m) - np.round(c))))
        diff = np.abs(np.round(m) - np.round(c)).reshape(PIX, PIX)
        ys, xs = np.nonzero(diff)
        if len(xs) == 0:
            return 0.0, 0.0
        cy = (PIX - 1) * 0.5
        cx = (PIX - 1) * 0.5
        ang = float(np.arctan2(float(ys.mean()) - cy, float(xs.mean()) - cx))
        return h * float(np.cos(ang)), h * float(np.sin(ang))

    def points(self, b=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.scores:
            z = np.zeros(0, dtype=np.float64)
            return z, z, z, z
        uv = np.array([self.project(m, b) for m in self.metals], dtype=np.float64)
        u = uv[:, 0]
        v = uv[:, 1]
        s = np.asarray(self.scores, dtype=np.float64)
        r = np.hypot(u, v)
        return u, v, s, r
