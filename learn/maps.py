# -*- coding: utf-8 -*-
"""지표별 구리 점유 맵. 원의 중심 가설."""
from __future__ import annotations

import numpy as np

from core import PIX


class MetricMaps:
    METRICS = ("score", "d_use", "eta")

    def __init__(self):
        z = lambda: np.zeros((PIX, PIX), dtype=np.float64)
        self.map_acc = {k: z() for k in self.METRICS}
        self.wsum = {k: 0.0 for k in self.METRICS}
        self.baseline = {k: 0.0 for k in self.METRICS}
        self.n_map = {k: 0 for k in self.METRICS}
        self.preds: dict[str, list[float]] = {k: [] for k in self.METRICS}
        self.trues: dict[str, list[float]] = {k: [] for k in self.METRICS}
        self.best_parts = {k: -1e18 for k in self.METRICS}
        self.n_on = np.zeros((PIX, PIX), dtype=np.float64)
        self.n_off = np.zeros((PIX, PIX), dtype=np.float64)
        self.n_eval = 0
        self.n_unique_on = np.zeros((PIX, PIX), dtype=np.float64)
        self.seen_masks: set[bytes] = set()

    def accuracy(self, key: str) -> float:
        ys = self.trues[key][-48:]
        ps = self.preds[key][-48:]
        if len(ys) < 8:
            return 0.25
        a = np.asarray(ps, dtype=np.float64)
        b = np.asarray(ys, dtype=np.float64)
        if float(a.std()) < 1e-9 or float(b.std()) < 1e-9:
            return 0.08
        r = float(np.corrcoef(a, b)[0, 1])
        if not np.isfinite(r):
            return 0.08
        return float(np.clip(r, 0.05, 1.0))

    def _pred(self, metal: np.ndarray, key: str) -> float:
        if self.wsum[key] < 1e-12:
            return 0.0
        p = self.map_acc[key] / self.wsum[key]
        m = np.asarray(metal, dtype=np.float64)
        return float(np.sum((m - 0.5) * (p - 0.5)))

    def prob_map(self, key: str = "score") -> np.ndarray:
        if self.wsum[key] < 1e-12:
            return np.full((PIX, PIX), 0.5, dtype=np.float64)
        p = self.map_acc[key] / self.wsum[key]
        return np.clip(p, 0.02, 0.98)

    def mix(self, key: str = "score") -> float:
        p = self.prob_map(key)
        return float(np.mean((p > 0.15) & (p < 0.85)))

    def n_unique(self) -> int:
        return len(self.seen_masks)

    def detected_frac(self, key: str = "score", thresh: float = 0.30) -> float:
        """한 번이라도 구리로 찍어 본 칸 비율."""
        return float(np.mean(self.n_unique_on > 0))

    def reveal_rgb(self, key: str = "score") -> np.ndarray:
        """탐색한 모든 마스크의 합. 어두운 칸=아직 구리로 안 봄, 밝을수록 여러 경우에서 on."""
        hits = self.n_unique_on
        mx = float(np.max(hits)) if hits.size else 0.0
        rgb = np.zeros((PIX, PIX, 3), dtype=np.float64)
        if mx < 1.0:
            return rgb
        t = np.log1p(hits) / np.log1p(mx)
        # 조각모음: 안 본 칸 검정, 본 칸 초록→노랑→흰
        rgb[..., 0] = t * t
        rgb[..., 1] = 0.25 + 0.75 * t
        rgb[..., 2] = 0.08 * (1.0 - t)
        rgb[hits <= 0] = 0.0
        return np.clip(rgb, 0.0, 1.0)

    def observe(self, metal: np.ndarray, parts: dict[str, float]) -> float:
        m = np.asarray(metal, dtype=np.float64)
        on = m >= 0.5
        self.n_on += on.astype(np.float64)
        self.n_off += (~on).astype(np.float64)
        self.n_eval += 1
        raw = np.asarray(metal, dtype=np.uint8).tobytes()
        if raw not in self.seen_masks:
            self.seen_masks.add(raw)
            self.n_unique_on += on.astype(np.float64)
        w_score = 0.05
        for key, val in parts.items():
            if key not in self.METRICS:
                continue
            self.n_map[key] += 1
            if self.n_map[key] == 1:
                self.baseline[key] = val
            pred = self._pred(metal, key)
            self.preds[key].append(pred)
            self.trues[key].append(val)
            self.preds[key] = self.preds[key][-128:]
            self.trues[key] = self.trues[key][-128:]
            self.baseline[key] = 0.9 * self.baseline[key] + 0.1 * val
            acc = self.accuracy(key)
            w = float(np.exp(np.clip((val - self.baseline[key]) / 2.0, -8.0, 8.0))) * acc
            self.map_acc[key] += w * m
            self.wsum[key] += w
            if val > self.best_parts[key]:
                self.best_parts[key] = val
            if key == "score":
                w_score = w
        return w_score

    def pick_metric(self, rng: np.random.Generator) -> str:
        acc = np.asarray([self.accuracy(k) for k in self.METRICS], dtype=np.float64)
        acc = np.maximum(acc, 0.05)
        acc = acc / acc.sum()
        return str(rng.choice(self.METRICS, p=acc))
