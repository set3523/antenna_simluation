# -*- coding: utf-8 -*-
"""PES chart log. A cell is a bin of nearby cases, not a copper pixel."""
from __future__ import annotations

import numpy as np

from core import PIX


class ChartLog:
    def __init__(self):
        self.metals: list[np.ndarray] = []
        self.scores: list[float] = []
        self.d_uses: list[float] = []
        self.etas: list[float] = []
        self.d_aps: list[float] = []
        self.d_peaks: list[float] = []
        self.beam_dbs: list[float] = []
        self.leaves: list[float] = []
        self.uvs: list[tuple[float, float] | None] = []
        self.local_uvs: list[tuple[float, float] | None] = []
        self.kinds: list[str] = []
        self.roots: list[int] = []
        # (rid, tip_u, tip_v, pred_u, pred_v, key)
        self.trends: list[tuple[int, float, float, float, float, str]] = []
        self.root_paths: list[tuple[str, int, list[tuple[float, float]]]] = []
        self.center: np.ndarray | None = None
        self.origin_u: int = 0
        self.origin_v: int = 0
        self.r_probed: float = 0.0
        self.r_seed: float = 0.0

    def set_center(self, metal: np.ndarray):
        """Holds elite mask references only. Origin is `set_origin` / `bind_chart_origin` (0,0)."""
        self.center = np.asarray(metal, dtype=np.float64)

    def set_origin(self, u: int, v: int):
        """Global chart (u,v) origin. A neutral anchor such as index=0, separate from the gate rectangle."""
        self.origin_u = int(u)
        self.origin_v = int(v)

    def origin_uv(self) -> tuple[int, int]:
        return (int(self.origin_u), int(self.origin_v))

    def set_r_seed(self, r: float):
        self.r_seed = float(r)

    def set_r_probed(self, r: float):
        self.r_probed = max(self.r_probed, float(r))

    _LOCAL_MAX = 500_000  # r_max ~520; mask->uv project is not used

    def _local_from_global(self, tu: float, tv: float) -> tuple[float, float]:
        ou, ov = int(self.origin_u), int(self.origin_v)
        cap = int(self._LOCAL_MAX)
        if abs(ou) > cap or abs(ov) > cap:
            ou, ov = 0, 0
        return float(tu) - float(ou), float(tv) - float(ov)

    def _point_indices(self) -> list[int]:
        """scores indices: only points whose FDTD sample uv lies within the local radius (to align with points/series)."""
        cap = int(self._LOCAL_MAX)
        out: list[int] = []
        for i in range(len(self.scores)):
            loc = None
            if i < len(self.local_uvs):
                loc = self.local_uvs[i]
            if loc is None and i < len(self.uvs) and self.uvs[i] is not None:
                loc = self._local_from_global(self.uvs[i][0], self.uvs[i][1])
            if loc is None:
                continue
            du, dv = float(loc[0]), float(loc[1])
            if abs(du) > cap or abs(dv) > cap:
                continue
            out.append(i)
        return out

    def observe(
        self,
        metal: np.ndarray,
        score: float,
        uv: tuple[float, float] | None = None,
        kind: str = "seed",
        d_use: float | None = None,
        eta: float | None = None,
        root: int = 0,
        d_ap: float | None = None,
        d_peak: float | None = None,
        beam_db: float | None = None,
        leave: float | None = None,
    ):
        self.metals.append(np.asarray(metal, dtype=np.uint8).copy())
        self.scores.append(float(score))
        self.d_uses.append(float(d_use) if d_use is not None else float("nan"))
        self.etas.append(float(eta) if eta is not None else float("nan"))
        self.d_aps.append(float(d_ap) if d_ap is not None else float("nan"))
        lv = float(leave) if leave is not None else (float(d_peak) if d_peak is not None else float("nan"))
        self.d_peaks.append(lv)
        self.leaves.append(lv)
        self.beam_dbs.append(float(beam_db) if beam_db is not None else float("nan"))
        if uv is None:
            self.uvs.append(None)
            self.local_uvs.append(None)
        else:
            tu, tv = float(uv[0]), float(uv[1])
            self.uvs.append((tu, tv))
            self.local_uvs.append(self._local_from_global(tu, tv))
        self.kinds.append(str(kind or "seed"))
        self.roots.append(int(root or 0))

    def replace_at_uv(
        self,
        uv: tuple[float, float],
        metal: np.ndarray,
        score: float,
        *,
        kind: str | None = None,
        d_use: float | None = None,
        eta: float | None = None,
        d_ap: float | None = None,
        d_peak: float | None = None,
        beam_db: float | None = None,
        leave: float | None = None,
    ):
        """Same chart cell: update the mask only. Pin the post-hill-climb coordinate to the original grid."""
        tu, tv = float(uv[0]), float(uv[1])
        for i in range(len(self.uvs) - 1, -1, -1):
            got = self.uvs[i]
            if got is None:
                continue
            if abs(float(got[0]) - tu) > 1e-3 or abs(float(got[1]) - tv) > 1e-3:
                continue
            self.metals[i] = np.asarray(metal, dtype=np.uint8).copy()
            self.scores[i] = float(score)
            if d_use is not None:
                self.d_uses[i] = float(d_use)
            if eta is not None:
                self.etas[i] = float(eta)
            if d_ap is not None and i < len(self.d_aps):
                self.d_aps[i] = float(d_ap)
            if d_peak is not None and i < len(self.d_peaks):
                self.d_peaks[i] = float(d_peak)
            if leave is not None:
                if i < len(self.leaves):
                    self.leaves[i] = float(leave)
                if i < len(self.d_peaks):
                    self.d_peaks[i] = float(leave)
            if beam_db is not None and i < len(self.beam_dbs):
                self.beam_dbs[i] = float(beam_db)
            if kind is not None and i < len(self.kinds):
                self.kinds[i] = str(kind)
            return True
        return False

    def observe_or_replace(
        self,
        uv: tuple[float, float],
        metal: np.ndarray,
        score: float,
        **kwargs,
    ) -> None:
        if not self.replace_at_uv(uv, metal, score, **kwargs):
            self.observe(metal, score, uv=uv, **kwargs)

    def mark_kind(self, metal: np.ndarray, kind: str = "seed"):
        raw = np.asarray(metal, dtype=np.uint8).tobytes()
        for i in range(len(self.metals) - 1, -1, -1):
            if np.asarray(self.metals[i], dtype=np.uint8).tobytes() == raw:
                if i < len(self.kinds):
                    self.kinds[i] = str(kind)
                return

    def n(self) -> int:
        return len(self.scores)

    def project(self, metal: np.ndarray, b=None) -> tuple[float, float]:
        """Mask index -> (u,v). b is ignored (kept for compatibility)."""
        from circle.index_map import mask_to_uv

        _ = b
        return mask_to_uv(metal)

    def points(self, b=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Local (du,dv) relative to the anchor. Uses FDTD sample uv only (no project)."""
        _ = b
        idx = self._point_indices()
        if not idx:
            z = np.zeros(0, dtype=np.float64)
            return z, z, z, z
        u_out: list[float] = []
        v_out: list[float] = []
        s_out: list[float] = []
        for i in idx:
            loc = self.local_uvs[i]
            if loc is None:
                loc = self._local_from_global(self.uvs[i][0], self.uvs[i][1])
            du, dv = float(loc[0]), float(loc[1])
            u_out.append(du)
            v_out.append(dv)
            s_out.append(float(self.scores[i]))
        if not s_out:
            z = np.zeros(0, dtype=np.float64)
            return z, z, z, z
        u = np.asarray(u_out, dtype=np.float64)
        v = np.asarray(v_out, dtype=np.float64)
        s = np.asarray(s_out, dtype=np.float64)
        r = np.hypot(u, v)
        return u, v, s, r

    def series_at_points(self, key: str) -> np.ndarray:
        """Same rows as points(), for aligning predict and kinds."""
        idx = self._point_indices()
        if not idx:
            return np.zeros(0, dtype=np.float64)
        full = self.series(key)
        ix = np.asarray(idx, dtype=np.intp)
        if full.size <= int(np.max(ix)):
            pad = np.full(int(np.max(ix)) + 1 - full.size, np.nan)
            full = np.concatenate([full, pad])
        return np.asarray(full, dtype=np.float64)[ix]

    def kinds_at_points(self) -> np.ndarray:
        idx = self._point_indices()
        if not idx:
            return np.array([], dtype=object)
        k = self.kinds_arr()
        return k[np.asarray(idx, dtype=np.intp)]

    def roots_at_points(self) -> np.ndarray:
        idx = self._point_indices()
        if not idx:
            return np.zeros(0, dtype=np.int32)
        r = self.roots_arr()
        return r[np.asarray(idx, dtype=np.intp)]

    def point_rows(self) -> np.ndarray:
        """Row number of points()[k] in scores / metals / uvs."""
        idx = self._point_indices()
        if not idx:
            return np.zeros(0, dtype=np.intp)
        return np.asarray(idx, dtype=np.intp)

    def kinds_arr(self) -> np.ndarray:
        n = len(self.scores)
        if n == 0:
            return np.array([], dtype=object)
        kinds = list(self.kinds) + ["seed"] * max(0, n - len(self.kinds))
        return np.asarray(kinds[:n], dtype=object)

    def roots_arr(self) -> np.ndarray:
        n = len(self.scores)
        if n == 0:
            return np.zeros(0, dtype=np.int32)
        roots = list(self.roots) + [0] * max(0, n - len(self.roots))
        return np.asarray(roots[:n], dtype=np.int32)

    def reconnect_by_heading(self, sep: float = 0.52):
        """Reconnect to the trend line whose angle and r trend fits better. Spatial inference is unchanged."""
        from circle.pattern import reconnect_by_pattern

        reconnect_by_pattern(self, sep=sep)

    def series(self, key: str) -> np.ndarray:
        if key in ("leave", "leave_db"):
            a = np.asarray(self.leaves if self.leaves else self.d_peaks, dtype=np.float64)
        elif key in ("d_peak", "d_db"):
            a = np.asarray(self.d_peaks, dtype=np.float64)
        elif key == "d_ap":
            a = np.asarray(self.d_aps, dtype=np.float64)
        elif key in ("beam", "beam_db"):
            a = np.asarray(self.beam_dbs, dtype=np.float64)
        elif key == "d_use":
            a = np.asarray(self.d_uses, dtype=np.float64)
        elif key == "eta":
            a = np.asarray(self.etas, dtype=np.float64)
        else:
            a = np.asarray(self.scores, dtype=np.float64)
        if a.size < len(self.scores):
            pad = np.full(len(self.scores) - a.size, np.nan)
            a = np.concatenate([a, pad])
        return a[: len(self.scores)].copy()
