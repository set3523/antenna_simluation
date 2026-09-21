# -*- coding: utf-8 -*-
"""차트 원 위의 반지름 스케줄. 연속 스텝 / 이산 고리 / 고정 간격."""
from __future__ import annotations

from dataclasses import dataclass


class RadiusSchedule:
    name = "base"

    def radii(self) -> list[float]:
        raise NotImplementedError


@dataclass
class LinearStep(RadiusSchedule):
    """작은 간격으로 r 을 늘림. 거의 연속."""

    r_max: float
    step: float = 4.0
    name: str = "linear"

    def radii(self) -> list[float]:
        step = max(float(self.step), 1e-6)
        r_max = max(float(self.r_max), step)
        out = []
        r = step
        while r <= r_max + 1e-9:
            out.append(float(r))
            r += step
        return out or [r_max]


@dataclass
class DiscreteRings(RadiusSchedule):
    """지정한 r 만. 고리 몇 겹."""

    rings: list
    name: str = "discrete"

    def radii(self) -> list[float]:
        rs = sorted({float(x) for x in self.rings if float(x) > 0})
        return rs or [8.0]


@dataclass
class FixedInterval(RadiusSchedule):
    """r = interval, 2*interval, ... <= r_max."""

    r_max: float
    interval: float = 12.0
    name: str = "interval"

    def radii(self) -> list[float]:
        d = max(float(self.interval), 1e-6)
        r_max = max(float(self.r_max), d)
        n = int(r_max // d)
        return [float(d * i) for i in range(1, n + 1)] or [d]


def make_schedule(
    kind: str,
    *,
    r_max: float = 24.0,
    step: float = 4.0,
    interval: float = 12.0,
    rings: list | None = None,
) -> RadiusSchedule:
    k = str(kind).lower().strip()
    if k in ("linear", "continuous", "step"):
        return LinearStep(r_max=r_max, step=step)
    if k in ("discrete", "rings"):
        return DiscreteRings(rings=list(rings or [8.0, 16.0, 24.0]))
    if k in ("interval", "fixed"):
        return FixedInterval(r_max=r_max, interval=interval)
    raise ValueError(f"CIRCLE_R_MODE 는 linear | discrete | interval: {kind!r}")
