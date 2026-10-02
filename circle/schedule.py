# -*- coding: utf-8 -*-
"""Radius schedule on the chart circle. Continuous step / discrete rings / fixed interval."""
from __future__ import annotations

from dataclasses import dataclass


class RadiusSchedule:
    name = "base"

    def radii(self) -> list[float]:
        raise NotImplementedError


@dataclass
class LinearStep(RadiusSchedule):
    """Grow r in small steps. Nearly continuous."""

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
    """Only the given r values. A few rings."""

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
    raise ValueError(f"CIRCLE_R_MODE must be linear | discrete | interval: {kind!r}")
