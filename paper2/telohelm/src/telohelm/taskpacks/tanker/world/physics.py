"""Exact motion for the Experiment 2 world (DISTURBANCES.md draft 0.3, sections 2 to 5 and 10).

Units inside this module: nautical miles, seconds, radians. Headings and orbit
angles are measured clockwise from north; x points east and y north. A
velocity of speed s on heading h is (s sin h, s cos h).
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Callable

from scipy import integrate

from .. import dynamics as _D
from ..dynamics import G; from ..dynamics import KNOT; from ..dynamics import NM_PER_S_PER_KNOT; from ..dynamics import TWO_PI; from ..dynamics import MAX_EVALUATIONS; from ..dynamics import NUMERIC_ALLOWANCE; from ..dynamics import Approach; from ..dynamics import ReceiverState; from ..dynamics import _arc; from ..dynamics import _shortest_difference; from ..dynamics import kt; from ..dynamics import receiver_at; from ..dynamics import turn_direction; from ..dynamics import turn_end_time; from ..dynamics import turn_rate; from ..dynamics import turn_remaining; from ..dynamics import wrap

# The receiver's motion and the closest-approach search are the task's public dynamics (packs/tanker_dynamics.py),
# moved there unchanged on 23 October 2026; this module keeps the world's own parts: the declared limits, the orbit
# change and the join assessment. TIE_BAND is read from there at call time.

# Declared limits (DISTURBANCES.md sections 3 to 5). L, the bound on how fast the
# receiver-tanker distance can change, is computed from these in one place (B6).
MAX_TAS_KT, MAX_WIND_KT, MAX_TANKER_KT = 480.0, 60.0, 300.0
DISTANCE_RATE_BOUND = (MAX_TAS_KT + MAX_WIND_KT + MAX_TANKER_KT) * NM_PER_S_PER_KNOT

ORBIT_CHANGE_START, ORBIT_CHANGE_DURATION = 60.0, 120.0


@dataclass(frozen=True)
class Orbit:
    """A clockwise circular orbit whose radius may change by delta, along a half-cosine, from 60 to 180 s.

    The tanker holds its total ground speed v (DISTURBANCES.md section 4), so its
    angular rate is sqrt(v^2 - rdot^2) / r.
    """
    cx: float
    cy: float
    r0: float
    v: float                  # total ground speed, NM/s
    theta0: float             # angle at t = 0, clockwise from north
    delta: float = 0.0        # radius change, NM
    start: float = ORBIT_CHANGE_START
    duration: float = ORBIT_CHANGE_DURATION

    def radius(self, t: float) -> float:
        if t <= self.start or self.delta == 0.0:
            return self.r0
        if t >= self.start + self.duration:
            return self.r0 + self.delta
        return self.r0 + self.delta * (1 - math.cos(math.pi * (t - self.start) / self.duration)) / 2

    def radius_rate(self, t: float) -> float:
        if self.delta == 0.0 or t <= self.start or t >= self.start + self.duration:
            return 0.0
        return self.delta * math.pi / (2 * self.duration) * math.sin(math.pi * (t - self.start) / self.duration)

    def _angle_rate(self, t: float) -> float:
        rd = self.radius_rate(t)
        if abs(rd) > self.v:
            raise ValueError("radius changes faster than the tanker's speed")
        return math.sqrt(self.v * self.v - rd * rd) / self.radius(t)

    def angle(self, t: float) -> float:
        if t <= self.start or self.delta == 0.0:
            return self.theta0 + self.v * t / self.r0
        a_start = self.theta0 + self.v * self.start / self.r0
        end = self.start + self.duration
        upper = min(t, end)
        during, _ = integrate.quad(self._angle_rate, self.start, upper, epsabs=1e-13, epsrel=1e-13, limit=200)
        if t <= end:
            return a_start + during
        return a_start + during + self.v * (t - end) / (self.r0 + self.delta)

    def position(self, t: float) -> tuple[float, float]:
        r, a = self.radius(t), self.angle(t)
        return self.cx + r * math.sin(a), self.cy + r * math.cos(a)


    def velocity(self, t: float) -> tuple[float, float]:
        """Instantaneous ground velocity, differentiating the declared orbit position."""
        r, a, rd, ad = self.radius(t), self.angle(t), self.radius_rate(t), self._angle_rate(t)
        return rd * math.sin(a) + r * ad * math.cos(a), rd * math.cos(a) - r * ad * math.sin(a)


def closest_approach(distance: Callable[[float], float], t0: float, t1: float,
                     rate_bound: float = DISTANCE_RATE_BOUND, gap: float = 1e-6,
                     numeric_allowance: float = NUMERIC_ALLOWANCE, stop_below: float | None = None,
                     stop_above: float | None = None, max_evaluations: int = MAX_EVALUATIONS) -> Approach:
    """The public search with the world's rate bound and the original positional/keyword interface."""
    return _D.closest_approach(distance, t0, t1, rate_bound, gap, numeric_allowance,
                              stop_below, stop_above, max_evaluations)


def __getattr__(name: str):
    if name == "TIE_BAND":                 # read only; set it in packs/tanker_dynamics.py
        return _D.TIE_BAND
    raise AttributeError(name)


def join_assessment(distance: Callable[[float], float], t0: float, t1: float, tolerance: float = 0.25,
                  flag_margin: float = 1e-5, rate_bound: float = DISTANCE_RATE_BOUND,
                  max_evaluations: int = MAX_EVALUATIONS) -> tuple[str, Approach]:
    """'join', 'no join', 'flagged' or 'unresolved' (B6, section 10).

    'join' means min D <= tolerance - flag_margin and 'no join' means min D > tolerance + flag_margin, both
    on the conditions of Approach. 'flagged' means the enclosure was narrowed to flag_margin/2 without either,
    so min D lies within 1.5 * flag_margin of the tolerance; every minimum within flag_margin of it is flagged
    or unresolved. 'unresolved' means the work cap was reached first.
    """
    a = closest_approach(distance, t0, t1, rate_bound=rate_bound, gap=flag_margin / 2,
                         stop_below=tolerance - flag_margin, stop_above=tolerance + flag_margin,
                         max_evaluations=max_evaluations)
    if a.upper <= tolerance - flag_margin:
        return "join", a
    if a.lower > tolerance + flag_margin:
        return "no join", a
    return ("unresolved" if a.status == "unresolved" else "flagged"), a


def join_decision(distance: Callable[[float], float], t0: float, t1: float, tolerance: float = 0.25,
                  flag_margin: float = 1e-5, rate_bound: float = DISTANCE_RATE_BOUND,
                  max_evaluations: int = MAX_EVALUATIONS) -> str:
    """Compatibility wrapper; the episode uses join_assessment to retain the numerical enclosure."""
    return join_assessment(distance, t0, t1, tolerance, flag_margin, rate_bound, max_evaluations)[0]
