"""The lower bound on the earliest possible contact used to label hidden-set cases (note B8; Codex's
derivation in finding B8.16.2).

From position x at time t_H, an aircraft that moves no faster than S in any direction, with no turn limit
and no report to make, cannot be within the tolerance of a tanker at p(s) while
    q(s) = |p(s) − x| − S·(s − t_H) − tolerance > 0.
With the tanker's speed at most V, q changes no faster than V + S, so on an interval [a, b]
    q(s) ≥ q(a) − (V + S)(b − a)   for every s in [a, b].
The search moves from left to right and drops every interval that this bound, less a numerical allowance,
proves positive, splitting the rest down to a set width. The left edge of the first interval it cannot
drop is a lower bound on the earliest contact. If it drops everything up to a time, no contact is possible
before that time. The allowance is an empirical budget for position error, not a proved bound.
"""
from __future__ import annotations

import math
from typing import Callable

WIDTH_S = 0.01
ALLOWANCE_NM = 1e-6
TOLERANCE_NM = 0.25
MAX_INTERVALS = 2_000_000


def earliest_contact_bound(x: tuple[float, float], t_start: float, speed_bound: float,
                           tanker_position: Callable[[float], tuple[float, float]], tanker_speed: float,
                           until: float, width: float = WIDTH_S, allowance: float = ALLOWANCE_NM,
                           tolerance: float = TOLERANCE_NM, max_intervals: int = MAX_INTERVALS) -> dict:
    """Returns {'status': 'bound', 'lower': L} with L at or below the earliest relaxed contact; or
    {'status': 'excluded', 'until': until} when no contact is possible up to until; or 'unresolved'."""
    if until < t_start:
        raise ValueError("until precedes the start")
    rate = tanker_speed + speed_bound

    def q(s: float) -> float:
        px, py = tanker_position(s)
        return math.hypot(px - x[0], py - x[1]) - speed_bound * (s - t_start) - tolerance

    stack = [(t_start, until)]
    examined = 0
    while stack:
        a, b = stack.pop()
        examined += 1
        if examined > max_intervals:
            return {"status": "unresolved", "examined": examined}
        if q(a) - allowance - rate * (b - a) > 0:
            continue                                   # no contact anywhere in [a, b]
        if b - a <= width:
            return {"status": "bound", "lower": a, "examined": examined}
        m = (a + b) / 2
        stack.append((m, b))                          # right half later
        stack.append((a, m))                          # left half first
    return {"status": "excluded", "until": until, "examined": examined}
