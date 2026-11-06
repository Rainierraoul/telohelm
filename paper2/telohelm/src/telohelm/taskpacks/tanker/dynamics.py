"""The tanker task's public dynamics: how a receiver turns and flies, a circular orbit without change, and the
closest-approach search (slice 2(c), second half; Codex's TURN235 review: "argument-driven turn/orbit maths").

Moved unchanged from world/physics.py, which now imports them, so the world and the prediction tool share one
source. Everything here reads only its arguments: no scenario, no truth, no disturbance. The orbit change, the
declared limits and the join assessment stay in the world. The half-turn tie band lives here and is read at call
time; set it here (dynamics.TIE_BAND) for sensitivity runs.

Units: nautical miles, seconds, radians. Headings and orbit angles are measured clockwise from north; x points
east and y north. A velocity of speed s on heading h is (s sin h, s cos h).
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Callable

G = 9.80665                      # m/s^2
KNOT = 1852.0 / 3600.0           # m/s per knot
NM_PER_S_PER_KNOT = 1.0 / 3600.0
TWO_PI = 2.0 * math.pi

def kt(v: float) -> float:
    """Knots to nautical miles per second."""
    return v * NM_PER_S_PER_KNOT


def wrap(angle: float) -> float:
    return angle % TWO_PI


def turn_rate(tas_kt: float, bank_deg: float = 30.0) -> float:
    """Ideal level-turn rate in radians per second: g tan(bank) / V."""
    return G * math.tan(math.radians(bank_deg)) / (tas_kt * KNOT)


# Half-turn tie band (D17, class D). A command whose shortest-way difference from the heading is within TIE_BAND
# of 180 degrees, boundary included, turns right; otherwise the receiver turns the shorter way. The first G3 run used
# 1e-15, narrower than the rounding of reversal commands (paper2/results/gaps/G3-RESULTS.md). Read at call time, so
# the G3 sensitivity runs can set it.
TIE_BAND = 1e-9


def _shortest_difference(heading: float, commanded: float) -> float:
    """commanded − heading, wrapped to [−π, π)."""
    return (commanded - heading + math.pi) % TWO_PI - math.pi


def turn_direction(heading: float, commanded: float) -> int:
    """+1 to turn clockwise (right), -1 anticlockwise, 0 if already there. Within TIE_BAND of a half turn, right."""
    diff = _shortest_difference(heading, commanded)
    if diff == 0.0:
        return 0
    if abs(abs(diff) - math.pi) <= TIE_BAND:
        return 1
    return 1 if diff > 0 else -1


def turn_remaining(heading: float, commanded: float) -> float:
    """The angle still to turn in the direction turn_direction chooses (D17: a right turn inside the tie band can be
    slightly more than π, so the shortest-way angle would end it early)."""
    s = turn_direction(heading, commanded)
    diff = _shortest_difference(heading, commanded)
    if s == 0:
        return 0.0
    if s > 0:
        return diff if diff > 0 else diff + TWO_PI
    return -diff


@dataclass(frozen=True)
class ReceiverState:
    t: float
    x: float
    y: float
    heading: float           # actual heading, radians
    commanded: float         # commanded heading, radians
    tas: float               # true airspeed, NM/s
    omega: float             # turn rate, rad/s


def _arc(p0x: float, p0y: float, h0: float, s: int, omega: float, tas: float, wx: float, wy: float,
         dt: float) -> tuple[float, float, float]:
    """Position and heading after dt of a constant-rate turn, exact."""
    h = h0 + s * omega * dt
    k = tas / (s * omega)
    x = p0x + k * (math.cos(h0) - math.cos(h)) + wx * dt
    y = p0y + k * (math.sin(h) - math.sin(h0)) + wy * dt
    return x, y, h


def receiver_at(state: ReceiverState, wind: tuple[float, float], t: float) -> ReceiverState:
    """The receiver's exact state at time t >= state.t, holding the commanded heading."""
    dt = t - state.t
    if dt < 0:
        raise ValueError("cannot move backwards in time")
    wx, wy = wind
    s = turn_direction(state.heading, state.commanded)
    x, y, h = state.x, state.y, state.heading
    if s != 0:
        tau = turn_remaining(h, state.commanded) / state.omega
        if dt <= tau:
            x, y, h = _arc(x, y, h, s, state.omega, state.tas, wx, wy, dt)
            return ReceiverState(t, x, y, wrap(h), state.commanded, state.tas, state.omega)
        x, y, _ = _arc(x, y, h, s, state.omega, state.tas, wx, wy, tau)
        h, dt = state.commanded, dt - tau
    x += (state.tas * math.sin(h) + wx) * dt
    y += (state.tas * math.cos(h) + wy) * dt
    return ReceiverState(t, x, y, wrap(h), state.commanded, state.tas, state.omega)


def turn_end_time(state: ReceiverState) -> float | None:
    """When the current turn ends, or None if not turning."""
    if turn_direction(state.heading, state.commanded) == 0:
        return None
    return state.t + turn_remaining(state.heading, state.commanded) / state.omega


@dataclass(frozen=True)
class CircularOrbit:
    """A clockwise circular orbit with no change: the world's Orbit with delta 0, the same formula."""
    cx: float
    cy: float
    r0: float
    v: float                  # ground speed, NM/s
    theta0: float             # angle at t = 0, clockwise from north

    def position(self, t: float) -> tuple[float, float]:
        a = self.theta0 + self.v * t / self.r0
        return self.cx + self.r0 * math.sin(a), self.cy + self.r0 * math.cos(a)


# The allowance e for error in an evaluated distance, |D_hat - D| <= e. It is qualified empirically (component
# checks in the tests), not proved; every classification below is certain only on that condition and on L.
NUMERIC_ALLOWANCE = 1e-10
MAX_EVALUATIONS = 200_000


@dataclass(frozen=True)
class Approach:
    """An enclosure of the minimum distance on [t0, t1] (backlog B6, as corrected for N10.1 to N10.3).

    lower <= min D <= upper, provided |D_hat - D| <= e and |dD/dt| <= L. t_at_upper is an evaluated time at
    which D <= upper; it need not be a minimiser, and a minimiser need not be unique.
    """
    upper: float
    lower: float
    t_at_upper: float
    evaluations: int
    status: str        # 'resolved': upper - lower <= gap; 'stopped': an early-stop test settled it;
                       # 'unresolved': the work cap was reached first


def closest_approach(distance: Callable[[float], float], t0: float, t1: float,
                     rate_bound: float, gap: float = 1e-6,
                     numeric_allowance: float = NUMERIC_ALLOWANCE, stop_below: float | None = None,
                     stop_above: float | None = None, max_evaluations: int = MAX_EVALUATIONS) -> Approach:
    """Best-first branch and bound for the minimum of the distance on [t0, t1].

    Upper bound U = min over evaluated times of D_hat + e. Lower bound on an interval with midpoint m and
    width w: D_hat(m) - e - L*w/2. An interval whose lower bound exceeds U cannot hold the minimum and is
    dropped. The search stops when U - lower <= gap, when U <= stop_below, when lower > stop_above, or,
    with status 'unresolved', when another split would pass max_evaluations.
    """
    e = numeric_allowance
    if gap <= 2 * e:
        raise ValueError("gap must exceed twice the numerical allowance")
    evals = 0

    def f(t: float) -> float:
        nonlocal evals
        evals += 1
        return distance(t)

    fa, fb = f(t0), f(t1)
    upper, t_up = (fa + e, t0) if fa <= fb else (fb + e, t1)
    heap: list[tuple[float, float, float]] = []

    def push(a: float, b: float) -> None:
        nonlocal upper, t_up
        m = (a + b) / 2
        fm = f(m)
        if fm + e < upper:
            upper, t_up = fm + e, m
        heapq.heappush(heap, (fm - e - rate_bound * (b - a) / 2, a, b))

    push(t0, t1)
    while True:
        lower = min(heap[0][0], upper) if heap else upper
        if stop_below is not None and upper <= stop_below or stop_above is not None and lower > stop_above:
            status = "stopped"
            break
        if upper - lower <= gap:
            status = "resolved"
            break
        if evals + 2 > max_evaluations:
            status = "unresolved"
            break
        lb, a, b = heapq.heappop(heap)
        if lb > upper:
            continue
        m = (a + b) / 2
        push(a, m)
        push(m, b)
    return Approach(upper=upper, lower=lower, t_at_upper=t_up, evaluations=evals, status=status)


