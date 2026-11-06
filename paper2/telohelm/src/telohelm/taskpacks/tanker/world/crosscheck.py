"""Independent checks on the exact reference (plan, Appendix A, row 8; DISTURBANCES.md section 10).

The world computes motion from closed forms (physics.receiver_at, Orbit.position). This module integrates
the same equations of motion numerically, written separately from those formulas:
    x' = TAS·sin h + w_x,   y' = TAS·cos h + w_y,
    h' = ±ω while the heading differs from the command, then 0,
    θ' = sqrt(v² − r'(t)²) / r(t) for the tanker, with r(t) the declared half-cosine radius.
Two integrators are provided: fixed-step classical Runge–Kutta (row 8's 0.1 s and 0.05 s steps), and
SciPy's DOP853 at tight tolerances with an event at the end of each turn. Both locate the end of a turn
from the heading equation itself, since h' is constant during a turn. Agreement between methods that
share an equation would not catch a mistake in that equation; the hand-calculated tests remain for that.
"""
from __future__ import annotations

import math
from typing import Callable

from scipy.integrate import solve_ivp

from . import physics as P


def _turn_sign(h: float, c: float) -> int:
    """Written separately from physics.turn_direction; shares only the declared constant P.TIE_BAND (D17)."""
    d = (c - h + math.pi) % P.TWO_PI - math.pi
    if d == 0.0:
        return 0
    if abs(abs(d) - math.pi) <= P.TIE_BAND:
        return 1                                           # within the tie band of 180 degrees turns right
    return 1 if d > 0 else -1


def _remaining(h: float, c: float, s: int) -> float:
    """Heading still to turn in direction s, in [0, 2π)."""
    return ((c - h) * s) % P.TWO_PI


def tanker_rate(orbit: P.Orbit) -> Callable[[float], float]:
    def rate(t: float) -> float:
        rd = orbit.radius_rate(t)
        return math.sqrt(orbit.v * orbit.v - rd * rd) / orbit.radius(t)
    return rate


def _rk4_second(state: list[float], c: float, tas: float, omega: float, wind: tuple[float, float],
                rate: Callable[[float], float], t0: float, dt: float) -> list[float]:
    """Advance [x, y, h, theta] by one second under command c, in steps of dt, ending turns exactly."""
    x, y, h, th = state
    t = t0
    n = round(1.0 / dt)
    for _ in range(n):
        step = dt
        while step > 1e-15:
            s = _turn_sign(h, c)
            sub = step
            if s != 0:
                sub = min(step, _remaining(h, c, s) / omega)
            hd = s * omega

            def f(tt: float, hh: float) -> tuple[float, float, float]:
                return tas * math.sin(hh) + wind[0], tas * math.cos(hh) + wind[1], rate(tt)

            k1 = f(t, h)
            k2 = f(t + sub / 2, h + hd * sub / 2)
            k3 = f(t + sub / 2, h + hd * sub / 2)
            k4 = f(t + sub, h + hd * sub)
            x += sub / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            y += sub / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            th += sub / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
            h = c if (s != 0 and sub < step) else h + hd * sub
            t += sub
            step -= sub
    return [x, y, h, th]


def fly_fixed_step(rcv: P.ReceiverState, wind: tuple[float, float], orbit: P.Orbit,
                   commands: dict[int, float], seconds: int, dt: float,
                   signs: list | None = None) -> list[tuple[float, float, float, float]]:
    """Receiver and tanker positions at every whole second, by fixed-step RK4. commands: second -> radians.
    If signs is given, the turn direction chosen at each command second is appended as (second, sign)."""
    rate = tanker_rate(orbit)
    state = [rcv.x, rcv.y, rcv.heading, orbit.theta0]
    c = rcv.commanded
    out = []
    for t in range(seconds + 1):
        r = orbit.radius(float(t))
        out.append((state[0], state[1], orbit.cx + r * math.sin(state[3]), orbit.cy + r * math.cos(state[3])))
        if t == seconds:
            break
        c = commands.get(t, c)
        if signs is not None and t in commands:
            signs.append((t, _turn_sign(state[2], c)))
        state = _rk4_second(state, c, rcv.tas, rcv.omega, wind, rate, float(t), dt)
    return out


def fly_general(rcv: P.ReceiverState, wind: tuple[float, float], orbit: P.Orbit,
                commands: dict[int, float], seconds: int,
                rtol: float = 1e-12, atol: float = 1e-13,
                signs: list | None = None) -> list[tuple[float, float, float, float]]:
    """The same positions by SciPy's DOP853, one second at a time, with an event at each turn's end."""
    rate = tanker_rate(orbit)
    x, y, h, th = rcv.x, rcv.y, rcv.heading, orbit.theta0
    c = rcv.commanded
    out = []
    for t in range(seconds + 1):
        r = orbit.radius(float(t))
        out.append((x, y, orbit.cx + r * math.sin(th), orbit.cy + r * math.cos(th)))
        if t == seconds:
            break
        c = commands.get(t, c)
        if signs is not None and t in commands:
            signs.append((t, _turn_sign(h, c)))
        t0, t1 = float(t), float(t + 1)
        while t0 < t1:
            s = _turn_sign(h, c)
            hd = s * rcv.omega
            end = t1 if s == 0 else min(t1, t0 + _remaining(h, c, s) / rcv.omega)

            def rhs(tt, z, hd=hd, h0=h, t_start=t0):
                hh = h0 + hd * (tt - t_start)
                return [rcv.tas * math.sin(hh) + wind[0], rcv.tas * math.cos(hh) + wind[1], rate(tt)]

            if end > t0:
                sol = solve_ivp(rhs, (t0, end), [x, y, th], method="DOP853", rtol=rtol, atol=atol)
                x, y, th = sol.y[:, -1]
            h = c if (s != 0 and end < t1) else h + hd * (end - t0)
            t0 = end
            if s != 0 and end < t1:
                continue
    return out


def fly_exact(rcv: P.ReceiverState, wind: tuple[float, float], orbit: P.Orbit,
              commands: dict[int, float], seconds: int,
              signs: list | None = None) -> list[tuple[float, float, float, float]]:
    """The world's closed forms, for comparison."""
    out = []
    state = rcv
    for t in range(seconds + 1):
        px, py = orbit.position(float(t))
        out.append((state.x, state.y, px, py))
        if t == seconds:
            break
        if t in commands:
            if signs is not None:
                signs.append((t, P.turn_direction(state.heading, commands[t])))
            state = P.ReceiverState(state.t, state.x, state.y, state.heading, commands[t], state.tas, state.omega)
        state = P.receiver_at(state, wind, float(t + 1))
    return out


def max_difference(a: list[tuple[float, ...]], b: list[tuple[float, ...]]) -> tuple[float, float]:
    """Largest receiver and tanker position differences, in nautical miles."""
    rx = max(math.hypot(p[0] - q[0], p[1] - q[1]) for p, q in zip(a, b))
    tx = max(math.hypot(p[2] - q[2], p[3] - q[3]) for p, q in zip(a, b))
    return rx, tx
