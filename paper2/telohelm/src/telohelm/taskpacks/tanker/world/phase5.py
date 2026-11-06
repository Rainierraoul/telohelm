"""Phase-5 help cases: generation, the help certificate and the with-help witness.

DISTURBANCES.md draft 0.3, section 11, with Codex's acceptance criteria (21:51 on
9 October; C3.1 to C3.5 on 10 October). For development seeds only: nothing here
touches a scored stream.

Construction: the named tanker's reports stop for good at t_b, before an
unannounced orbit change starting at 60 s; the goal has a join deadline T. Two
worlds, a and b, differ only in the sign of the radius change (+20% and -20%).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from . import episode as E
from . import physics as P
from .pilot import ReferencePilot

GRID_STEP = 0.5               # seconds, certificate grid
# Conditional empirical error budgets, not validated enclosures. Each inequality receives
# one TOTAL allowance covering up to two positions plus norm/arithmetic evaluation.
POSITION_ALLOWANCE = 4e-7     # NM per evaluated tanker position
NORM_ARITHMETIC_ALLOWANCE = 2e-7  # NM per inequality (norm, products and subtraction)
NUMERIC_ALLOWANCE = 1e-6      # NM total per inequality; >= 2*POSITION + NORM_ARITHMETIC
assert NUMERIC_ALLOWANCE >= 2 * POSITION_ALLOWANCE + NORM_ARITHMETIC_ALLOWANCE
BLACKOUT_RANGE = (20, 59)     # t_b, whole seconds (proposed)
WINDS = [(s, d) for s in (20, 40, 60) for d in (0, 90, 180, 270)]   # knots, from-direction degrees
DISTANCES = (10, 15, 20, 25, 30)


@dataclass(frozen=True)
class Draw:
    seed: int
    world_a: E.Scenario       # radius grows
    world_b: E.Scenario       # radius shrinks
    blackout: int
    generous_speed: float     # NM/s, airspeed plus wind speed


def wind_vector(speed_kt: float, from_deg: float) -> tuple[float, float]:
    """The velocity the air moves with, NM/s. Wind 'from 270' blows towards the east."""
    to = math.radians((from_deg + 180.0) % 360.0)
    return P.kt(speed_kt) * math.sin(to), P.kt(speed_kt) * math.cos(to)


def draw(seed: int) -> Draw:
    rng = random.Random(seed)
    tas = rng.uniform(420, 480)
    rcv = P.ReceiverState(0.0, rng.uniform(-5, 5), rng.uniform(-5, 5), math.radians(90), math.radians(90),
                          P.kt(tas), P.turn_rate(tas))
    wspeed, wfrom = rng.choice(WINDS)
    cx, cy = rng.uniform(50, 80), rng.uniform(35, 60)
    r0, v = rng.uniform(6, 10), P.kt(rng.uniform(260, 300))
    theta0 = rng.uniform(0, 2 * math.pi)
    sign = rng.choice((-1, 1))
    c2 = (rng.uniform(50, 80), cy + sign * rng.uniform(44, 56))
    shell2 = E.Tanker("SHELL 2", P.Orbit(c2[0], c2[1], rng.uniform(6, 10), P.kt(rng.uniform(260, 300)),
                                         rng.uniform(0, 2 * math.pi)))
    blackout = rng.randint(*BLACKOUT_RANGE)
    goal = E.Goal(1, "SHELL 1", float(rng.choice(DISTANCES)))
    worlds = []
    for delta in (0.2 * r0, -0.2 * r0):
        shell1 = E.Tanker("SHELL 1", P.Orbit(cx, cy, r0, v, theta0, delta))
        worlds.append(E.Scenario(episode_id=f"dev-{seed}", receiver=rcv, wind=wind_vector(wspeed, wfrom),
                                 tankers=(shell1, shell2), goals=((0, goal),),
                                 dropouts=(("SHELL 1", blackout, None),)))
    return Draw(seed, worlds[0], worlds[1], blackout, P.kt(tas + wspeed))


def certificate_limit(d: Draw, horizon: float = 1800.0, e: float = E.JOIN_TOLERANCE) -> dict[str, Any]:
    """Latest deadline T for which no single path can reach both worlds' tanker by T, with margins.

    Necessary conditions for a path from x0 at time 0 that comes within e of world
    first's tanker at t1 and of world second's at t2 >= t1:
      1. |p_first(t1) - x0| <= S t1 + e
      2. |p_second(t2) - p_first(t1)| <= S (t2 - t1) + 2e
    On a grid of step h, condition 1's slack changes by at most (S + V1) h/2 and
    condition 2's by at most (2S + V1 + V2) h/2 between a point and its nearest grid
    point, so a grid pair 'possibly feasible' within those margins covers every
    continuous pair, including pairs whose grid points round into the opposite order.
    The certificate holds for every T below the earliest possibly feasible t2, less h.
    """
    S, h = d.generous_speed, GRID_STEP
    x0 = np.array([d.world_a.receiver.x, d.world_a.receiver.y])
    ts = np.arange(0.0, horizon + h / 2, h)
    orbits = {"a": d.world_a.tankers[0].orbit, "b": d.world_b.tankers[0].orbit}
    pos = {k: np.array([o.position(float(t)) for t in ts]) for k, o in orbits.items()}
    earliest = {}
    for first, second in (("a", "b"), ("b", "a")):
        v1, v2 = orbits[first].v, orbits[second].v
        m1 = (S + v1) * h / 2 + NUMERIC_ALLOWANCE
        m2 = (2 * S + v1 + v2) * h / 2 + NUMERIC_ALLOWANCE
        ok1 = np.linalg.norm(pos[first] - x0, axis=1) - S * ts - e <= m1
        found = None
        for i in np.nonzero(ok1)[0]:
            j0 = max(i - 1, 0)       # grid rounding can swap two nearly simultaneous visits
            gaps = np.linalg.norm(pos[second][j0:] - pos[first][i], axis=1) - S * (ts[j0:] - ts[i]) - 2 * e
            hit = np.nonzero(gaps <= m2)[0]
            if hit.size:
                t2 = ts[j0 + hit[0]]
                found = t2 if found is None else min(found, t2)
            if found is not None and ts[i] >= found:
                break
        earliest[first + second] = found
    finite = [v for v in earliest.values() if v is not None]
    limit = (min(finite) - h) if finite else horizon
    deadline = math.floor(limit)
    margins = strict_grid_margins(d, deadline, ts, pos, e)
    return {"earliest_possible_t2": earliest, "deadline_limit": limit,
            "checked_deadline": deadline, "strict_margin_nm": margins,
            "settings": {"grid_step_s": h, "horizon_s": horizon, "join_tolerance_nm": e,
                         "generous_speed_nm_s": S, "position_allowance_nm": POSITION_ALLOWANCE,
                         "norm_arithmetic_allowance_nm": NORM_ARITHMETIC_ALLOWANCE,
                         "total_inequality_allowance_nm": NUMERIC_ALLOWANCE,
                         "numerical_qualification": "conditional on empirical budgets, not validated bounds"}}


def strict_grid_margins(d: Draw, deadline: int, ts: np.ndarray | None = None,
                        pos: dict[str, np.ndarray] | None = None,
                        join_tolerance_nm: float = E.JOIN_TOLERANCE) -> dict[str, float | None]:
    """Smallest strict violation of at least one necessary inequality, for each visit order.

    Exhaust the time square through the integer deadline, with j >= i-1. No
    first-reach pruning/early-exit search is used. Positive means every covered
    pair fails at least one relaxed inequality, conditional on the error budgets.
    """
    if deadline < 0:
        return {"ab": None, "ba": None}
    h, S = GRID_STEP, d.generous_speed
    orbits = {"a": d.world_a.tankers[0].orbit, "b": d.world_b.tankers[0].orbit}
    if ts is None:
        ts = np.arange(0, deadline + h / 2, h)
    if pos is None:
        pos = {k: np.array([o.position(float(t)) for t in ts]) for k, o in orbits.items()}
    use = ts <= deadline
    times = ts[use]; positions = {k: p[use] for k, p in pos.items()}
    x0 = np.array([d.world_a.receiver.x, d.world_a.receiver.y])
    margins = {}
    for first, second in (("a", "b"), ("b", "a")):
        v1, v2 = orbits[first].v, orbits[second].v
        m1 = (S + v1) * h / 2 + NUMERIC_ALLOWANCE
        m2 = (2*S + v1 + v2) * h / 2 + NUMERIC_ALLOWANCE
        one = np.linalg.norm(positions[first] - x0, axis=1) - S*times - join_tolerance_nm - m1
        smallest = math.inf
        for i, t1 in enumerate(times):
            j = max(0, i-1)
            two = (np.linalg.norm(positions[second][j:] - positions[first][i], axis=1)
                   - S*(times[j:] - t1) - 2*join_tolerance_nm - m2)
            smallest = min(smallest, float(np.min(np.maximum(one[i], two))))
        margins[first+second] = smallest
    return margins


def fly(scenario: E.Scenario, ask_help: bool, max_steps: int = 1800, retain_trace: bool = False) -> dict[str, Any]:
    ep = E.Episode(scenario)
    start = ep.reset()
    pilot = ReferencePilot(start["briefing"], ask_help=ask_help, horizon=float(max_steps))
    bundle = start["bundle"]
    trace = []
    for _ in range(max_steps):
        action = pilot.act(bundle)
        if retain_trace:
            trace.append({"bundle": bundle, "action": action})
        bundle = ep.step(action)
        if bundle is None:
            break
    return ep.outcome() | {"joined_at": ep.record.joined_at,
                           "trace": {"briefing": start["briefing"], "steps": trace} if retain_trace else None}


def assess(seed: int) -> dict[str, Any]:
    return assess_draw(draw(seed))


def assess_draw(d: Draw) -> dict[str, Any]:
    """assess() for a draw made elsewhere: the stream generator draws its phase 5 episodes itself
    (study2/world/streams.py, help_episode)."""
    cert = certificate_limit(d)
    witness = {k: fly(w, ask_help=True, retain_trace=True) for k, w in (("a", d.world_a), ("b", d.world_b))}
    joins = [w["joined_at"] for w in witness.values()]
    met = all(w["goal_met"] for w in witness.values())
    record = {"seed": d.seed, "blackout": d.blackout, "certificate": cert,
              "witness": witness}
    if not all(w["settled"] for w in witness.values()):
        return record | {"label": "unknown", "reason": "with-help witness has numerical flags or is unfinished"}
    if not met or any(j is None for j in joins):
        return record | {"label": "unknown", "reason": "with-help witness failed in at least one world"}
    deadline = math.floor(cert["deadline_limit"])
    if any(m is None or m <= 0 for m in cert["strict_margin_nm"].values()):
        return record | {"label": "unknown", "reason": "no strict conditional certificate margin", "deadline": deadline}
    if deadline < max(joins):
        return record | {"label": "unknown", "reason": "no deadline window", "window_s": cert["deadline_limit"] - max(joins)}
    confirm = {k: fly(replace(w, goals=((0, replace(w.goals[0][1], deadline=deadline)),)), ask_help=True, retain_trace=True)
               for k, w in (("a", d.world_a), ("b", d.world_b))}
    record["confirmation"] = confirm
    if not all(c["goal_met"] and c["settled"] for c in confirm.values()):
        return record | {"label": "unknown", "reason": "witness failed once the deadline was set", "deadline": deadline}
    return record | {"label": "help certified", "deadline": deadline, "window_s": cert["deadline_limit"] - max(joins)}
