"""Stream generation for Experiment 2, phases 1 to 5 (MVP; Experiment 2 protocol section 4, DISTURBANCES.md sections
3 to 7, 9, 11 and 13). Call episode() for any phase.

Phase 3 has four blocks of 5 episodes, each with one of the 12 windy settings: indices 0 to 4 an orbit change alone,
5 to 9 a dropout alone, 10 to 14 late information alone, 15 to 19 all three together. Its draws follow phase 2's,
then draw the disturbances, so a phase 3 episode's receiver, tankers, wind and goal come from the same calls as a
phase 2 draw would make. The settings, drawn uniformly, are DISTURBANCES.md's:
- orbit change: SHELL 1's radius changes by a fraction of r0 drawn from -20% to +20%, starting at 60 s over 120 s,
  holding total ground speed (section 4); SHELL 2 never changes;
- dropout: SHELL 1's reports stop for 5 or 30 seconds, starting at a whole second from 60 to 600 (section 7);
- late information: every report is delayed by 5, 30 or 120 seconds (section 7; 0 is phase 1 and 2's no delay).
The orbit change's size is drawn over its whole range, so a change can be small. That reading of 'up to 20% either
way' is a declared generation choice, recorded in DISTURBANCES.md section 13.

Phase 4 repeats phase 3's blocks and draws, then adds one goal change (paper2/notes/PHASE4-GOAL-CHANGE-PROPOSAL.md,
as corrected in Codex's TURN 137 review). With x0 the receiver's start, S its true airspeed plus the wind speed, and
for each tanker its centre c and a conservative upper bound r_up on its radius (r0 + |delta|), the distance from the
aircraft to any point of that orbit at time t is at least L(t) = |c - x0| - S*t - r_up, whatever it has flown. The
change time is a whole second drawn uniformly from 60 to t_max, where t_max is the last whole second strictly before
L for SHELL 1 can reach the join tolerance, less 10 s, so no path can have joined before the goal changes. With
probability one half the new goal keeps SHELL 1 with another reporting distance, otherwise it names SHELL 2; the
distance is drawn uniformly from the reporting distances no greater than L(t_g) - 3 for the new goal's tanker (for
SHELL 1, other than the current one). That is section 9's inbound rule checked against every possible path at the
moment the goal changes; it does not guarantee that every path can still report and join later, which the reference
pilot's flight checks for the drawn scenario as in every phase. An empty window or an empty set of distances makes
the draw unusable; it is replaced like an unsolvable one, and the attempt keeps its reason and details. The uniform
time and the even split are distributions over attempts; replacement changes the accepted mixture, so the attempts
and realised cases are reported.

Each episode's identifier is its stream, phase and index; its seed is derived from those and the attempt number, and
every draw uses NumPy's Generator with the PCG64 bit generator, created afresh from that seed. Phase 1 is still air;
phase 2 has one of the 12 windy settings, drawn uniformly. The receiver, the named tanker SHELL 1 and the second tanker
SHELL 2 are drawn from the ranges of DISTURBANCES.md sections 3 and 4, with no orbit change, dropout or late
information in these phases. Every episode must be reference-pilot solvable: the observation-only reference pilot
meets the goal without help. A draw that fails is replaced, at most 500 attempts per episode, and every attempt is
kept with its reason and the reference pilot's whole outcome. Only a settled success certifies a draw. Generation stops
with an error at the cap, and the error keeps every attempt.

Phase 5 has the help cases of DISTURBANCES.md section 11 (help_episode). A draw makes the same receiver, tankers, wind
and reporting distance as a phase 2 draw, in the same order, then SHELL 1's blackout second t_b (20 to 59, its reports
stop for good) and which of the two worlds is flown. World a's SHELL 1 radius grows by 20% from 60 s and world b's
shrinks by 20%; the two are identical in everything the aircraft can observe. world/phase5.py's assess_draw then
certifies the draw: no path flown without help can join both worlds' tanker by the deadline, and the reference pilot
with help meets the whole goal by it in both. A draw that is not certified is replaced, up to the same 500 attempts,
and every attempt is kept with its reason and assessment. The accepted episode is the flown world with the
certificate's deadline. Its full assessment, flight traces included, is kept apart (Generated.witness), because each
trace runs to hundreds of kilobytes; the attempt records keep a SHA-256 and step count for every trace.

Layout v2 (paper2/notes/STUDY-DECISIONS-CLAUDE-20261026.md, revisions 2 and 3, agreed with Codex in TURN241) changes
phase 3 only, so that recovery after a learning boundary can be observed: indices 0 to 9 carry an orbit change and 10 to
19 late information, each with wind, and no dropouts. Each half-block of five late-information episodes carries the
delays 5, 30, 30, 120 and 120 seconds, in an order drawn once per stream and half-block from its own seed, so retries
never change an episode's delay and every stream gets the same mixture. Phases 1, 2, 4 and 5 are as in v1. The default
layout stays v1, so historical streams and their saved-world replays draw exactly as before.

Development draft for the MVP; the generator's fingerprint is recorded at the protocol freeze, not here.
"""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

import telohelm.taskpacks.tanker.world.episode as E
import telohelm.taskpacks.tanker.world.physics as P
import telohelm.taskpacks.tanker.world.phase5 as phase5

ATTEMPT_CAP = 500
DISTANCES = phase5.DISTANCES                 # the report distances, nautical miles
WINDS = phase5.WINDS                         # the 12 windy settings: (knots, from-direction degrees)
GENERATED_PHASES = (1, 2, 3, 4)              # drawn by draw(); phase 5 by draw_help()
HELP_PHASE = 5
PHASE3_BLOCKS = ("orbit", "dropout", "delay", "all")      # 5 episodes each, in this order
LAYOUTS = ("v1", "v2")
PHASE3_V2_BLOCKS = ("orbit", "delay")                     # layout v2: 10 episodes each, in this order
V2_DELAY_HALF = (5, 30, 30, 120, 120)                       # layout v2: the delays of each half-block of five
ORBIT_CHANGE_FRACTION = 0.2
DROPOUT_SECONDS = (5, 30)
DROPOUT_ONSET = (60, 600)
DELAYS = (5, 30, 120)
GOAL_CHANGE_FROM = 60                 # earliest goal change, seconds
GOAL_CHANGE_MARGIN = 10               # seconds kept between the latest change and the earliest possible join
INBOUND_MARGIN_NM = 3.0               # DISTURBANCES.md section 9


class GenerationError(RuntimeError):
    """An episode could not be drawn within its attempt cap. `attempts` keeps every attempt made, in order, with its
    seed, reason, details and the reference pilot's whole outcome where it flew (Codex's GEN139.CAP)."""

    def __init__(self, message: str, attempts: list[dict] | None = None):
        super().__init__(message)
        self.attempts = list(attempts or [])


class Unusable(ValueError):
    """A draw that cannot be used, for a recorded reason; it is replaced like an unsolvable one."""

    def __init__(self, reason: str, details: dict):
        super().__init__(reason)
        self.reason, self.details = reason, details


def episode_id(stream: int, phase: int, index: int) -> str:
    return f"s{stream}-p{phase}-e{index:02d}"


def episode_seed(stream: int, phase: int, index: int, attempt: int) -> int:
    digest = hashlib.sha256(f"telohelm-stream|{stream}|{phase}|{index}|{attempt}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def v2_delay(stream: int, index: int) -> int:
    """Layout v2's delay for phase 3 index 10 to 19: its place in a per-stream, per-half-block order of V2_DELAY_HALF."""
    if not 10 <= index < 20:
        raise ValueError("layout v2 delays belong to phase 3 indices 10 to 19")
    digest = hashlib.sha256(f"telohelm-stream-v2-delays|{stream}|{index // 5}".encode()).digest()
    order = np.random.Generator(np.random.PCG64(int.from_bytes(digest[:8], "big"))).permutation(len(V2_DELAY_HALF))
    return V2_DELAY_HALF[int(order[index % 5])]


def draw(stream: int, phase: int, index: int, attempt: int = 0, layout: str = "v1") -> E.Scenario:
    """One candidate scenario for a phase 1 to 4 episode. It is not yet checked for solvability. Raises Unusable
    for a phase 4 draw with no goal-change window or no eligible new goal."""
    if phase not in GENERATED_PHASES:
        raise GenerationError(f"phase {phase} is not generated yet")
    if layout not in LAYOUTS:
        raise ValueError(f"unknown stream layout {layout!r}")
    rng = np.random.Generator(np.random.PCG64(episode_seed(stream, phase, index, attempt)))
    tas = rng.uniform(420, 480)
    rcv = P.ReceiverState(0.0, rng.uniform(-5, 5), rng.uniform(-5, 5), math.radians(90), math.radians(90),
                          P.kt(tas), P.turn_rate(tas))
    cx, cy = rng.uniform(50, 80), rng.uniform(35, 60)
    shell1 = E.Tanker("SHELL 1", P.Orbit(cx, cy, rng.uniform(6, 10), P.kt(rng.uniform(260, 300)),
                                         rng.uniform(0, 2 * math.pi)))
    north = rng.integers(0, 2) == 1
    c2y = cy + (1 if north else -1) * rng.uniform(44, 56)
    shell2 = E.Tanker("SHELL 2", P.Orbit(rng.uniform(50, 80), c2y, rng.uniform(6, 10), P.kt(rng.uniform(260, 300)),
                                         rng.uniform(0, 2 * math.pi)))
    if phase == 1:
        wind = (0.0, 0.0)
    else:
        speed, frm = WINDS[int(rng.integers(0, len(WINDS)))]
        wind = phase5.wind_vector(speed, frm)
    goal = E.Goal(1, "SHELL 1", float(DISTANCES[int(rng.integers(0, len(DISTANCES)))]))
    delay, dropouts = 0, ()
    if phase in (3, 4):
        block = PHASE3_V2_BLOCKS[index // 10] if layout == "v2" and phase == 3 else PHASE3_BLOCKS[index // 5]
        if block in ("orbit", "all"):
            o = shell1.orbit
            delta = o.r0 * rng.uniform(-ORBIT_CHANGE_FRACTION, ORBIT_CHANGE_FRACTION)
            shell1 = E.Tanker("SHELL 1", P.Orbit(o.cx, o.cy, o.r0, o.v, o.theta0, delta))
        if block in ("dropout", "all"):
            length = int(DROPOUT_SECONDS[int(rng.integers(0, len(DROPOUT_SECONDS)))])
            onset = int(rng.integers(DROPOUT_ONSET[0], DROPOUT_ONSET[1] + 1))
            dropouts = (("SHELL 1", onset, onset + length),)
        if block in ("delay", "all"):
            delay = int(DELAYS[int(rng.integers(0, len(DELAYS)))])
            if layout == "v2" and phase == 3:
                delay = v2_delay(stream, index)
    goals: tuple = ((0, goal),)
    if phase == 4:
        goals = goals + ((*_goal_change(rng, rcv, wind, shell1, shell2, goal),),)
    return E.Scenario(episode_id=episode_id(stream, phase, index), receiver=rcv, wind=wind,
                      tankers=(shell1, shell2), goals=goals, delay_s=delay, dropouts=dropouts)


def _goal_change(rng: np.random.Generator, rcv: P.ReceiverState, wind: tuple[float, float], shell1: E.Tanker,
                 shell2: E.Tanker, goal: E.Goal) -> tuple[int, E.Goal]:
    """Phase 4's goal change: (second, new goal). See the module notes."""
    speed = rcv.tas + math.hypot(*wind)

    def lower(t: Any, k: E.Tanker) -> float:
        o = k.orbit
        return math.hypot(o.cx - rcv.x, o.cy - rcv.y) - speed * t - (o.r0 + abs(o.delta))

    tau = lower(0.0, shell1) - E.JOIN_TOLERANCE
    tau = tau / speed                                    # L for SHELL 1 reaches the tolerance at tau
    t_max = math.ceil(tau) - 1 - GOAL_CHANGE_MARGIN
    details = {"t_max": t_max, "tau": tau}
    if t_max < GOAL_CHANGE_FROM:
        raise Unusable("no goal-change window", details)
    t_g = int(rng.integers(GOAL_CHANGE_FROM, t_max + 1))
    new_tanker = shell2 if int(rng.integers(0, 2)) == 1 else shell1
    bound = lower(float(t_g), new_tanker) - INBOUND_MARGIN_NM
    eligible = [d for d in DISTANCES if d <= bound and not (new_tanker is shell1 and d == goal.report_at_nm)]
    details.update({"t_g": t_g, "tanker": new_tanker.callsign, "bound_nm": bound, "eligible_nm": eligible})
    if not eligible:
        raise Unusable("no eligible new goal", details)
    distance = float(eligible[int(rng.integers(0, len(eligible)))])
    return t_g, E.Goal(goal.version + 1, new_tanker.callsign, distance)


@dataclass
class Generated:
    scenario: E.Scenario
    attempts: list[dict] = field(default_factory=list)
    witness: dict | None = None          # phase 5 only: the accepted draw's full assessment, traces included
    layout: str = "v1"


def episode(stream: int, phase: int, index: int, layout: str = "v1") -> Generated:
    """The accepted draw for an episode of any phase: help certified in phase 5, reference-pilot solvable otherwise."""
    if layout not in LAYOUTS:
        raise ValueError(f"unknown stream layout {layout!r}")
    if phase == HELP_PHASE:
        return replace(help_episode(stream, index), layout=layout)
    return solvable_episode(stream, phase, index, layout)


def draw_help(stream: int, index: int, attempt: int = 0) -> tuple[phase5.Draw, str]:
    """One candidate phase 5 draw, not yet certified, and the world flown ("a" or "b"). See the module notes."""
    seed = episode_seed(stream, HELP_PHASE, index, attempt)
    rng = np.random.Generator(np.random.PCG64(seed))
    tas = rng.uniform(420, 480)
    rcv = P.ReceiverState(0.0, rng.uniform(-5, 5), rng.uniform(-5, 5), math.radians(90), math.radians(90),
                          P.kt(tas), P.turn_rate(tas))
    cx, cy = rng.uniform(50, 80), rng.uniform(35, 60)
    r0, v, theta0 = rng.uniform(6, 10), P.kt(rng.uniform(260, 300)), rng.uniform(0, 2 * math.pi)
    north = rng.integers(0, 2) == 1
    c2y = cy + (1 if north else -1) * rng.uniform(44, 56)
    shell2 = E.Tanker("SHELL 2", P.Orbit(rng.uniform(50, 80), c2y, rng.uniform(6, 10), P.kt(rng.uniform(260, 300)),
                                         rng.uniform(0, 2 * math.pi)))
    speed, frm = WINDS[int(rng.integers(0, len(WINDS)))]
    goal = E.Goal(1, "SHELL 1", float(DISTANCES[int(rng.integers(0, len(DISTANCES)))]))
    blackout = int(rng.integers(phase5.BLACKOUT_RANGE[0], phase5.BLACKOUT_RANGE[1] + 1))
    flown = "a" if int(rng.integers(0, 2)) == 0 else "b"
    worlds = [E.Scenario(episode_id=episode_id(stream, HELP_PHASE, index), receiver=rcv,
                         wind=phase5.wind_vector(speed, frm),
                         tankers=(E.Tanker("SHELL 1", P.Orbit(cx, cy, r0, v, theta0, delta)), shell2),
                         goals=((0, goal),), dropouts=(("SHELL 1", blackout, None),))
              for delta in (ORBIT_CHANGE_FRACTION * r0, -ORBIT_CHANGE_FRACTION * r0)]
    return phase5.Draw(seed, worlds[0], worlds[1], blackout, P.kt(tas + speed)), flown


def _digest_traces(assessment: dict) -> dict:
    """The assessment with each flight trace replaced by its SHA-256 and step count."""
    out = deepcopy(assessment)
    for part in ("witness", "confirmation"):
        for flight in (out.get(part) or {}).values():
            trace = flight.get("trace")
            if trace is not None:
                text = json.dumps(trace, sort_keys=True, default=str)
                flight["trace"] = {"sha256": hashlib.sha256(text.encode()).hexdigest(), "steps": len(trace["steps"])}
    return out


def help_episode(stream: int, index: int) -> Generated:
    """The first help-certified phase 5 draw for this episode, with every attempt and its assessment."""
    attempts = []
    for attempt in range(ATTEMPT_CAP):
        d, flown = draw_help(stream, index, attempt)
        a = phase5.assess_draw(d)
        certified = a["label"] == "help certified"
        attempts.append({"attempt": attempt, "seed": d.seed, "help_certified": certified,
                         "reason": None if certified else a.get("reason"), "flown": flown,
                         "assessment": _digest_traces(a)})
        if certified:
            w = d.world_a if flown == "a" else d.world_b
            goal = replace(w.goals[0][1], deadline=a["deadline"])
            return Generated(replace(w, goals=((0, goal),)), attempts, witness=a)
    raise GenerationError(f"{episode_id(stream, HELP_PHASE, index)}: no help-certified draw in {ATTEMPT_CAP} attempts",
                          attempts)


def solvable_episode(stream: int, phase: int, index: int, layout: str = "v1") -> Generated:
    """The first reference-pilot-solvable draw for this episode, with every attempt and its outcome."""
    attempts = []
    for attempt in range(ATTEMPT_CAP):
        seed = episode_seed(stream, phase, index, attempt)
        try:
            # v1 keeps its historical call, which tests and replays replace with four-argument draws.
            s = draw(stream, phase, index, attempt) if layout == "v1" else draw(stream, phase, index, attempt, layout)
        except Unusable as u:
            attempts.append({"attempt": attempt, "seed": seed, "solvable": False, "reason": u.reason,
                             "details": u.details})
            continue
        outcome = phase5.fly(s, ask_help=False)
        # Certified only by a settled success: a nominal success the world could not settle numerically is not
        # certified, and the draw is replaced with the whole outcome kept (Codex's GEN139.UNSETTLED).
        met, settled = outcome.get("goal_met") is True, outcome.get("settled") is True
        ok = met and settled
        # Settledness is judged first: an unsettled outcome is unknown whichever way its nominal result went
        # (Codex's GEN141.UNKNOWN); only a settled failure is a reference failure.
        reason = None if ok else ("not solvable by the reference pilot" if settled else "reference outcome not settled")
        attempts.append({"attempt": attempt, "seed": seed, "solvable": ok, "reason": reason, "outcome": outcome})
        if ok:
            return Generated(s, attempts, layout=layout)
    raise GenerationError(f"{episode_id(stream, phase, index)}: no solvable draw in {ATTEMPT_CAP} attempts", attempts)


def schedule(stream: int, phases: tuple[int, ...] = GENERATED_PHASES, per_phase: int = 20) -> list[Generated]:
    """A stream's episodes in order: per_phase episodes of each phase."""
    return [episode(stream, p, i) for p in phases for i in range(per_phase)]


def generation_record(g: Generated) -> dict:
    """What generation did for one episode, for keeping beside a collection: the accepted episode and every attempt
    with its seed, reason, details and the reference pilot's whole outcome."""
    row = {"episode": g.scenario.episode_id, "attempts": g.attempts}
    if g.layout != "v1":                  # v1 records keep their historical bytes
        row["layout"] = g.layout
    return row


def cap_record(stream: int, phase: int, index: int, error: GenerationError) -> dict:
    """The record kept when generation stops at the cap: the scheduled episode, every attempt and the error."""
    return {"episode": episode_id(stream, phase, index), "attempts": error.attempts, "accepted": False,
            "error": str(error)}
