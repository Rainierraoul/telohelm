"""Development cases for the goal monitor, labelled relative to the reference pilot (Simon, 14 October 2026).

    study2/.venv/bin/python -m study2.devcases --stream 4401 --out <dir> [--rewrites 20]

Why: until the monitor holds cases, no rewritten check can pass the replay test, so no procedure can be adopted and
Telohelm cannot learn in a development run. The scored cases of note B8 need proved labels and are not ready. This
module builds a separate development set so that a development stream can run end to end now. It never touches a
scored stream: the store it writes is a development store (hidden_set/store.py), which accepts only development
streams 4400 to 4499 and rules beginning 'dev.', and a scored store refuses both.

The labels use B8's named alternative, 'wrong relative to the reference pilot', under rule dev.reference.v1. The
reference pilot (world/pilot.py) sees only what Telohelm sees. In episodes it flew to the goal, with the world's
outcome settled, its own heading at a second is labelled right, and these are labelled wrong:
- geometry: its heading turned by 90 to 270 degrees;
- identity: the heading a reference pilot would fly to the other tanker, when that differs by 45 degrees or more;
- stale: the bearing straight at the goal tanker's newest report when that report is over 10 seconds old, when that
  differs by 30 degrees or more;
- goal_change: after a goal change, the heading a reference pilot still following the first goal would fly, when that
  differs by 45 degrees or more.
For the report: a trigger at the ordered distance while the observed range is still more than 0.5 NM above it, doing
nothing while it is more than 5 NM above, and the reference's own report at the second it makes it are right; a
trigger 4 or 6 NM away from the ordered distance, the correct report sent while the range is more than 3 NM above the
ordered distance, and a report naming the other tanker at the reference's second are wrong (kind identity).

None of this is a proof of the kind B8 requires. A one-second heading costs little time, so a heading 'wrong relative
to the reference' may not be wrong in B8's sense. The set exists to exercise learning, adoption and undo in
development, and nothing measured with it is an experiment result.

Cases come from episodes of other development streams, never the one being flown: the stored and right cases from one
source stream, each rewrite's fresh cases from further ones. Each case is the exact context the live check receives
(runtime/context.py) and the output a procedure would return.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import telohelm.qualification.store as H
from telohelm.runtime.context import ContextBuilder
import telohelm.taskpacks.tanker.world.episode as E
from . import streams as streams
from telohelm.taskpacks.tanker.world.pilot import ReferencePilot

RULE = "dev.reference.v1"
VERIFIER_NAME = "development consistency check of dev.reference.v1"
HEADING_KINDS = ("geometry", "identity", "stale", "goal_change")
GEOMETRY_TURNS = (90.0, 135.0, 180.0, 225.0, 270.0)
IDENTITY_MIN_DEG, STALE_MIN_DEG, GOAL_CHANGE_MIN_DEG = 45.0, 30.0, 45.0
STALE_AGE_S = 10.0
TRIGGER_OFFSETS_NM = (-6.0, -4.0, 4.0, 6.0)
TRIGGER_RIGHT_MARGIN_NM, IDLE_MARGIN_NM, EARLY_REPORT_MARGIN_NM = 0.5, 5.0, 3.0
STRIDE_S = 5                      # one moment in every 5 seconds of a reference flight
STORED_WRONG_PER_DECISION = 40    # 10 of each heading kind; 40 report cases of kind identity


def angdiff(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def bearing(x0: float, y0: float, x1: float, y1: float) -> float:
    return math.degrees(math.atan2(x1 - x0, y1 - y0)) % 360.0


@dataclass
class Moment:
    """One second of a reference flight: the exact context and what the reference and its shadows would do."""
    episode: str
    phase: int
    t: int
    goal: dict
    ctx: dict
    reference: dict                # the reference pilot's action at this second
    other_heading: float | None    # a reference pilot flying to the other tanker
    first_goal_heading: float | None  # a reference pilot still following the first goal
    reported: bool                 # whether the reference has already reported under this goal version
    other_tanker: str
    control: str
    source: int                    # the development stream the episode was drawn from


@dataclass
class Flight:
    episode: str
    phase: int
    goal_met: bool
    settled: bool
    moments: list[Moment] = field(default_factory=list)


def _shadow(briefing: dict, tanker: str) -> ReferencePilot:
    b = copy.deepcopy(briefing)
    b["goal"] = {**b["goal"], "tanker": tanker}
    return ReferencePilot(b, ask_help=False)


def fly(scenario: E.Scenario, phase: int, source: int, max_steps: int = 3600) -> Flight:
    """Fly the reference pilot, building Telohelm's context at each second exactly as the episode loop does."""
    ep = E.Episode(scenario)
    start = ep.reset()
    briefing = start["briefing"]
    pilot = ReferencePilot(briefing, ask_help=False)
    callsigns = [k["callsign"] for k in briefing["tankers"]]
    first_tanker = briefing["goal"]["tanker"]
    # One shadow reference pilot per tanker, flying from the real aircraft's state and seeing every report, but never
    # told of a goal change: each gives the intercept heading to its own tanker.
    shadows = {c: _shadow(briefing, c) for c in callsigns}
    builder = ContextBuilder(briefing)
    bundle = start["bundle"]
    moments, reported_versions = [], set()
    for _ in range(max_steps):
        ctx = copy.deepcopy(builder.update(bundle))
        goal = copy.deepcopy(builder.goal)
        blind = {**bundle, "goal": None}
        headings = {c: shadows[c].act(blind)["heading_deg"] for c in callsigns}
        other_now = next(c for c in callsigns if c != goal["tanker"])
        action = pilot.act(bundle)
        moments.append(Moment(scenario.episode_id, phase, ctx["t"], goal, ctx, dict(action), headings[other_now],
                              headings[first_tanker] if goal["version"] > briefing["goal"]["version"] else None,
                              goal["version"] in reported_versions, other_now, briefing["control_callsign"], source))
        if "report" in action:
            reported_versions.add(goal["version"])
        bundle = ep.step(action)
        if bundle is None:
            break
    out = ep.outcome()
    return Flight(scenario.episode_id, phase, out.get("goal_met") is True, out.get("settled") is True, moments)


def flights(source_stream: int, phases: tuple[int, ...] = (1, 2, 3, 4), per_phase: int = 20) -> Iterator[Flight]:
    for p in phases:
        for i in range(per_phase):
            try:
                g = streams.solvable_episode(source_stream, p, i)
            except streams.GenerationError:
                continue
            yield fly(g.scenario, p, source_stream)


# -- labelled candidates ----------------------------------------------------------------------------------------

def heading_candidates(m: Moment) -> list[tuple[str, str, dict, dict]]:
    """(disposition, kind, output, evidence) for one moment's heading decision."""
    ref = m.reference["heading_deg"]
    out = [("right", None, {"heading_deg": ref}, {"construction": "reference", "reference_heading": ref})]
    turn = GEOMETRY_TURNS[int(hashlib.sha256(f"{m.episode}|{m.t}".encode()).hexdigest(), 16) % len(GEOMETRY_TURNS)]
    out.append(("wrong", "geometry", {"heading_deg": (ref + turn) % 360.0},
                {"construction": "reference_turned", "reference_heading": ref, "turn_deg": turn}))
    if m.other_heading is not None and angdiff(m.other_heading, ref) >= IDENTITY_MIN_DEG:
        out.append(("wrong", "identity", {"heading_deg": m.other_heading},
                    {"construction": "other_tanker", "reference_heading": ref, "other_tanker": m.other_tanker,
                     "min_deg": IDENTITY_MIN_DEG}))
    age = m.ctx.get("report_age_s")
    if age is not None and age > STALE_AGE_S:
        h = bearing(m.ctx["x"], m.ctx["y"], m.ctx["report_x"], m.ctx["report_y"])
        if angdiff(h, ref) >= STALE_MIN_DEG:
            out.append(("wrong", "stale", {"heading_deg": h},
                        {"construction": "at_stale_report", "reference_heading": ref, "report_age_s": age,
                         "min_deg": STALE_MIN_DEG}))
    if m.first_goal_heading is not None and angdiff(m.first_goal_heading, ref) >= GOAL_CHANGE_MIN_DEG:
        out.append(("wrong", "goal_change", {"heading_deg": m.first_goal_heading},
                    {"construction": "first_goal", "reference_heading": ref, "min_deg": GOAL_CHANGE_MIN_DEG}))
    return out


def report_text(control: str, distance: float, tanker: str) -> str:
    return f"{control}, aircraft 2, {distance:g} NM from {tanker}"


def report_candidates(m: Moment) -> list[tuple[str, str, dict, dict]]:
    if m.reported:
        return []
    d, rng = m.goal["report_at_nm"], m.ctx.get("observed_range_nm")
    out = []
    if "report" in m.reference:
        out.append(("right", None, {"report": m.reference["report"]},
                    {"construction": "reference_report", "reference_report": m.reference["report"]}))
        out.append(("wrong", "identity", {"report": report_text(m.control, d, m.other_tanker)},
                    {"construction": "report_other_tanker", "reference_report": m.reference["report"],
                     "other_tanker": m.other_tanker, "control_callsign": m.control, "report_at_nm": d}))
        return out
    if rng is None:
        return out
    if rng > d + TRIGGER_RIGHT_MARGIN_NM:
        out.append(("right", None, {"report_trigger_nm": d},
                    {"construction": "trigger_at_order", "report_at_nm": d, "observed_range_nm": rng}))
        off = TRIGGER_OFFSETS_NM[int(hashlib.sha256(f"{m.episode}|{m.t}|r".encode()).hexdigest(), 16) % 4]
        if 0 < d + off <= 1000:
            out.append(("wrong", "identity", {"report_trigger_nm": d + off},
                        {"construction": "trigger_off_order", "report_at_nm": d, "offset_nm": off}))
    if rng > d + IDLE_MARGIN_NM:
        out.append(("right", None, {}, {"construction": "idle_far", "report_at_nm": d, "observed_range_nm": rng}))
    if rng > d + EARLY_REPORT_MARGIN_NM:
        out.append(("wrong", "identity", {"report": report_text(m.control, d, m.goal["tanker"])},
                    {"construction": "early_report", "report_at_nm": d, "observed_range_nm": rng,
                     "control_callsign": m.control}))
    return out


def verify(case: dict) -> bool:
    """Development consistency check: the output is what the recorded construction gives, and the recorded margin
    holds. It re-derives each label from its evidence and the stored context; it is not a proof (module notes)."""
    try:
        lab, ev, out, ctx = case["label"], case["label"]["evidence"], case["output"], case["context"]
        if lab["rule"] != RULE:
            return False
        c, disp = ev["construction"], lab["disposition"]
        goal = case["goal"]
        if case["decision"] == "heading":
            ref, h = ev["reference_heading"], out["heading_deg"]
            if c == "reference":
                return disp == "right" and h == ref
            if disp != "wrong":
                return False
            if c == "reference_turned":
                return ev["turn_deg"] in GEOMETRY_TURNS and abs(angdiff(h, (ref + ev["turn_deg"]) % 360.0)) < 1e-9
            if c == "other_tanker":
                return ev["other_tanker"] != goal["tanker"] and angdiff(h, ref) >= IDENTITY_MIN_DEG
            if c == "at_stale_report":
                stale = bearing(ctx["x"], ctx["y"], ctx["report_x"], ctx["report_y"])
                return (ctx["report_age_s"] > STALE_AGE_S and abs(angdiff(h, stale)) < 1e-9
                        and angdiff(h, ref) >= STALE_MIN_DEG)
            if c == "first_goal":
                return goal["version"] > 1 and angdiff(h, ref) >= GOAL_CHANGE_MIN_DEG
            return False
        d, rng = goal["report_at_nm"], ctx.get("observed_range_nm")
        if c == "reference_report":
            return disp == "right" and out == {"report": ev["reference_report"]}
        # Report texts are rebuilt in full from the recorded construction and compared (DEV161.REPORT).
        if c == "report_other_tanker":
            return (disp == "wrong" and ev["other_tanker"] != goal["tanker"] and ev["report_at_nm"] == d
                    and out == {"report": report_text(ev["control_callsign"], d, ev["other_tanker"])})
        if c == "trigger_at_order":
            return disp == "right" and out == {"report_trigger_nm": d} and rng > d + TRIGGER_RIGHT_MARGIN_NM
        if c == "trigger_off_order":
            return disp == "wrong" and ev["offset_nm"] in TRIGGER_OFFSETS_NM and out == {"report_trigger_nm": d + ev["offset_nm"]}
        if c == "idle_far":
            return disp == "right" and out == {} and rng > d + IDLE_MARGIN_NM
        if c == "early_report":
            return (disp == "wrong" and ev["report_at_nm"] == d and rng > d + EARLY_REPORT_MARGIN_NM
                    and out == {"report": report_text(ev["control_callsign"], d, goal["tanker"])})
        return False
    except (KeyError, TypeError):
        return False


# -- building the store ---------------------------------------------------------------------------------------------

def _case(stream: int, split: str, decision: str, kind: str | None, m: Moment, output: dict, disposition: str,
          evidence: dict, ordinal: int, source_stream: int, rewrite: int | None = None) -> dict:
    origin = {"episode": m.episode, "second": m.t, "generator": f"{RULE}:{evidence['construction']}",
              "seed": source_stream, "ordinal": ordinal, "attempt": 0, "phase": m.phase}
    if rewrite is not None:
        origin["rewrite"] = rewrite
    return {"schema": H.SCHEMA, "stream": stream, "split": split, "decision": decision, "kind": kind,
            "interface": "action", "goal": m.goal, "context": m.ctx, "output": output, "origin": origin,
            "label": {"disposition": disposition, "rule": RULE, "evidence": evidence,
                      "reason": f"{'right' if disposition == 'right' else 'wrong'} relative to the reference pilot "
                                f"({evidence['construction']}); development only, not a B8 proof"}}


def _pools(fl: list[Flight], decision: str) -> tuple[list, list]:
    """Right and wrong candidates of one decision from settled, goal-meeting flights, as (moment, disposition, kind,
    output, evidence)."""
    right, wrong = [], []
    for f in fl:
        if not (f.goal_met and f.settled):
            continue
        strided = f.moments[::STRIDE_S]
        for m in strided:
            for disp, kind, out, ev in (heading_candidates(m) if decision == "heading" else report_candidates(m)):
                (right if disp == "right" else wrong).append((m, disp, kind, out, ev))
        if decision == "report":     # the reference's report second is rarely on the stride
            for m in f.moments:
                if "report" in m.reference and all(m is not x for x in strided):
                    for disp, kind, out, ev in report_candidates(m):
                        (right if disp == "right" else wrong).append((m, disp, kind, out, ev))
    return right, wrong


def _pick_wrong(wrong: list, n: int, rng: random.Random, decision: str) -> list:
    if decision == "report":
        return rng.sample(wrong, min(n, len(wrong)))
    by = {k: [w for w in wrong if w[2] == k] for k in HEADING_KINDS}
    share = n // len(HEADING_KINDS)
    chosen = []
    for k in HEADING_KINDS:
        chosen += rng.sample(by[k], min(share, len(by[k])))
    rest = [w for w in wrong if w not in chosen]
    chosen += rng.sample(rest, min(n - len(chosen), len(rest)))
    return chosen


def build(stream: int, out: Path, rewrites: int = 20, per_phase: int = 10, log=print) -> dict[str, Any]:
    """Build (or complete) the development store for one development stream. Returns the counts admitted."""
    if stream not in H.DEVELOPMENT_STREAMS:
        raise ValueError(f"{stream} is not a development stream")
    sources = [s for s in H.DEVELOPMENT_STREAMS if s != stream]
    store = H.CaseStore(out, verifier=verify, verifier_name=VERIFIER_NAME, development=True)
    rng = random.Random(f"devcases|{stream}")
    counts: dict[str, int] = {}

    def admit(c: dict) -> None:
        store.admit(c)
        key = f"{c['split']}|{c['decision']}"
        counts[key] = counts.get(key, 0) + 1

    # Stored wrong and right cases from the first source stream.
    src = sources[0]
    fl = list(flights(src, per_phase=per_phase))
    log(f"source {src}: {len(fl)} flights, {sum(f.goal_met and f.settled for f in fl)} usable")
    for decision in H.DECISIONS:
        right, wrong = _pools(fl, decision)
        if len(right) < H.RIGHT_PER_DECISION or not wrong:
            raise RuntimeError(f"source {src} gives {len(right)} right and {len(wrong)} wrong {decision} candidates")
        for i, (m, disp, kind, o, ev) in enumerate(rng.sample(right, H.RIGHT_PER_DECISION)):
            admit(_case(stream, "stored_right", decision, None, m, o, disp, ev, i, src))
        for i, (m, disp, kind, o, ev) in enumerate(_pick_wrong(wrong, STORED_WRONG_PER_DECISION, rng, decision)):
            admit(_case(stream, "stored_wrong", decision, kind, m, o, disp, ev, i, src))
    # Fresh wrong cases, 100 per decision for each rewrite ordinal, from further source streams, never reused.
    pool: dict[str, list] = {d: [] for d in H.DECISIONS}
    next_source = 1
    for n in range(1, rewrites + 1):
        for decision in H.DECISIONS:
            while len(pool[decision]) < H.FRESH_PER_REWRITE:
                if next_source >= len(sources):
                    raise RuntimeError("development sources exhausted")
                s = sources[next_source]
                next_source += 1
                more = list(flights(s, per_phase=per_phase))
                log(f"source {s}: {len(more)} flights for fresh cases")
                for d in H.DECISIONS:
                    _, w = _pools(more, d)
                    rng.shuffle(w)
                    pool[d] += w
            take, pool[decision] = pool[decision][:H.FRESH_PER_REWRITE], pool[decision][H.FRESH_PER_REWRITE:]
            for i, (m, disp, kind, o, ev) in enumerate(take):
                admit(_case(stream, "fresh_wrong", decision, kind, m, o, disp, ev, i, m.source, rewrite=n))
    summary = {"stream": stream, "rule": RULE, "rewrites": rewrites, "admitted": counts,
               "coverage": store.coverage(stream)}
    (Path(out) / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stream", type=int, required=True, help="the development stream the cases serve (4400 to 4499)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rewrites", type=int, default=20)
    p.add_argument("--per-phase", type=int, default=10)
    a = p.parse_args(argv)
    s = build(a.stream, a.out, a.rewrites, a.per_phase)
    print(json.dumps(s["admitted"], indent=2))


if __name__ == "__main__":
    main()
