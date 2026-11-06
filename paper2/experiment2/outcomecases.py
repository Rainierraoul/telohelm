"""Development cases labelled by outcome (option C, paper2/notes/OPTION-C-PLAN.md; Simon's D41 development rules).

    study2/.venv/bin/python -m study2.outcomecases --stream 4401 --out <dir> [--rewrites 10] [--workers 12]

Why: the reference-relative labels of devcases.py (dev.reference.v1) called a heading wrong when it disagreed with the
reference pilot, so no check Telohelm wrote could pass the replay test without guessing the pilot's method. Here a
candidate action is labelled by what it does to the goal, settled by the world's own simulator from the true state.

The label contract, rule dev.outcome.v3 (21 October 2026; paper2/notes/OUTCOME-RULE-V3.md, after Codex's TURN207
review; v1 and v2 are kept in the history and their stores are refused by runs that learn checks):
- Every label is settled in the quiet continuation of its second: the episode as it stands, with nothing still to
  come. Later goal changes and dropouts are taken out, a dropout under way ends at once, and an orbit change the
  context does not yet show is taken out. The episode's own future is never used, so the same context and output
  get the same label whichever episode they come from. Where the context cannot show the radar delay, the quiet
  continuation is flown under every delay the generator can draw that gives exactly the same context, and the case
  is kept only if all of them give the same label; otherwise it is unknown. Only the radar delay and a hidden orbit
  change are varied: this is a bounded family of continuations, not a proof that nothing hidden matters.
- Heading. The candidate heading is commanded for HOLD_S seconds, after which the reference pilot carries on. The
  baseline is the same with the reference pilot's own heading of that second held for the same HOLD_S seconds. The
  candidate is right if the aircraft joins at most RIGHT_DELAY_S seconds after the baseline does, and wrong if it
  misses the join or joins more than WRONG_DELAY_S seconds after it. A delay between the two is marginal: the case
  is left out and counted. If the baseline does not join in time, every heading at that second is unknown. The hold
  stands for a procedure that keeps producing that heading (one second of a turned heading changes almost nothing,
  paper2/development/option-c/).
- Report. At a second before the goal's report is made, the candidate report action (a report now, a trigger setting,
  or nothing) is applied once. The reference pilot then flies the headings and makes its own report only while no
  report has been made under the goal and no trigger is armed; when its own report is taken away it may report on a
  later second. The candidate is right if exactly one report is made under the goal and it is correct, which in the
  quiet continuation is the goal in effect at that second. Besides the geometry candidates, the second at which the
  reference reports also carries reports whose content is wrong: another controller, the other tanker, or another
  distance.
- An outcome the world cannot settle is unknown, and the case is left out and counted. Two candidates with the same
  public context and output but opposite labels are both left out and counted.
Nothing here reaches the learner: it sees only the replay test's two facts (protocol section 5).

Sources are fixed before construction by partition(): development sources 4400 to 4404 were used to design this
rule and the reference checks and are never a source again, and each run stream has its own disjoint sources for
its replay store (stored cases from the first, fresh wrong cases for each rewrite from the rest), its eligibility
pool and its final held-back cases. The verifier recomputes every label from its evidence.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from . import devcases as DC
import telohelm.qualification.store as H
from telohelm.runtime.context import ContextBuilder
import telohelm.taskpacks.tanker.world.episode as E
from . import streams as streams
from telohelm.taskpacks.tanker.world.pilot import ReferencePilot

RULE = "dev.outcome.v3"
VERIFIER_NAME = "development outcome replay of dev.outcome.v3"
HOLD_S = 120                         # development values (paper2/notes/OPTION-C-PLAN.md); the protocol fixes its own
RIGHT_DELAY_S, WRONG_DELAY_S = 15, 60    # class D, declared on development data (paper2/notes/OUTCOME-RULE-V2.md)
TURNS_DEG = (-90.0, -45.0, -20.0, 20.0, 45.0, 90.0, 180.0)
TRIGGER_OFFSETS_NM = (-4.0, 4.0)
WRONG_CONTROLLER, WRONG_DISTANCE_NM = "Brisbane", 5.0    # content-negative reports (Codex's TURN207 finding 3)
STRIDE_S = 30                        # one moment in every 30 seconds of a reference flight
STORED_WRONG = 40


def _goal(ep: E.Episode) -> dict:
    g = ep.goal
    return {"version": g.version, "tanker": g.tanker, "report_at_nm": g.report_at_nm, "deadline": g.deadline}


def _reported(ep: E.Episode) -> bool:
    return any(r.get("goal_version") == ep.goal.version for r in ep.record.reports)


def _scenario(source: int, phase: int, index: int):
    return streams.solvable_episode(source, phase, index).scenario


def _prefix(source: int, phase: int, index: int, t: int):
    """The episode, pilot, context builder and next bundle at second t of the reference flight, or None if it ended."""
    return _prefix_of(_scenario(source, phase, index), t)


def _prefix_of(scenario, t: int):
    ep = E.Episode(scenario)
    start = ep.reset()
    pilot = ReferencePilot(start["briefing"], ask_help=False)
    builder = ContextBuilder(start["briefing"])
    bundle = start["bundle"]
    for _ in range(t):
        builder.update(bundle)
        bundle = ep.step(pilot.act(bundle))
        if bundle is None:
            return None
    return ep, pilot, builder, bundle


def _continue(ep, pilot, bundle, heading=None, hold=0, report_action=None) -> dict:
    """Fly on from a prefix. A heading override is held for `hold` seconds; a report action is applied once, first.
    The reference reports only while no report is made under the goal and no trigger is armed. When its own report
    is taken away, it is told it has not reported, so it tries again on a later second (the pilot marks a goal as
    reported as soon as it decides to report)."""
    k = 0
    while bundle is not None:
        reported_for = pilot.reported_for
        a = dict(pilot.act(bundle))
        if heading is not None and k < hold:
            a["heading_deg"] = heading
        if report_action is not None:
            own = "report" in a
            if k == 0:                       # the candidate replaces the reference's own report decision once
                a.pop("report", None)
                a.update(report_action)
            elif "report" in a and (_reported(ep) or ep.trigger is not None):
                a.pop("report")
            if own and (k == 0 or "report" not in a):     # its own report was taken away
                pilot.reported_for = reported_for
        bundle = ep.step(a)
        k += 1
    return ep.outcome()


def _joined_at(o: dict):
    return o.get("ended_at") if o.get("joined") and o.get("in_time") else None


def _disposition(base: int, join: int | None) -> str:
    """A heading's label from the baseline's join second and its own (None if it missed the join)."""
    if join is None or join - base > WRONG_DELAY_S:
        return "wrong"
    return "right" if join - base <= RIGHT_DELAY_S else "marginal"


def _heading_from(prefix, base, heading: float):
    ep, pilot, _, bundle = (copy.deepcopy(x) for x in prefix)
    o = _continue(ep, pilot, bundle, heading=heading, hold=HOLD_S)
    if o.get("settled") is not True:
        return "unknown", {"reason": "outcome not settled"}
    join = _joined_at(o)
    ev = {"base_join_s": base, "baseline": "reference heading held", "join_s": join, "hold_s": HOLD_S,
          "right_delay_s": RIGHT_DELAY_S, "wrong_delay_s": WRONG_DELAY_S}
    return _disposition(base, join), ev


def _report_from(prefix, action: dict):
    ep, pilot, _, bundle = (copy.deepcopy(x) for x in prefix)
    if _reported(ep):
        return "unknown", {"reason": "the goal's report was already made"}
    o = _continue(ep, pilot, bundle, report_action=action)
    if o.get("settled") is not True or o.get("report_correct_options") not in ([True], [False]):
        return "unknown", {"reason": "outcome not settled"}
    ok = o.get("reports_under_goal") == 1 and o.get("report_correct") is True
    return ("right" if ok else "wrong"), {"reports_under_goal": o.get("reports_under_goal"),
                                          "report_correct": o.get("report_correct")}


def _reference_heading(prefix) -> float:
    """The reference pilot's heading at the prefix's second, from its own copy (act() changes the pilot's state)."""
    _, pilot, _, bundle = (copy.deepcopy(x) for x in prefix)
    return pilot.act(bundle)["heading_deg"]


def _base_from(prefix):
    """The baseline's join second: the reference pilot's own heading held for HOLD_S seconds, then the reference
    carrying on. None if that does not settle or does not join in time."""
    heading = _reference_heading(prefix)
    ep, pilot, _, bundle = (copy.deepcopy(x) for x in prefix)
    o = _continue(ep, pilot, bundle, heading=heading, hold=HOLD_S)
    return _joined_at(o) if o.get("settled") is True else None


def _context(prefix) -> dict:
    _, _, builder, bundle = (copy.deepcopy(x) for x in prefix)
    return builder.update(bundle)


def _quiet(scenario, t: int):
    """The scenario with nothing still to come after second t: later goal changes and dropouts taken out, and a
    dropout under way ended at once (its last lost measurement is at t)."""
    drops = tuple((c, start, end if end is not None and end <= t + 1 else t + 1)
                  for c, start, end in scenario.dropouts if start <= t)
    return replace(scenario, goals=tuple(g for g in scenario.goals if g[0] <= t), dropouts=drops)


def _family(scenario, t: int, ctx: dict) -> list:
    """Prefixes at second t of the quiet continuations the context cannot tell apart, the scenario's own radar delay
    first. An orbit change is taken out where taking it out leaves the context exactly the same (it has not shown
    yet); then every radar delay the generator can draw (world/streams.py) that leaves the context exactly the same
    is a member."""
    quiet = _quiet(scenario, t)
    for i, k in enumerate(quiet.tankers):
        if k.orbit.delta:
            level = replace(quiet, tankers=tuple(replace(x, orbit=replace(x.orbit, delta=0.0)) if j == i else x
                                                 for j, x in enumerate(quiet.tankers)))
            p = _prefix_of(level, t)
            if p is not None and _context(p) == ctx:
                quiet = level
    out = []
    for d in [scenario.delay_s] + [d for d in (0,) + tuple(streams.DELAYS) if d != scenario.delay_s]:
        p = _prefix_of(replace(quiet, delay_s=d), t)
        if p is not None and _context(p) == ctx:
            out.append(p)
    if not out or out[0][0].s.delay_s != scenario.delay_s:
        raise RuntimeError(f"the quiet continuation of {scenario.episode_id} at {t} s does not give the same context")
    return out


def _label_in(prefix, base, decision: str, output: dict):
    if decision == "heading":
        if base is None:
            return "unknown", {"reason": "no settled baseline join"}
        return _heading_from(prefix, base, output["heading_deg"])
    return _report_from(prefix, output)


def _labels(scenario, t: int, candidates: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """(disposition, evidence) for each (decision, output) at second t, settled in the quiet continuations the context
    cannot tell apart; kept only if they all agree, otherwise unknown. Evidence is from the scenario's own delay."""
    p = _prefix_of(scenario, t)
    if p is None:
        return [("unknown", {"reason": "flight ended before this second"}) for _ in candidates]
    worlds = _family(scenario, t, _context(p))
    bases = [_base_from(w) for w in worlds]
    delays = [w[0].s.delay_s for w in worlds]
    out = []
    for decision, output in candidates:
        results = [_label_in(w, b, decision, output) for w, b in zip(worlds, bases)]
        disp, ev = results[0]
        if len({r[0] for r in results}) > 1:
            disp, ev = "unknown", {**ev, "reason": "the continuations the context cannot tell apart disagree",
                                   "labels_by_delay": {str(d): r[0] for d, r in zip(delays, results)}}
        out.append((disp, {**ev, "delays_compared": delays}))
    return out


def heading_label(source: int, phase: int, index: int, t: int, heading: float):
    """('right' | 'wrong' | 'marginal' | 'unknown', evidence) for a heading candidate under the rule, from scratch."""
    return _labels(_scenario(source, phase, index), t, [("heading", {"heading_deg": heading})])[0]


def report_label(source: int, phase: int, index: int, t: int, action_json: str):
    return _labels(_scenario(source, phase, index), t, [("report", json.loads(action_json))])[0]


def moments(source: int, per_phase: int, phases=(1, 2)) -> list[dict]:
    """Public contexts at every STRIDE_S-th second of settled, goal-meeting reference flights of one source stream,
    and at the second the reference pilot reports, which is where a report's content can be tested on its own."""
    out = []
    for phase in phases:
        # Phases 3 and 4 hold four blocks of 5 (orbit change, dropout, late reports, all three; world/streams.py), so
        # their indices are taken one from each block in turn, so that every block is covered.
        order = [b * 5 + k for k in range(5) for b in range(4)] if phase in (3, 4) else list(range(20))
        for index in order[:per_phase]:
            try:
                g = streams.solvable_episode(source, phase, index)
            except streams.GenerationError:
                continue
            ep = E.Episode(g.scenario)
            start = ep.reset()
            pilot = ReferencePilot(start["briefing"], ask_help=False)
            builder = ContextBuilder(start["briefing"])
            bundle = start["bundle"]
            control = start["briefing"]["control_callsign"]
            tankers = [k["callsign"] for k in start["briefing"]["tankers"]]
            seen = []
            while bundle is not None:
                ctx = copy.deepcopy(builder.update(bundle))
                act = pilot.act(bundle)
                if ctx["t"] % STRIDE_S == 0 or "report" in act:
                    seen.append({"source": source, "phase": phase, "index": index, "episode": g.scenario.episode_id,
                                 "t": ctx["t"], "ctx": ctx, "goal": _goal(ep), "reference_heading": act["heading_deg"],
                                 "reported": _reported(ep), "control": control, "tankers": tankers,
                                 "reporting": "report" in act})
                bundle = ep.step(act)
            o = ep.outcome()
            if o.get("goal_met") is True and o.get("settled") is True:
                out += seen
    return out


def _candidates(m: dict) -> list[tuple[str, dict, dict]]:
    """(decision, output, label key) for one moment."""
    ref = m["reference_heading"]
    out = [("heading", {"heading_deg": ref}, {"construction": "reference"})]
    for turn in TURNS_DEG:
        out.append(("heading", {"heading_deg": (ref + turn) % 360.0}, {"construction": "turned", "turn_deg": turn}))
    if not m["reported"] and m["ctx"].get("observed_range_nm") is not None:
        d = m["goal"]["report_at_nm"]
        out.append(("report", {}, {"construction": "nothing"}))
        out.append(("report", {"report_trigger_nm": d}, {"construction": "trigger_at_order"}))
        for off in TRIGGER_OFFSETS_NM:
            if 0 < d + off <= 1000:
                out.append(("report", {"report_trigger_nm": d + off}, {"construction": "trigger_off", "offset_nm": off}))
        tanker = m["goal"]["tanker"]
        out.append(("report", {"report": DC.report_text(m["control"], d, tanker)}, {"construction": "report_now"}))
        if m.get("reporting"):          # the reference reports this second, so only the content can be wrong
            other = next((k for k in m.get("tankers", ()) if k != tanker), None)
            wrong = [("controller", DC.report_text(WRONG_CONTROLLER, d, tanker)),
                     ("distance", DC.report_text(m["control"], d + WRONG_DISTANCE_NM, tanker))]
            if other is not None:
                wrong.append(("tanker", DC.report_text(m["control"], d, other)))
            for field, text in wrong:
                out.append(("report", {"report": text}, {"construction": "wrong_content", "field": field}))
    return out


def _label_moment(m: dict) -> list[dict]:
    """Every candidate of one moment, through _labels, the function verify() uses from scratch."""
    sc = _scenario(m["source"], m["phase"], m["index"])
    p = _prefix_of(sc, m["t"])
    if p is not None and _reference_heading(p) != m["reference_heading"]:
        raise RuntimeError(f"the rebuilt prefix of {m['episode']} at {m['t']} s gives another reference heading")
    cands = _candidates(m)
    rows = []
    for (decision, output, key), (disp, ev) in zip(cands, _labels(sc, m["t"], [(d, o) for d, o, _ in cands])):
        rows.append({"moment": m, "decision": decision, "output": output, "disposition": disp,
                     "evidence": {**key, **ev, "source": m["source"], "phase": m["phase"], "index": m["index"],
                                  "second": m["t"]}})
    return rows


def label_all(ms: list[dict], workers: int) -> tuple[list[dict], dict]:
    with Pool(workers) as pool:
        rows = [r for rs in pool.map(_label_moment, ms) for r in rs]
    return _keep(rows)


def _keep(rows: list[dict]) -> tuple[list[dict], dict]:
    """The rows that become cases, and counts of those left out: unknown, marginal and conflicting."""
    counts = {"labelled": len(rows), "unknown": sum(r["disposition"] == "unknown" for r in rows),
              "marginal": sum(r["disposition"] == "marginal" for r in rows)}
    # Exclusions by phase, decision and disposition, so that what is left out can be reported (Codex's TURN207).
    by: dict[str, int] = {}
    for r in rows:
        key = f"p{r['moment']['phase']}|{r['decision']}|{r['disposition']}"
        by[key] = by.get(key, 0) + 1
    counts["by_phase"] = dict(sorted(by.items()))
    rows = [r for r in rows if r["disposition"] in ("right", "wrong")]
    # The same public context and output with opposite labels: both are left out (hidden futures differ).
    keyed: dict[str, set] = {}
    for r in rows:
        keyed.setdefault(json.dumps([r["moment"]["ctx"], r["output"]], sort_keys=True), set()).add(r["disposition"])
    conflict = {k for k, v in keyed.items() if len(v) > 1}
    counts["conflicting"] = sum(json.dumps([r["moment"]["ctx"], r["output"]], sort_keys=True) in conflict for r in rows)
    rows = [r for r in rows if json.dumps([r["moment"]["ctx"], r["output"]], sort_keys=True) not in conflict]
    return rows, counts


def verify(case: dict) -> bool:
    """Rebuild the moment from its evidence, require the stored context, goal and origin to be exactly that moment's
    (Codex's TURN 189 finding: a case whose context was changed must not pass), then recompute the label under
    the current rule and compare. A case labelled under another rule does not verify."""
    try:
        lab, ev = case["label"], case["label"]["evidence"]
        if lab["rule"] != RULE or lab["disposition"] not in ("right", "wrong"):
            return False
        p = _prefix(ev["source"], ev["phase"], ev["index"], ev["second"])
        if p is None:
            return False
        ep, _, builder, bundle = (copy.deepcopy(x) for x in p)
        ctx = builder.update(bundle)
        origin = case["origin"]
        if ctx != case["context"] or _goal(ep) != case["goal"] or ctx["t"] != ev["second"]:
            return False
        if origin.get("episode") != _scenario(ev["source"], ev["phase"], ev["index"]).episode_id or \
                origin.get("second") != ev["second"] or origin.get("seed") != ev["source"] or \
                origin.get("phase") != ev["phase"]:
            return False
        sc = _scenario(ev["source"], ev["phase"], ev["index"])
        disp = _labels(sc, ev["second"], [(case["decision"], case["output"])])[0][0]
        return disp == lab["disposition"]
    except (KeyError, TypeError, ValueError):
        return False


def store_rules(store: H.CaseStore) -> set:
    """Every label rule in a store, so a run that learns checks can refuse a store that is not purely outcome-labelled."""
    return {c["label"].get("rule") for c in store.cases()}


def _case(stream: int, split: str, r: dict, ordinal: int, rewrite: int | None = None) -> dict:
    m, ev = r["moment"], r["evidence"]
    origin = {"episode": m["episode"], "second": m["t"], "generator": f"{RULE}:{ev['construction']}",
              "seed": m["source"], "ordinal": ordinal, "attempt": 0, "phase": m["phase"]}
    if rewrite is not None:
        origin["rewrite"] = rewrite
    # The store takes B8's kinds; as in devcases.py, a wrong heading is geometry and a wrong report action identity.
    kind = None if r["disposition"] == "right" else ("geometry" if r["decision"] == "heading" else "identity")
    return {"schema": H.SCHEMA, "stream": stream, "split": split, "decision": r["decision"], "kind": kind,
            "interface": "action", "goal": m["goal"], "context": m["ctx"], "output": r["output"], "origin": origin,
            "label": {"disposition": r["disposition"], "rule": RULE, "evidence": ev,
                      "reason": f"{r['disposition']} by outcome under {RULE} (hold {HOLD_S} s, right within "
                                f"{RIGHT_DELAY_S} s and wrong beyond {WRONG_DELAY_S} s of the held reference heading); "
                                "development only"}}


# Fixed before any v3 store is built (Codex's TURN207 finding 2). Sources 4400 to 4404 were used to design the rule
# and the reference checks C0, so they are never a source again. Each run stream, the rehearsal's then the scored
# ones, has its own block: ROLES gives each role's number of sources, in this order.
C0_DEVELOPMENT = tuple(range(4400, 4405))
RUN_STREAMS = (4401, 4402, 4403, 4404) + tuple(H.SCORED_STREAMS)
ROLES = (("replay", 5), ("eligibility", 1), ("final", 1))
# 26 October 2026 (Claude, TURN244, after Codex's TURN243 review): the M1 timing rehearsals and two development test
# files generated scenarios from 4441 to 4444, part of scored stream 4302's replay block. Those sources are retired,
# never reused, and replaced prospectively by unused development identities. The exposure is inventoried in
# paper2/reproduction/m1-reserved-exposure-claude-20261026.json.
EXPOSED_SOURCES = {4441: 4461, 4442: 4462, 4443: 4463, 4444: 4464}


def partition() -> dict[int, dict[str, tuple[int, ...]]]:
    """{run stream: {role: source streams}}: disjoint blocks from 4405 on, none of them a run stream, used in C0's
    development or retired after exposure (EXPOSED_SOURCES maps each retired source to its replacement)."""
    out, nxt = {}, C0_DEVELOPMENT[-1] + 1
    for stream in RUN_STREAMS:
        out[stream] = {}
        for role, n in ROLES:
            out[stream][role] = tuple(range(nxt, nxt + n))
            nxt += n
    out = {stream: {role: tuple(EXPOSED_SOURCES.get(x, x) for x in block) for role, block in roles.items()}
           for stream, roles in out.items()}
    used = [x for roles in out.values() for block in roles.values() for x in block]
    assert len(used) == len(set(used)) and set(used) <= set(H.DEVELOPMENT_STREAMS)
    assert not set(used) & (set(C0_DEVELOPMENT) | set(RUN_STREAMS) | set(EXPOSED_SOURCES))
    return out


def duplicates(*case_lists: list[dict]) -> list[str]:
    """(context, output) keys that occur in more than one of the given lists of cases, for checking that replay,
    eligibility and final cases share none."""
    seen: dict[str, int] = {}
    for i, cases in enumerate(case_lists):
        for c in cases:
            key = json.dumps([c["context"], c["output"]], sort_keys=True)
            if seen.setdefault(key, i) != i:
                seen[key] = -1
    return [k for k, v in seen.items() if v == -1]


def build(stream: int, out: Path, rewrites: int = 10, per_phase: int = 5, workers: int = 12, log=print,
          phases: tuple[int, ...] = (1, 2), development: bool = False) -> dict:
    """A run stream's replay store, from its replay sources in partition(): stored right and wrong cases from the
    first, fresh wrong cases for each rewrite from the rest. With development, a store from the development sources
    C0_DEVELOPMENT instead, keyed by the first of them, for working on the reference checks; it is never a run's."""
    if development:
        stream, sources = C0_DEVELOPMENT[0], list(C0_DEVELOPMENT)
    else:
        if stream not in H.DEVELOPMENT_STREAMS:
            raise ValueError(f"{stream} is not a development stream; a scored stream's store is the protocol's (B8)")
        if stream not in RUN_STREAMS:
            raise ValueError(f"{stream} has no sources in partition()")
        sources = list(partition()[stream]["replay"])
    checked: dict[str, bool] = {}

    def lookup(case: dict) -> bool:
        # verify() is run for every case before admission, in parallel; the store reads its result here.
        return checked.get(json.dumps(case, sort_keys=True)) is True

    store = H.CaseStore(out, verifier=lookup, verifier_name=VERIFIER_NAME, development=True)
    staged: list[dict] = []
    rng = random.Random(f"outcomecases|{RULE}|{stream}")
    counts: dict[str, Any] = {"admitted": {}, "labelling": []}

    def admit(c: dict) -> None:
        staged.append(c)

    def pools(source: int):
        rows, c = label_all(moments(source, per_phase, phases), workers)
        counts["labelling"].append({"source": source, **c})
        log(f"source {source}: {c}")
        by = {(d, disp): [r for r in rows if r["decision"] == d and r["disposition"] == disp]
              for d in H.DECISIONS for disp in ("right", "wrong")}
        return by

    first = pools(sources[0])
    for d in H.DECISIONS:
        right, wrong = first[(d, "right")], first[(d, "wrong")]
        if len(right) < H.RIGHT_PER_DECISION or len(wrong) < STORED_WRONG:
            raise RuntimeError(f"source {sources[0]} gives {len(right)} right and {len(wrong)} wrong {d} cases")
        for i, r in enumerate(rng.sample(right, H.RIGHT_PER_DECISION)):
            admit(_case(stream, "stored_right", r, i))
        for i, r in enumerate(rng.sample(wrong, STORED_WRONG)):
            admit(_case(stream, "stored_wrong", r, i))
    pool = {d: [] for d in H.DECISIONS}
    nxt = 1
    for n in range(1, rewrites + 1):
        for d in H.DECISIONS:
            while len(pool[d]) < H.FRESH_PER_REWRITE:
                if nxt >= len(sources):
                    raise RuntimeError("development sources exhausted")
                more = pools(sources[nxt])
                nxt += 1
                for dd in H.DECISIONS:
                    w = more[(dd, "wrong")]
                    rng.shuffle(w)
                    pool[dd] += w
            take, pool[d] = pool[d][:H.FRESH_PER_REWRITE], pool[d][H.FRESH_PER_REWRITE:]
            for i, r in enumerate(take):
                admit(_case(stream, "fresh_wrong", r, i, rewrite=n))
    with Pool(workers) as pool:
        results = pool.map(verify, staged, chunksize=8)
    for c, ok in zip(staged, results):
        checked[json.dumps(c, sort_keys=True)] = ok
    counts["verified"] = sum(results)
    counts["refused_by_verifier"] = len(results) - sum(results)
    for c, ok in zip(staged, results):
        if ok:
            store.admit(c)
            k = f"{c['split']}|{c['decision']}"
            counts["admitted"][k] = counts["admitted"].get(k, 0) + 1
    summary = {"stream": stream, "rule": RULE, "development_store": development, "hold_s": HOLD_S, "right_delay_s": RIGHT_DELAY_S,
               "wrong_delay_s": WRONG_DELAY_S, "rewrites": rewrites, "replay_sources": sources,
               "phases": list(phases),
               **counts, "coverage": store.coverage(stream)}
    (Path(out) / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stream", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rewrites", type=int, default=10)
    p.add_argument("--per-phase", type=int, default=5)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--phases", type=int, nargs="+", default=[1, 2], help="source phases (1 to 4 include disturbances)")
    p.add_argument("--development", action="store_true", help="a store from the development sources 4400 to 4404")
    a = p.parse_args(argv)
    s = build(a.stream, a.out, a.rewrites, a.per_phase, a.workers, phases=tuple(a.phases), development=a.development)
    print(json.dumps(s["admitted"]))


if __name__ == "__main__":
    main()
