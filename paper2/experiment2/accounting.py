"""Combined method-seconds and costs across conditions (decision D24; paper2/notes/COST-ATTRIBUTION-DESIGN.md with
Codex's TURN 131 corrections C131.D1, CUT, OVERLAP and DENOM). It reads records; it changes none.

Method-seconds. Every condition's seconds go into one set of categories. A category the condition cannot have is
null, never zero. Applicable counts sum to K, the action seconds of the ending; fractions are counts over K when K is
above zero and null otherwise. Pooling adds counts and K.

Costs. Each measured thing is one row: one identity, one purpose tag (running, build, trial, replay, planting,
setup), one process (policy, world, a model role, tool, procedure, check, allocation), its action step (None for
work done before the first action, which every ending includes), and its processor seconds, wall seconds, tokens or
GPU hours, each null when it was not measured. A purpose total is a view over these rows, never a charge added
beside them. A row whose time is already inside an inclusive parent names the resources the parent covers, and
those are not counted twice. A recovery row carries an earlier call's charge and is never charged again.

At an ending with K action seconds, a row issued before step K counts in full even if its answer came later; such
a model call is also listed as straddling, and one with an unknown answer step as of unknown placement. Rows issued
at or after K are excluded.

The headline is a vector, not a single number: processor seconds of local work; input and output tokens and call
counts by model role, with failed and unknown counts; allocated GPU hours. Wall time is a diagnostic. A total with
any unknown part is null, with the known subtotal and the unknown count beside it.

Denominators come from the fixed schedule: scheduled, attempted, naturally complete (declined episodes included),
policy-failed, censored and not attempted are counted apart. Episodes after a policy failure are not attempted, not
failures. Settled, nominal and unsettled goal results are kept apart.

Learning costs follow decision D1: a build at schedule position k adds C/(N - k) to positions k to N - 1 and nothing
earlier (ledger.amortised); the actual and cumulative figures are given beside it. A matched subset takes its
positions' existing shares and redistributes nothing. Development code; nothing here is a result.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from telohelm.runtime.ledger import TOOL_KINDS

from telohelm.runtime.ledger import amortised

CATEGORIES = ("procedure", "small_model", "reasoning_model", "calculator", "adaptive_controller",
              "holding_pending", "holding_none")
APPLICABLE = {
    "telohelm": {"procedure", "small_model", "reasoning_model", "holding_pending", "holding_none"},
    "telohelm_no_replay": {"procedure", "small_model", "reasoning_model", "holding_pending", "holding_none"},
    "reasoning_alone": {"reasoning_model", "holding_pending", "holding_none"},
    "calculator": {"calculator", "holding_none"},
    "adaptive_controller": {"adaptive_controller", "holding_none"},
}
PURPOSES = ("running", "build", "trial", "replay", "planting", "setup")
FLIGHT = ("running", "replay")             # the purposes that belong to a flight's ending; a replay re-flies one
TIME = ("cpu_s", "wall_s")


# -- method-seconds -----------------------------------------------------------------------------------------------

def method_seconds(condition: str, counts: dict[str, int]) -> dict[str, Any]:
    """One episode's counts in the combined categories. A count in a category the condition cannot have is refused,
    never guessed into another."""
    applicable = APPLICABLE[condition]
    bad = set(counts) - applicable
    if bad:
        raise ValueError(f"{condition} cannot have {sorted(bad)}")
    k = sum(counts.values())
    out = {c: (int(counts.get(c, 0)) if c in applicable else None) for c in CATEGORIES}
    return {"condition": condition, "K": k, "counts": out,
            "fractions": {c: (None if n is None or k == 0 else n / k) for c, n in out.items()}}


def from_fixed(condition: str, method: dict[str, Any]) -> dict[str, Any]:
    """The comparator's method-seconds: issued seconds under the condition's own source, held seconds as
    holding_none (a fixed policy has nothing pending)."""
    shares = method["shares"]
    return method_seconds(condition, {condition: shares.get(condition, 0), "holding_none": shares.get("held", 0)})


def from_telohelm(condition: str, attribution: dict[str, Any]) -> dict[str, Any]:
    """attribution.derive's counts, whose categories are already the combined ones."""
    return method_seconds(condition, {c: n for c, n in attribution["counts"].items() if n})


def pool_method_seconds(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pooled counts over pooled K, for one condition."""
    conditions = {r["condition"] for r in results}
    if len(conditions) != 1:
        raise ValueError("pool one condition at a time")
    k = sum(r["K"] for r in results)
    counts = {c: (None if results and results[0]["counts"][c] is None else sum(r["counts"][c] for r in results))
              for c in CATEGORIES}
    return {"condition": conditions.pop() if conditions else None, "K": k, "episodes": len(results),
            "counts": counts, "fractions": {c: (None if n is None or k == 0 else n / k) for c, n in counts.items()}}


# -- cost rows --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Row:
    id: str
    condition: str
    episode: str | None
    purpose: str
    process: str
    step: int | None                       # action step issued at; None = before the first action
    cpu_s: float | None = None
    wall_s: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    gpu_h: float | None = None
    status: str = "ok"
    answered_step: int | None = None       # model calls: the step the answer arrived; None = unknown placement
    covered: tuple[str, ...] = ()          # resources already counted inside an inclusive parent
    recovers: str | None = None            # a recovery carrier for an earlier row; never charged again
    parent: str | None = None
    measured: tuple[str, ...] = field(default=())    # resources this row was meant to measure (null = unknown)

    def __post_init__(self):
        if self.purpose not in PURPOSES:
            raise ValueError(f"unknown purpose {self.purpose!r}")


def cut(rows: Iterable[Row], k: int) -> dict[str, list[Row]]:
    """C131.CUT: what a flight ending with K action seconds includes. Only running and replay rows belong to a
    flight; build, trial, planting and setup rows go to 'other_purpose', so a learning call at its attempt's step 0 is
    never part of an earlier flight's ending (ACC133.BUILD). A recovery carrier is kept only if the original call it carries
    is counted (ACC133.RECOVERY)."""
    rows = list(rows)
    out = {"counted": [], "excluded": [], "straddling": [], "unknown_placement": [], "carriers": [],
           "other_purpose": []}
    for r in rows:
        if r.recovers is not None:
            continue
        if r.purpose not in FLIGHT:
            out["other_purpose"].append(r)
        elif r.step is None or r.step < k:
            out["counted"].append(r)
            if r.process.startswith("model:"):
                if r.answered_step is None:
                    out["unknown_placement"].append(r)
                elif r.answered_step >= k:
                    out["straddling"].append(r)
        else:
            out["excluded"].append(r)
    counted = {r.id for r in out["counted"]}
    roots = _roots(rows)
    out["carriers"] = [r for r in rows if r.recovers is not None and roots.get(r.id) in counted]
    return out


def _roots(rows: list[Row]) -> dict[str, str]:
    """For each recovery carrier, the original call it carries: a later recovery may link to an earlier one, so the
    links are followed back to a row that is not itself a carrier."""
    link = {r.id: r.recovers for r in rows if r.recovers is not None}
    roots = {}
    for rid in link:
        seen, cur = set(), rid
        while cur in link and cur not in seen:
            seen.add(cur)
            cur = link[cur]
        roots[rid] = cur
    return roots


CARRIED = ("tokens_in", "tokens_out")      # a recovery carrier may carry these for its original; never time


def total(rows: Iterable[Row], resource: str) -> dict[str, Any]:
    """A total that is null if any part is unknown, with the known subtotal and the unknown count. Recovery carriers
    are not added; resources covered by a parent are left out. If an original's tokens are unknown, the first carrier
    linked to it among these rows supplies them, once, as the gateway charges them (ACC133.RECOVERY). An original's
    unknown wall or processor time stays unknown: a carrier's time is a cache read, not the call's."""
    rows = list(rows)
    roots = _roots(rows)
    first_carrier: dict[str, Row] = {}
    for r in rows:
        if r.recovers is not None and roots[r.id] not in first_carrier:
            first_carrier[roots[r.id]] = r
    known, unknown = 0.0, 0
    for r in rows:
        if r.recovers is not None or resource in r.covered or resource not in r.measured:
            continue
        v = getattr(r, resource)
        if v is None and resource in CARRIED and r.id in first_carrier:
            v = getattr(first_carrier[r.id], resource)
        if v is None:
            unknown += 1
        else:
            known += v
    return {"total": None if unknown else known, "known_subtotal": known, "unknown": unknown}


def resource_vector(rows: list[Row]) -> dict[str, Any]:
    """The headline: local processor seconds; tokens and calls by model role; allocated GPU hours. Wall time is a
    diagnostic only."""
    local = [r for r in rows if not r.process.startswith("model:") and r.process != "allocation"]
    roles = sorted({r.process.split(":", 1)[1] for r in rows if r.process.startswith("model:")})
    by_role = {}
    for role in roles:
        rr = [r for r in rows if r.process == f"model:{role}"]
        originals = [r for r in rr if r.recovers is None]
        by_role[role] = {"calls": len(originals), "not_answered": sum(r.status != "answered" for r in originals),
                         "tokens_in": total(rr, "tokens_in"), "tokens_out": total(rr, "tokens_out")}
    return {"cpu_s": total(local, "cpu_s"), "models": by_role,
            "gpu_h": total([r for r in rows if r.process == "allocation"], "gpu_h"),
            "wall_s_diagnostic": total(local, "wall_s")}


def by_purpose(rows: list[Row]) -> dict[str, dict[str, Any]]:
    """Views over the same rows, one per purpose tag; their parts are never added beside the rows."""
    return {p: resource_vector([r for r in rows if r.purpose == p]) for p in PURPOSES
            if any(r.purpose == p for r in rows)}


# -- learning allocation (D1) --------------------------------------------------------------------------------------

def allocate_learning(running: list[float], builds: dict[int, float], attempted: int | None = None) -> dict[str, Any]:
    """C131.D1 over the fixed schedule of len(running) positions. 'amortised' is D1's view; 'actual' charges each
    build where it happened; 'cumulative' is the build cost incurred by each position; the raw total is kept. When
    the stream stopped after `attempted` positions, the later positions' shares are marked as not observed."""
    n = len(running)
    view = amortised(builds, running)
    actual = [running[i] + builds.get(i, 0.0) for i in range(n)]
    cumulative, so_far = [], 0.0
    for i in range(n):
        so_far += builds.get(i, 0.0)
        cumulative.append(so_far)
    attempted = n if attempted is None else attempted
    return {"amortised": view, "actual": actual, "cumulative_build": cumulative,
            "raw_total": sum(running) + sum(builds.values()),
            "observed": [i < attempted for i in range(n)]}


def matched_subset(per_position: list[float], positions: list[int]) -> dict[str, Any]:
    """A matched subset selects existing shares; it redistributes nothing onto the subset."""
    return {"positions": list(positions), "values": [per_position[i] for i in positions],
            "total": sum(per_position[i] for i in positions)}


# -- denominators ------------------------------------------------------------------------------------------------

def denominators(scheduled: list[str], records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """C131.DENOM over the fixed schedule. A record has status 'ok' or 'policy_failed', complete, censored and its
    outcome. A scheduled episode with no record was not attempted."""
    counts = {"scheduled": len(scheduled), "attempted": 0, "natural_complete": 0, "policy_failed": 0, "censored": 0,
              "not_attempted": 0, "declined_complete": 0}
    goals = {"settled_met": 0, "settled_not_met": 0, "unsettled": 0, "nominal_met": 0, "unavailable": 0}
    for ep in scheduled:
        r = records.get(ep)
        if r is None:
            counts["not_attempted"] += 1
            continue
        counts["attempted"] += 1
        if r.get("status") == "policy_failed":
            counts["policy_failed"] += 1
            continue
        if not r.get("complete"):
            counts["censored"] += 1
            continue
        counts["natural_complete"] += 1
        if (r.get("condition_outcome") or {}).get("declined"):
            counts["declined_complete"] += 1
        o = r.get("outcome")
        if o is None:
            goals["unavailable"] += 1          # a record from before the whole outcome was kept (ACC133.OUTCOME)
            continue
        options = o.get("goal_met_options") or [bool(o.get("goal_met"))]
        goals["nominal_met"] += bool(o.get("goal_met"))
        if o.get("settled") is False or len(set(options)) > 1:
            goals["unsettled"] += 1
        elif options[0]:
            goals["settled_met"] += 1
        else:
            goals["settled_not_met"] += 1
    views = {"settled_met_over_scheduled": (goals["settled_met"] / counts["scheduled"]
                                            if counts["scheduled"] else None),
             "settled_met_over_complete": (goals["settled_met"] / counts["natural_complete"]
                                           if counts["natural_complete"] else None)}
    return {"counts": counts, "goals": goals, "views": views,
            "note": "settled_met_over_scheduled is a lower bound on completion, not a claim that the rest failed"}


def cost_per_success(cost: float | None, successes: int | None) -> float | None:
    """All incurred cost over justified successes; unavailable (None) when either is unknown or there are none."""
    return None if cost is None or not successes else cost / successes


# -- readers -----------------------------------------------------------------------------------------------------

def read_fixed(ep_dir: Path) -> dict[str, Any]:
    """Rows and method-seconds from a comparator episode folder (comparators.run_pilot_episode)."""
    ep_dir = Path(ep_dir)
    summary = json.loads((ep_dir / "summary.json").read_text())
    condition, episode = summary["condition"], summary["episode"]
    rows = []
    for i, w in enumerate(json.loads(l) for l in (ep_dir / "work.jsonl").read_text().splitlines()):
        process = "world" if w["scope"] == "world_step" else "policy"
        rows.append(Row(f"{condition}/{episode}/w{i}", condition, episode, "running", process,
                        None if w["scope"] == "policy_reset" else w["t"],
                        cpu_s=w["cpu_ns"] / 1e9, wall_s=w["wall_s"], measured=TIME))
    return {"summary": summary, "rows": rows, "method_seconds": from_fixed(condition, summary["method_seconds"]),
            "unmeasured": []}


def read_telohelm(ep_dir: Path, condition: str = "telohelm") -> dict[str, Any]:
    """Rows from a Telohelm episode folder (run_stream): model and prediction rows from ledger.jsonl, procedure and
    check runs from the episode log's decisions. Folders written before ledger.jsonl existed give model rows from the
    gateway log with processor time unknown, and predictions are listed as unmeasured. The world's own stepping is
    not timed on this path and is listed as unmeasured."""
    ep_dir = Path(ep_dir)
    summary = json.loads((ep_dir / "summary.json").read_text())
    episode = summary["episode"]
    rows: list[Row] = []
    unmeasured = ["world stepping"]
    ledger = ep_dir / "ledger.jsonl"
    if ledger.exists():
        for c in (json.loads(l) for l in ledger.read_text().splitlines()):
            tool = c["kind"] in TOOL_KINDS
            rows.append(Row(c["request_id"], condition, c["episode"], c["scope"],
                            "tool" if tool else f"model:{c['kind']}", c["action_step"],
                            cpu_s=c.get("cpu_s"), wall_s=c.get("wall_s"), tokens_in=c.get("tokens_in"),
                            tokens_out=c.get("tokens_out"), status=c["status"], answered_step=c.get("answered_step"),
                            recovers=c.get("recovers"), covered=("wall_s",) if tool else (),
                            measured=("cpu_s", "wall_s") if tool else ("wall_s", "tokens_in", "tokens_out")))
    else:
        unmeasured.append("prediction calls (no ledger.jsonl)")
        events = [json.loads(line)["payload"] for line in (ep_dir / "gateway.log").read_text().splitlines()]
        starts = {(e.get("request") or {}).get("request_id"): e.get("request") or {}
                  for e in events if e.get("kind") == "model_start"}
        for p in events:
            if p.get("kind") == "model_end":
                resp = p.get("response") or {}
                role = json.loads(p["request_id"])[3]
                step = (starts.get(p["request_id"]) or {}).get("action_step")
                rows.append(Row(p["request_id"], condition, episode, "running", f"model:{role}", step,
                                wall_s=p.get("wall_s"), tokens_in=resp.get("tokens_in"),
                                tokens_out=resp.get("tokens_out"), status=p["status"],
                                measured=("wall_s", "tokens_in", "tokens_out")))
    # Procedure and check timings are in the episode log; the broker's log leaves timings out of its logical record
    # (RP61.OUTCOME). On this path there is one action a second from 0, so a decision's step is its second.
    for line in (ep_dir / "episode.log").read_text().splitlines():
        p = json.loads(line)["payload"]
        if p.get("kind") != "decision":
            continue
        step = p.get("action_step", p["t"])
        for j, t in enumerate(p.get("tried") or []):
            base = f"{condition}/{episode}/{step}/{p['decision']}/{j}"
            rows.append(Row(base, condition, episode, "running", "procedure", step,
                            cpu_s=t.get("cpu_seconds"), wall_s=t.get("wall_seconds"), status=t.get("status", "ok"),
                            measured=TIME))
            for m, ch in enumerate(t.get("checks") or []):
                rows.append(Row(f"{base}/check{m}", condition, episode, "running", "check", step,
                                cpu_s=ch.get("cpu_seconds"), wall_s=ch.get("wall_seconds"),
                                status=ch.get("status", "ok"), measured=TIME))
    if "outcome" not in summary:
        unmeasured.append("whole outcome (the summary predates it; settled status unavailable)")
    import telohelm.runtime.attribution as A
    derived = A.derive(ep_dir / "broker.log", ep_dir / "broker.heads", episode)
    return {"summary": summary, "rows": rows, "unmeasured": unmeasured,
            "method_seconds": from_telohelm(condition, derived)}


def read_learning(stream_dir: Path, condition: str = "telohelm") -> dict[str, Any]:
    """The learning ledger of a Telohelm stream (run_stream's learning-ledger.jsonl): proposer calls as build rows,
    each anchored to the completed episode that prompted its attempt. Learning work that is not yet timed is listed
    as unmeasured, never as zero."""
    path = Path(stream_dir) / "learning-ledger.jsonl"
    rows = []
    if path.exists():
        for c in (json.loads(l) for l in path.read_text().splitlines()):
            rows.append(Row(c["request_id"], condition, c["episode"], c["scope"], f"model:{c['kind']}",
                            c["action_step"], wall_s=c.get("wall_s"), tokens_in=c.get("tokens_in"),
                            tokens_out=c.get("tokens_out"), status=c["status"], recovers=c.get("recovers"),
                            measured=("wall_s", "tokens_in", "tokens_out")))
    unmeasured = ["shadow trial flights (their logs are not kept)", "replay tests", "attempt bookkeeping"]
    if not path.exists():
        unmeasured.insert(0, "proposer calls (no learning-ledger.jsonl)")
    return {"rows": rows, "unmeasured": unmeasured}


def stream_allocation(schedule: list[str], running: dict[str, float | None], build_rows: list[Row],
                      resource: str, attempted: int | None = None) -> dict[str, Any]:
    """D1 over a stream's fixed schedule for one resource: running cost per scheduled episode, and each build row's
    cost at the schedule position of the completed episode it is anchored to. Unavailable, with the known parts, if
    any part is unknown."""
    by_position: dict[int, list[Row]] = {}
    for r in build_rows:
        if r.episode not in schedule:
            raise ValueError(f"build row {r.id} is anchored to {r.episode!r}, which is not in the schedule")
        by_position.setdefault(schedule.index(r.episode), []).append(r)
    builds = {i: total(rr, resource) for i, rr in by_position.items()}
    run = [running.get(ep) for ep in schedule]
    if any(v is None for v in run) or any(b["total"] is None for b in builds.values()):
        return {"available": False, "running": run,
                "builds": {i: b for i, b in builds.items()},
                "reason": "a running or build cost is unknown; no allocation is made"}
    return {"available": True, **allocate_learning(run, {i: b["total"] for i, b in builds.items()}, attempted)}
