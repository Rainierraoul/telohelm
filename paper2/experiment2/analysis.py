"""Experiment 2's analysis, written before any scored result (protocol amendment 7; sections 7 and 8).

It reads a flown study and writes the RQ1, RQ2 and help tables, per stream and across streams. Every position is
taken from its condition's state record when that record is final. A dispatcher stopped at the host's time limit can
leave its first, running state behind; its positions then come from the host's receipt, rebuilt from the records, and
without that receipt the condition is refused (Codex's TURN303 review). A summary is used only if it still has the
digest its record holds. L's learning attempts come from the durable learning journal; the final learning-events.json,
when present, must equal one whole session of it. Streams are the independent units: results are given per stream
and across streams as the values and their range, with no significance test and no interval (section 8).

- Episode outcome. Met alone (goal met, no help request), met with help, missed. A position that did not complete
  naturally (censored, paused, failed, interrupted, cut off or not attempted) and an unsettled outcome are counted
  apart and never as a miss or a goal. A position cut off or interrupted while flying made calls that are not all
  recorded, so any cost total it enters is unavailable (null), with the known subtotal kept.
- RQ1, on C's matched positions (five a phase in phases 1 to 4): L, C and C-full by outcome, by flight model calls
  and tokens, and by share of seconds per method. L's learning cost is added under decision D1: a build at schedule
  position k adds its cost divided by N - k to each of positions k to N - 1, with N = 100, forward only.
- RQ2, for each phase 3 block: L and F over the five positions after the block's first boundary, beside the learning
  attempts and adoptions at that boundary, and L's own five positions before against the five after. A stream with
  no attempt at the boundary is marked uninformative for recovery and still shown (amendment 7, item 7).
- Help, descriptively: requests by condition, phase and reason; phase 5 episodes are those labelled as needing help.

Only streams that ran are analysed: a declared stream with no condition state record was never started, and it is
listed with its disposition (the operator's launch-disposition.json in the stream folder, for example not run for
budget) and never counted as zero observations. Inside an analysed stream, unattempted positions stay in the
denominators. A token count that is unknown for any call makes that total and every D1 share it reaches unavailable
(null); the known subtotal and the number of calls of unknown usage are kept beside it (Codex's TURN295 review).

    python -m paper2.experiment2.analysis --study STUDY --out OUT [--receipts HOST-RECEIPTS.json ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ARMS = ("L", "C", "C-full", "F")
MATCHED = ("L", "C", "C-full")
N = 100                                   # L's scheduled positions in a scored stream (decision D1)
F_BLOCKS = ((3, range(0, 5), range(5, 10)), (3, range(10, 15), range(15, 20)))   # phase, before, after
METHODS = ("procedure", "small_model", "reasoning_model", "holding_pending", "holding_none")
COSTS = ("model_calls", "tokens_in", "tokens_out")
TOKENS = ("tokens_in", "tokens_out")
# Position statuses, as the runner's and the host's receipts write them.
NATURAL = ("complete", "censored", "boundary_open")      # the flight ended by itself; its summary gives the outcome
KNOWN_COST = NATURAL + ("policy_failed",)                 # a summary records the flight's whole cost
UNKNOWN_COST = ("interrupted", "cut_off", "failed", "damaged", "unknown", "unresolved")
NO_COST = ("not_attempted", "unattempted", "not_started", "paused")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _phase_index(episode: str) -> tuple[int, int]:
    """s4301-p3-e05 -> (3, 5)."""
    _, p, e = episode.split("-")
    return int(p[1:]), int(e[1:])


def outcome(summary: dict | None, status: str) -> str:
    """One position's outcome category. A position with no summary keeps its recorded status (not attempted,
    paused, failed); an incomplete one is censored, whatever its nominal result."""
    if summary is None:
        return status
    if summary.get("complete") is not True:
        return "censored"
    if summary.get("settled") is not True:
        return "unsettled"
    if summary.get("goal_met") is True:
        return "met_with_help" if summary.get("help_requests") else "met_alone"
    return "missed"


def _final(root: Path, arm: str) -> bool:
    """Whether the condition's dispatcher wrote its final state (any status but its first, running one)."""
    path = root / f"{arm}-state.json"
    return path.is_file() and json.loads(path.read_text()).get("status") not in (None, "running")


def positions(root: Path, arm: str, receipt: dict | None = None) -> dict[str, dict]:
    """episode -> {status, outcome, summary, cost} for one condition on one stream: from its final state record, or,
    when that record is not final, from the host's receipt for its dispatcher rebuilt from the records."""
    path = root / f"{arm}-state.json"
    if _final(root, arm):
        listed = json.loads(path.read_text())["positions"]
    elif receipt is not None:
        if receipt.get("receipts_from") != "records":
            raise ValueError(f"{arm}: the state record is not final, but the host receipt was read from a final one")
        listed = receipt["positions"]
    elif not path.is_file() and not (root / "arms" / arm).exists():
        return {}                         # neither dispatcher state nor a host receipt for this condition
    else:
        raise ValueError(f"{arm}: the state record is not final, and the host receipt that rebuilt it was not given")
    rows = {}
    for p in listed:
        status, summary = p["status"], None
        if status not in KNOWN_COST + UNKNOWN_COST + NO_COST:
            raise ValueError(f"{arm} {p['episode']}: unknown position status {status!r}")
        if p.get("input_matches_schedule") is False or p.get("copy_matches_snapshot") is False:
            raise ValueError(f"{arm} {p['episode']}: the flown inputs differ from the schedule")
        if p.get("summary"):
            file = root / p["summary"]
            if not file.is_file() or _sha(file) != p.get("summary_sha256"):
                raise ValueError(f"{arm} {p['episode']}: the summary differs from its state record")
            summary = json.loads(file.read_text())
        if status in KNOWN_COST and summary is None:
            raise ValueError(f"{arm} {p['episode']}: a {status} position has no summary bound to its record")
        # An interrupted worker's saved summary is kept by the runner as evidence only; its status governs.
        rows[p["episode"]] = {"status": status, "outcome": outcome(summary, status) if status in NATURAL else status,
                              "summary": summary if status in KNOWN_COST else None,
                              "cost": "known" if status in KNOWN_COST else
                                      "unknown" if status in UNKNOWN_COST else "none"}
    return rows


def counts(rows: list[dict]) -> dict:
    """Outcome counts and summed costs and seconds over some positions."""
    out = {"positions": len(rows), "outcomes": {}, "cost": {k: 0 for k in COSTS},
           "known_subtotal": {k: 0 for k in COSTS}, "unknown_calls": {k: 0 for k in TOKENS},
           "calls_with_unknown_usage": 0, "positions_with_unknown_cost": 0, "seconds": {m: 0 for m in METHODS}}
    for r in rows:
        out["outcomes"][r["outcome"]] = out["outcomes"].get(r["outcome"], 0) + 1
        if r.get("cost") == "unknown":
            # Cut off or interrupted while flying: its calls are not all recorded, so no total it enters is known.
            out["positions_with_unknown_cost"] += 1
            out["cost"], out["seconds"] = {k: None for k in COSTS}, None
            continue
        s = r["summary"]
        if s is None:                     # not attempted: no cost was incurred or recorded
            continue
        calls = s.get("model_calls") or 0
        out["known_subtotal"]["model_calls"] += calls
        if out["cost"]["model_calls"] is not None:
            out["cost"]["model_calls"] += calls
        for k in TOKENS:
            value = s.get(k)
            out["known_subtotal"][k] += s.get(f"{k}_known_subtotal", value if value is not None else 0) or 0
            out["unknown_calls"][k] += s.get(f"{k}_unknown_calls", 0 if value is not None else 1) or 0
            out["cost"][k] = None if value is None or out["cost"][k] is None else out["cost"][k] + value
        out["calls_with_unknown_usage"] += s.get("calls_with_unknown_usage") or 0
        if out["seconds"] is not None:
            steps = s.get("action_steps") or 0
            for m in METHODS:
                out["seconds"][m] += round((s.get("share_of_seconds") or {}).get(m, 0) * steps)
    met = out["outcomes"].get("met_alone", 0) + out["outcomes"].get("met_with_help", 0)
    calls = out["cost"]["model_calls"]
    out["model_calls_per_goal_met"] = calls / met if met and calls is not None else None
    return out


def _learning_events(folder: Path) -> tuple[list[dict], dict]:
    """L's learning events from its durable journal (learning-events.jsonl, written and synced event by event), every
    session in order. The final learning-events.json is written only when a session ends normally, so after a cut-off
    it is missing or holds an earlier session; when present it must equal one whole session of the journal, or the
    stream is refused. A last line cut short by the stop is left out and reported; any other unreadable line refuses."""
    journal, final = folder / "learning-events.jsonl", folder / "learning-events.json"
    aggregate = json.loads(final.read_text()) if final.is_file() else None
    if not journal.is_file():
        return aggregate or [], {"source": "learning-events.json" if aggregate is not None else None}
    lines = journal.read_text().split("\n")
    tail = lines.pop()                    # empty when the journal ends with a whole line
    sessions: dict[str, list] = {}
    for line in lines:
        if line.strip():
            row = json.loads(line)
            sessions.setdefault(row["session"], []).append(row["event"])
    if aggregate is not None and aggregate not in sessions.values():
        raise ValueError("L's learning-events.json differs from every session of its durable learning journal")
    return [e for events in sessions.values() for e in events], {
        "source": "learning-events.jsonl", "sessions": len(sessions), "cut_last_line": bool(tail.strip())}


def learning(root: Path, schedule: list[str], open_boundaries: frozenset = frozenset()) -> dict:
    """L's learning attempts and their cost, and each scheduled position's D1 share of that cost. An attempt at a
    boundary the host found open (started, never closed) was cut off, possibly with a call in flight whose usage is
    unknown, so its build cost and every total and share it reaches are unavailable."""
    folder = root / "arms" / "L"
    events, journal = _learning_events(folder)
    ledger = [json.loads(line) for line in (folder / "learning-ledger.jsonl").read_text().splitlines()
              if line.strip()] if (folder / "learning-ledger.jsonl").is_file() else []
    by_ordinal = {e["ordinal"]: e["episode"] for e in events if e.get("kind") == "completed"}
    attempts, n = [], len(schedule)
    for e in events:
        if e.get("kind") == "attempt_start":
            boundary = by_ordinal.get(e["ordinal"])
            attempts.append({"attempt": e["attempt"], "boundary": boundary, "results": [],
                             "open": boundary in open_boundaries})
        elif e.get("kind") == "attempt_result" and attempts and attempts[-1]["attempt"] == e["attempt"]:
            attempts[-1]["results"].append({"decision": e.get("decision"), "kind": e.get("entry_kind"),
                                            "passed": e.get("passed") is True})
    share = {ep: {k: 0.0 for k in COSTS} for ep in schedule}
    builds, known, unknown = {}, {k: 0 for k in COSTS}, {k: 0 for k in TOKENS}
    for a in attempts:
        if a["open"]:
            builds[a["boundary"]] = {k: None for k in COSTS}
    for row in ledger:
        if row.get("scope") != "build":
            continue
        b = builds.setdefault(row["episode"], {k: 0 for k in COSTS})
        known["model_calls"] += 1
        b["model_calls"] = None if b["model_calls"] is None else b["model_calls"] + 1
        for key in TOKENS:
            value = row.get(key)
            if value is None:
                unknown[key] += 1
                b[key] = None
            else:
                known[key] += value
                b[key] = None if b[key] is None else b[key] + value
    for episode, cost in builds.items():
        k = schedule.index(episode)
        for later in schedule[k:]:
            for key in COSTS:
                share[later][key] = (None if cost[key] is None or share[later][key] is None
                                     else share[later][key] + cost[key] / (n - k))
    total = {k: (None if any(b[k] is None for b in builds.values()) else sum(b[k] for b in builds.values()))
             for k in COSTS}
    return {"attempts": attempts, "build_cost_by_boundary": builds, "build_cost_total": total,
            "build_known_subtotal": known, "build_unknown_calls": unknown, "d1_share": share,
            "open_attempts": sum(a["open"] for a in attempts), "journal": journal,
            "floor_breaches": sum(e.get("kind") == "floor_breach" for e in events)}


def stream(root: Path, receipts: dict | None = None) -> dict:
    """Every table for one stream. receipts maps a dispatcher's name (ARM:STREAM) to its host receipt."""
    body = json.loads((root / "schedule.json").read_text())["schedule"]
    schedule = [r["episode"] for r in body["rows"]]
    receipts = receipts or {}
    receipt = lambda arm: receipts.get(f"{arm}:{body['stream']}")
    arms = {arm: positions(root, arm, receipt(arm)) for arm in ARMS}
    # A final L state closed every boundary; otherwise the host's receipt says which boundary the stop left open.
    open_boundaries = frozenset() if _final(root, "L") else frozenset(
        p["episode"] for p in (receipt("L") or {}).get("positions", []) if p.get("boundary") == "open")
    learn = learning(root, schedule, open_boundaries)
    c_eps = [r["episode"] for r in body["rows"] if r["c"]]
    rq1 = {}
    for arm in MATCHED:
        rows = [arms[arm][e] for e in c_eps if e in arms[arm]]
        rq1[arm] = {"pooled": counts(rows),
                    "by_phase": {p: counts([arms[arm][e] for e in c_eps if e in arms[arm] and _phase_index(e)[0] == p])
                                 for p in (1, 2, 3, 4)}}
    rq1["L"]["d1_learning_share_on_matched"] = {
        k: None if any(learn["d1_share"][e][k] is None for e in c_eps) else sum(learn["d1_share"][e][k] for e in c_eps)
        for k in COSTS}
    rq1["L_all_positions"] = {p: counts([arms["L"][e] for e in schedule if e in arms["L"] and _phase_index(e)[0] == p])
                              for p in sorted({_phase_index(e)[0] for e in schedule})}
    rq2 = []
    for phase, before, after in F_BLOCKS:
        prefix = schedule[0].rsplit("-", 2)[0]
        name = lambda i: f"{prefix}-p{phase}-e{i:02d}"
        boundary = name(before[-1])
        at = [a for a in learn["attempts"] if a["boundary"] == boundary]
        after_eps, before_eps = [name(i) for i in after], [name(i) for i in before]
        rq2.append({"boundary": boundary, "attempts": len(at),
                    "adoptions": sum(r["passed"] for a in at for r in a["results"]),
                    "informative": bool(at),
                    "L_after": counts([arms["L"][e] for e in after_eps if e in arms["L"]]),
                    "F_after": counts([arms["F"][e] for e in after_eps if e in arms["F"]]),
                    "L_before": counts([arms["L"][e] for e in before_eps if e in arms["L"]])})
    help_table = {}
    for arm in ARMS:
        for e, r in arms[arm].items():
            phase = _phase_index(e)[0]
            for h in ((r["summary"] or {}).get("outcome") or {}).get("help_requests") or []:
                key = f"{arm} phase {phase} {h.get('reason')}"
                help_table[key] = help_table.get(key, 0) + 1
    return {"stream": body["stream"], "mode": body["mode"],
            "statuses": {arm: _tally(r["status"] for r in arms[arm].values()) for arm in ARMS},
            "rq1": rq1, "learning": {k: v for k, v in learn.items() if k != "d1_share"}, "rq2": rq2,
            "help": help_table}


def _tally(values) -> dict:
    out = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def across(results: list[dict]) -> dict:
    """The four-stream view: each stream's value and their range, for the headline quantities (section 8)."""
    def spread(values):
        known = [v for v in values if v is not None]
        return {"per_stream": values, "range": [min(known), max(known)] if known else None}
    out = {}
    for arm in MATCHED:
        out[f"RQ1 {arm} goals met alone on matched positions"] = spread(
            [r["rq1"][arm]["pooled"]["outcomes"].get("met_alone", 0) for r in results])
        out[f"RQ1 {arm} flight model calls on matched positions"] = spread(
            [r["rq1"][arm]["pooled"]["cost"]["model_calls"] for r in results])
    for b in range(len(F_BLOCKS)):
        for arm in ("L_after", "F_after"):
            out[f"RQ2 block {b + 1} {arm} goals met alone"] = spread(
                [r["rq2"][b][arm]["outcomes"].get("met_alone", 0) for r in results])
        out[f"RQ2 block {b + 1} informative streams"] = spread([int(r["rq2"][b]["informative"]) for r in results])
    return out


def ran(root: Path) -> bool:
    """A stream ran if any condition has a state record there."""
    return any((root / f"{arm}-state.json").is_file() for arm in ARMS)


def disposition(root: Path) -> dict:
    """What happened to a declared stream: ran, or the operator's recorded launch disposition, or not started."""
    if ran(root):
        return {"status": "ran"}
    path = root / "launch-disposition.json"
    return json.loads(path.read_text()) if path.is_file() else {"status": "not started"}


def host_receipts(paths) -> dict:
    """Dispatcher name -> its receipt, from the host-receipts.json of every job that flew the study."""
    out = {}
    for path in paths:
        for d in json.loads(Path(path).read_text())["dispatch"]:
            if d["name"] in out:
                raise ValueError(f"two host receipts name dispatcher {d['name']}")
            out[d["name"]] = d
    return out


def analyse(study: Path, receipts=()) -> dict:
    study = Path(study)
    declared = json.loads((study / "study.json").read_text())["streams"]
    analysed = [s for s in declared if ran(study / str(s))]
    by_name = host_receipts(receipts)
    results = [stream(study / str(s), by_name) for s in analysed]
    return {"study": str(study), "declared_streams": declared, "analysed_streams": analysed,
            "dispositions": {str(s): disposition(study / str(s)) for s in declared},
            "per_stream": results, "across": across(results),
            "host_receipts": {str(path): _sha(Path(path)) for path in receipts},
            "rules": "protocol amendment 7 and sections 7 and 8; no significance test across four streams"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--receipts", type=Path, nargs="*", default=[],
                   help="the host-receipts.json of each job that flew the study; needed for a cut-off stream")
    a = p.parse_args(argv)
    result = analyse(a.study, a.receipts)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "analysis.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps(result["across"], indent=1))


if __name__ == "__main__":
    main()
