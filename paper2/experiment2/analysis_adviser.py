"""The reasoning model alone beside the learning condition, on the episodes drawn for it before any flight.

Added after the audit of the first scored stream, which showed that the no-learning conditions never put the heading
decision to the reasoning model. The frozen analysis (analysis.py) stays as it is; this module only reads it, and its
own digest is recorded before the first adviser flight. The same prepared starts and the fixed draw (seeds 6001 to
6004) are used, so nothing about any outcome chooses the episodes.

For each stream it gives, on the adviser's episodes, pooled and by phase: goals met alone, met with help and missed,
with model calls and tokens, for the adviser and for L. L's share of learning cost under decision D1 and its full
learning cost stand beside them. Run after the frozen analysis:

    python -m paper2.experiment2.analysis_adviser --study STUDY --out OUT --receipts HOST-RECEIPTS...
"""
import argparse
import json
from pathlib import Path

from . import analysis as A

PHASES = (1, 2, 3, 4, 5)


def adviser_positions(root: Path, receipt: dict | None) -> dict:
    """Read bound summaries, retaining known usage from answered calls when another call has unknown usage.

    Enrichment is in memory only: the frozen adviser summaries and ledgers are never rewritten.
    """
    rows = A.positions(root, "adviser", receipt)
    for episode, row in rows.items():
        if row["summary"] is None:
            continue
        summary = dict(row["summary"])
        ledger = root / "arms" / "adviser" / episode / "ledger.jsonl"
        calls = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
        if len(calls) != summary["model_calls"]:
            raise ValueError(f"adviser {episode}: ledger call count differs from its summary")
        for field in A.TOKENS:
            values = [call.get(field) for call in calls]
            if any(v is not None and (type(v) is not int or v < 0) for v in values):
                raise ValueError(f"adviser {episode}: invalid {field} usage")
            total = None if any(v is None for v in values) else sum(values)
            if total != summary.get(field):
                raise ValueError(f"adviser {episode}: ledger {field} differs from its summary")
            summary[field + "_known_subtotal"] = sum(v for v in values if v is not None)
            summary[field + "_unknown_calls"] = sum(v is None for v in values)
        summary["calls_with_unknown_usage"] = sum(any(c.get(k) is None for k in A.TOKENS) for c in calls)
        row["summary"] = summary
        row["call_records"] = calls
    return rows


def adviser_counts(rows: list[dict]) -> dict:
    """The adviser's method-seconds are explicit counts; procedure and small-model methods are inapplicable."""
    result = A.counts(rows)
    seconds = {"procedure": "not applicable", "small_model": "not applicable",
               "reasoning_model": 0, "holding_pending": 0, "holding_none": 0}
    for row in rows:
        if row.get("cost") == "unknown":
            seconds = None
            break
        if row["summary"] is None:
            continue
        recorded = row["summary"]["method_seconds"]
        if any(recorded.get(k) != "not applicable" for k in ("procedure", "small_model")):
            raise ValueError("adviser method-seconds must mark procedure and small model inapplicable")
        for method in ("reasoning_model", "holding_pending", "holding_none"):
            value = recorded[method]
            if type(value) is not int or value < 0:
                raise ValueError("adviser method-seconds must be nonnegative whole counts")
            seconds[method] += value
    result["seconds"] = seconds
    return result


def stream(root: Path, receipts: dict | None = None) -> dict:
    """The adviser and L on the adviser's episodes of one stream. receipts maps ARM:STREAM to its host receipt."""
    body = json.loads((root / "schedule.json").read_text())["schedule"]
    schedule = [r["episode"] for r in body["rows"]]
    matched = [r["episode"] for r in body["rows"] if r["adviser"]]
    receipts = receipts or {}
    receipt = lambda arm: receipts.get(f"{arm}:{body['stream']}")
    arms = {"adviser": adviser_positions(root, receipt("adviser")), "L": A.positions(root, "L", receipt("L"))}
    if not arms["adviser"]:
        return {"stream": body["stream"], "adviser": "not flown"}
    open_boundaries = frozenset() if A._final(root, "L") else frozenset(
        p["episode"] for p in (receipt("L") or {}).get("positions", []) if p.get("boundary") == "open")
    learn = A.learning(root, schedule, open_boundaries)
    table = {}
    for arm, rows in arms.items():
        on = [e for e in matched if e in rows]
        count = adviser_counts if arm == "adviser" else A.counts
        table[arm] = {"pooled": count([rows[e] for e in on]),
                      "by_phase": {p: count([rows[e] for e in on if A._phase_index(e)[0] == p]) for p in PHASES}}
    table["L"]["d1_learning_share_on_matched"] = {
        k: None if any(learn["d1_share"][e][k] is None for e in matched) else sum(learn["d1_share"][e][k] for e in matched)
        for k in A.COSTS}
    table["L"]["full_learning_cost"] = {"total": learn["build_cost_total"], "known_subtotal": learn["build_known_subtotal"]}
    common = [r["episode"] for r in body["rows"] if r["adviser"] and r["c"]]
    shared = {}
    for arm in ("adviser", "L", "C", "C-full"):
        rows = arms[arm] if arm in arms else A.positions(root, arm, receipt(arm))
        count = adviser_counts if arm == "adviser" else A.counts
        shared[arm] = count([rows[e] for e in common if e in rows])
    shared["L"]["d1_learning_share_on_matched"] = {
        k: None if any(learn["d1_share"][e][k] is None for e in common) else sum(learn["d1_share"][e][k] for e in common)
        for k in A.COSTS}
    provider_calls = [
        {"episode": episode, "phase": A._phase_index(episode)[0], "call_position": i + 1,
         **{k: call.get(k) for k in ("request_id", "status", "sim_second", "answered_step", "wall_s", "tokens_in", "tokens_out")}}
        for episode, row in arms["adviser"].items() for i, call in enumerate(row.get("call_records", []))]
    timeout_calls = [call for call in provider_calls if call["status"] in ("timed_out", "cut_off")]
    return {"stream": body["stream"], "mode": body["mode"], "matched_episodes": len(matched),
            "statuses": {arm: A._tally(r["status"] for r in rows.values()) for arm, rows in arms.items()},
            "table": table, "common_with_C": {"matched_episodes": len(common), "episode_ids": common, "table": shared},
            "recorded_provider_calls": provider_calls, "recorded_timeout_or_cutoff_calls": timeout_calls,
            "call_records_complete": not any(row.get("cost") == "unknown" for row in arms["adviser"].values())}


def analyse(study: Path, receipts=()) -> dict:
    declared = json.loads((study / "study.json").read_text())["streams"]
    held = A.host_receipts(receipts)
    results = [stream(study / str(s), held) for s in declared]
    counterfactual = study / "counterfactual-stops/adviser.json"
    return {"streams": results, "analysis_sha256": A._sha(Path(A.__file__)), "this_module_sha256": A._sha(Path(__file__)),
            "adviser_counterfactual_stop": json.loads(counterfactual.read_text()) if counterfactual.is_file() else None}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--receipts", type=Path, nargs="*", default=[])
    a = p.parse_args(argv)
    result = analyse(a.study, a.receipts)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "analysis-adviser.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
