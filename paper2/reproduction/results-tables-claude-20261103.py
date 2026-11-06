"""Tables for paper 2's results, read from the saved records of the four scored streams.

Claude, 3 November 2026. Reads only: the frozen analysis output that Codex produced with all four host receipts
(cluster-records/scored-4304-20261102/terminal/frozen-analysis/analysis.json, outside Git, digest checked below)
and Codex's four terminal audit records in this folder. Nothing is recomputed from raw flights, no model or
simulator runs, and no frozen file changes. Primary outcomes, matched costs, phase 3 comparisons and help counts come
from the frozen analysis. Learning counts, per-episode labels and attempt costs come from the audit records.

    python3 paper2/reproduction/results-tables-claude-20261103.py

writes results-tables-claude-20261103.json beside this file.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ANALYSIS = ROOT / "cluster-records/scored-4304-20261102/terminal/frozen-analysis/analysis.json"
ANALYSIS_SHA256 = "415fd80d33371dd5bdecffe69b6bd7c63cbf274f32d8a01a7f1ed072e10ba818"
AUDITS = {
    "4301": "scored-4339375-terminal-codex-20261030.json",
    "4302": "scored-4403902-terminal-codex-20261101.json",
    "4303": "scored-4499845-terminal-codex-20261102.json",
    "4304": "scored-4503929-terminal-codex-20261103.json",
}
OUT = HERE / "results-tables-claude-20261103.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outcome_counts(o):
    return {k: o.get(k, 0) for k in ("met_alone", "met_with_help", "missed")}


def cost(block):
    """Calls and tokens with unknown parts kept apart, as the frozen reader reports them."""
    return {
        "positions": block["positions"],
        "model_calls": block["cost"]["model_calls"],
        "tokens_in": block["cost"]["tokens_in"],
        "tokens_out": block["cost"]["tokens_out"],
        "calls_with_unknown_usage": block["calls_with_unknown_usage"],
        "heading_seconds": block["seconds"],
    }


def main():
    if sha(ANALYSIS) != ANALYSIS_SHA256:
        raise SystemExit("frozen analysis output does not match the audited digest")
    analysis = json.loads(ANALYSIS.read_text())
    audits = {s: json.loads((HERE / f).read_text()) for s, f in AUDITS.items()}
    streams = [str(s["stream"]) for s in analysis["per_stream"]]
    assert streams == list(AUDITS), streams

    out = {
        "inputs": {str(ANALYSIS.relative_to(ROOT)): ANALYSIS_SHA256}
        | {f"paper2/reproduction/{f}": sha(HERE / f) for f in AUDITS.values()},
        "dispositions": analysis["dispositions"],
        "gpu_hours": {s: audits[s]["actual_allocated_scored_gpu_hours"] for s in streams},
        "per_stream": {},
    }

    for ps in analysis["per_stream"]:
        s = str(ps["stream"])
        audit = audits[s]
        rq1 = ps["rq1"]
        row = {}

        # Goals: the learning condition over all 100 episodes and by phase, and the matched 20 for every condition.
        by_phase = {p: outcome_counts(rq1["L_all_positions"][p]["outcomes"]) for p in ("1", "2", "3", "4", "5")}
        row["L_all"] = {k: sum(v[k] for v in by_phase.values()) for k in ("met_alone", "met_with_help", "missed")}
        row["L_by_phase"] = by_phase
        row["matched"] = {arm: outcome_counts(rq1[arm]["pooled"]["outcomes"]) for arm in ("L", "C", "C-full")}

        # Flight cost on the matched 20, and the learning condition's share of learning cost on those episodes.
        row["matched_cost"] = {arm: cost(rq1[arm]["pooled"]) for arm in ("L", "C", "C-full")}
        row["matched_learning_share"] = rq1["L"]["d1_learning_share_on_matched"]
        # Known subtotals only; a timed-out call leaves its tokens unknown, counted separately below.
        row["L_all_flight_cost_known_subtotal"] = {
            k: sum(rq1["L_all_positions"][p]["known_subtotal"][k] for p in ("1", "2", "3", "4", "5"))
            for k in ("model_calls", "tokens_in", "tokens_out")
        }
        row["L_all_flight_unknown_calls"] = sum(
            rq1["L_all_positions"][p]["calls_with_unknown_usage"] for p in ("1", "2", "3", "4", "5"))

        # Learning: complete observed cost, attempts and what happened to each candidate.
        lrn = ps["learning"]
        row["learning_cost"] = {
            "known_subtotal": lrn["build_known_subtotal"],
            "unknown_calls": lrn["build_unknown_calls"],
        }
        row["learning_attempts"] = [
            {"boundary": a["boundary"], "results": [(r["decision"], r["passed"]) for r in a["results"]]}
            for a in lrn["attempts"]
        ]
        row["floor_breaches"] = lrn["floor_breaches"]
        counts = dict(audit["learning"]["learning_candidate_counts"])
        if "format_refused" in counts:  # 4301's audit used an earlier key for the same count
            counts["candidate_refused"] = counts.pop("format_refused")
        row["candidate_counts"] = {k: counts[k] for k in (
            "validly_formed_and_trialled", "trial_passed_and_adopted", "trial_failed", "candidate_refused",
            "adopted_with_later_L_exact_source_use", "adopted_without_later_L_exact_source_use")}

        # Phase 3: learning against the frozen library on the five episodes after each block's boundary.
        row["phase3"] = [
            {
                "block": i + 1,
                "boundary": b["boundary"],
                "attempts": b["attempts"],
                "adoptions": b["adoptions"],
                "informative": b["informative"],
                "L_before": outcome_counts(b["L_before"]["outcomes"]),
                "L_after": outcome_counts(b["L_after"]["outcomes"]),
                "F_after": outcome_counts(b["F_after"]["outcomes"]),
            }
            for i, b in enumerate(ps["rq2"])
        ]

        # Requests for help, by condition, phase and reason.
        row["help"] = ps["help"]

        # Per-episode labels of the learning condition, in stream order, for the trajectory figure.
        pos = [p for p in audit["positions"] if p["arm"] == "L"]
        pos.sort(key=lambda p: p["episode"])
        row["L_episodes"] = [
            {k: p[k] for k in ("episode", "phase", "label", "joined", "report_correct", "help_requests",
                               "model_calls")} | {"heading_seconds": p["heading_seconds"]}
            for p in pos
        ]
        out["per_stream"][s] = row

    # The first learning cycle of the first stream, chosen before the scored run.
    a1 = audits["4301"]
    first = a1["learning"]["first_cycle"]
    cand = next(c for c in a1["learning"]["learning_candidate_results"] if c["candidate"] == first["candidate"])
    att = next(a for a in a1["learning"]["learning_attempts"] if a["attempt"] == first["attempt"])
    out["first_cycle"] = {
        "summary": first,
        "attempt": {k: att[k] for k in ("boundary_episode", "model_calls", "tokens_in", "tokens_out")},
        "recent_trials": cand["recent_trials"],
        "later_episodes_decisions": cand["later_L_episodes"],
    }

    # Totals across the four streams.
    def total(path):
        acc = {}
        for s in streams:
            v = out["per_stream"][s]
            for key in path:
                v = v[key]
            for k, n in v.items():
                acc[k] = acc.get(k, 0) + n
        return acc

    out["totals"] = {
        "L_all": total(["L_all"]),
        "matched": {arm: total(["matched", arm]) for arm in ("L", "C", "C-full")},
        "candidate_counts": total(["candidate_counts"]),
        "gpu_hours": round(sum(out["gpu_hours"].values()), 6),
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["totals"], indent=1))


if __name__ == "__main__":
    main()
