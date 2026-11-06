"""What the flight records show behind paper 2's result tables: which decision each model was asked about, what the
small model answered for the heading, who set the report and when help was asked, and how the misses split.

Claude, 3 November 2026. Reads only the preserved originals of the four scored streams
(cluster-records/scored-4304-20261102/terminal/originals/, outside Git), Codex's four terminal audit records in this
folder, results-tables-claude-20261103.json and the frozen analysis output (digest checked below). No simulator, model, candidate or generator runs, and no record changes.

    python3 paper2/reproduction/flight-records-claude-20261103.py

writes flight-records-claude-20261103.json beside this file.
"""
import collections
import glob
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STUDY = ROOT / "cluster-records/scored-4304-20261102/terminal/originals/scored-4301-4304-20261029"
TABLES = HERE / "results-tables-claude-20261103.json"
ANALYSIS = ROOT / "cluster-records/scored-4304-20261102/terminal/frozen-analysis/analysis.json"
ANALYSIS_SHA256 = "415fd80d33371dd5bdecffe69b6bd7c63cbf274f32d8a01a7f1ed072e10ba818"
AUDITS = {
    "4301": "scored-4339375-terminal-codex-20261030.json",
    "4302": "scored-4403902-terminal-codex-20261101.json",
    "4303": "scored-4499845-terminal-codex-20261102.json",
    "4304": "scored-4503929-terminal-codex-20261103.json",
}
STREAMS = tuple(AUDITS)
OUT = HERE / "flight-records-claude-20261103.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episode_dirs(stream, arm):
    """Folders holding gateway.log, episode.log and summary.json for each episode of one condition."""
    pat = f"s{stream}-p*-e*" if arm == "L" else f"s{stream}-p*-e*/flight/s{stream}-p*-e*"
    dirs = sorted(Path(p) for p in glob.glob(str(STUDY / stream / "arms" / arm / pat)) if Path(p).is_dir())
    return [d for d in dirs if (d / "summary.json").is_file()]


def records(path):
    for line in open(path):
        row = json.loads(line)
        yield row.get("payload", row)


def model_requests(d):
    """Each flight model request: role, the decision asked about, and for headings the commanded and replied values."""
    asked, replies = {}, []
    for p in records(d / "gateway.log"):
        if p.get("kind") == "model_start":
            req = p["request"]
            msg = req["messages"][-1][1]
            heading = msg.startswith("Choose the heading")
            commanded = json.loads(msg.split("Context:\n", 1)[1])["commanded_heading_deg"] if heading else None
            asked[req["request_id"]] = (req["role"], "heading" if heading else "report", commanded)
        elif p.get("kind") == "model_end" and p["request_id"] in asked:
            role, decision, commanded = asked[p["request_id"]]
            reply = None
            if decision == "heading":
                m = re.search(r'"heading_deg"\s*:\s*(-?[0-9.]+)', p["response"].get("text") or "")
                reply = float(m.group(1)) if m else None
            replies.append((role, decision, commanded, reply))
    # A request with no recorded end, such as a timeout, still counts as asked.
    return asked, replies


def episode_events(d):
    """For the report decision, the first accepted output from each source and the first that armed a trigger or sent a
    report (a procedure may return an empty output, which does neither), then the first help request and the outcome."""
    report_output, report_armed, help_first = {}, {}, None
    for p in records(d / "episode.log"):
        k = p.get("kind")
        if k == "decision" and p["decision"] == "report" and p["source"] != "none":
            report_output.setdefault(p["source"], p["t"])
            # Armed only when the output carries a real trigger value or report text; {} and
            # {"report_trigger_nm": null} do neither.
            if any(v not in (None, "") for v in (p.get("part") or {}).values()):
                report_armed.setdefault(p["source"], p["t"])
        elif k == "help_request" and help_first is None:
            help_first = {"t": p["t"], "reason": p["reason"]}
    o = json.loads((d / "summary.json").read_text())["outcome"]
    return report_output, report_armed, help_first, o


def main():
    tables = json.loads(TABLES.read_text())
    if sha(ANALYSIS) != ANALYSIS_SHA256:
        raise SystemExit("frozen analysis output does not match the audited digest")
    analysis = json.loads(ANALYSIS.read_text())
    audits = {s: json.loads((HERE / f).read_text()) for s, f in AUDITS.items()}
    out = {"inputs": {str(ANALYSIS.relative_to(ROOT)): ANALYSIS_SHA256, str(TABLES.relative_to(ROOT)): sha(TABLES)} | {f"paper2/reproduction/{f}": sha(HERE / f)
                                                                      for f in AUDITS.values()},
           "originals": str(STUDY.relative_to(ROOT)), "per_stream": {}, "totals": {}}

    for s in STREAMS:
        schedule = {r["episode"]: r for r in json.loads((STUDY / s / "schedule.json").read_text())["schedule"]["rows"]}
        row = {"model_requests": {}, "small_model_heading_replies": {}}

        # 1 and 2. Which decision each model was asked about in flight, and whether the small model's heading reply
        # simply repeated the heading already commanded.
        for arm in ("L", "C", "C-full"):
            asked_by = collections.Counter()
            near = exact = changed = unparsed = 0
            for d in episode_dirs(s, arm):
                asked, replies = model_requests(d)
                for role, decision, _c in asked.values():
                    asked_by[f"{role} {decision}"] += 1
                for role, decision, commanded, reply in replies:
                    if role != "small_model" or decision != "heading":
                        continue
                    if reply is None:
                        unparsed += 1
                    elif abs((reply - commanded + 180) % 360 - 180) < 0.5:
                        near += 1
                        exact += reply == commanded
                    else:
                        changed += 1
            row["model_requests"][arm] = dict(sorted(asked_by.items()))
            # Within half a degree of the commanded heading, by the expression above; exact equality counted apart.
            row["small_model_heading_replies"][arm] = {"within_half_degree_of_commanded": near,
                                                       "exactly_commanded": exact, "changed": changed,
                                                       "unparsed": unparsed}

        # 3. In every episode of every condition: who first armed a report trigger or sent a report, and whether help
        # was asked before or after that.
        episodes = {}
        for arm in ("L", "C", "C-full"):
            for d in episode_dirs(s, arm):
                output, armed, help_first, o = episode_events(d)
                ep = d.name
                first_armed_t = min(armed.values()) if armed else None
                label = "missed" if not o["goal_met"] else ("met_with_help" if o["help_requests"] else "met_alone")
                episodes[(arm, ep)] = {
                    "arm": arm, "episode": ep, "phase": int(ep.split("-p")[1][0]), "label": label,
                    "joined": o["joined"], "report_correct": bool(o["report_correct"]),
                    "end_reason": o["end_reason"], "ended_at": o["ended_at"], "deadline": o["goal"].get("deadline"),
                    "report_output_sources": sorted(output, key=output.get),
                    "report_armed_by": sorted(armed, key=armed.get), "report_first_armed_t": first_armed_t,
                    "help": help_first,
                    "help_after_report_armed": (help_first["t"] > first_armed_t)
                    if help_first and first_armed_t is not None else None,
                }
        row["episodes"] = sorted(episodes.values(), key=lambda e: (e["arm"], e["episode"]))

        # The learning condition's goals met with help: who set the report, and the request reason and order.
        L = [e for e in row["episodes"] if e["arm"] == "L"]
        mwh = [e for e in L if e["label"] == "met_with_help"]
        row["L_met_with_help"] = {
            "episodes": len(mwh),
            "report_first_armed_by": dict(collections.Counter(e["report_armed_by"][0] for e in mwh
                                                              if e["report_armed_by"])),
            "help_reason": dict(collections.Counter(e["help"]["reason"] for e in mwh)),
            "help_strictly_after_report_armed": sum(1 for e in mwh if e["help_after_report_armed"] is True),
        }

        # 4. Misses in the learning condition by phase and kind.
        first_heading = next(a["boundary"] for a in tables["per_stream"][s]["learning_attempts"]
                             if any(dec == "heading" and ok for dec, ok in a["results"]))
        cold = {e["episode"] for e in L if e["episode"] <= first_heading}
        kinds = collections.Counter()
        for e in L:
            if e["label"] != "missed":
                continue
            if e["episode"] in cold:
                kind = "before_first_heading_procedure"
            elif e["joined"]:
                kind = "joined_report_wrong"
            elif e["report_correct"]:
                kind = "no_join_report_right"
            else:
                kind = "no_join_report_wrong"
            kinds[(e["phase"], kind)] += 1
        row["L_misses"] = {f"phase {p} {k}": n for (p, k), n in sorted(kinds.items())}
        row["first_heading_boundary"] = first_heading

        # Misses after a heading procedure existed where the aircraft never joined: how each episode ended.
        row["L_no_join_after_heading_procedure"] = [
            {k: e[k] for k in ("episode", "phase", "end_reason", "ended_at", "deadline", "report_correct")}
            for e in L if not e["joined"] and e["episode"] not in cold]

        # The disturbances in phases 3 and 4 behind each miss of the learning condition.
        def disturbance(ep):
            sc = schedule[ep]["scenario"]
            parts = []
            if any(abs(t["orbit"]["delta"]) > 0 for t in sc["tankers"]):
                parts.append("orbit change")
            if sc["dropouts"]:
                parts.append("report gap")
            if sc["delay_s"]:
                parts.append(f"delay {sc['delay_s']} s")
            if len(sc["goals"]) > 1:
                parts.append("goal change")
            return ", ".join(parts) or "wind only"
        row["L_misses_phase_3_4"] = [
            {"episode": e["episode"], "joined": e["joined"], "report_correct": e["report_correct"],
             "disturbance": disturbance(e["episode"])}
            for e in L if e["label"] == "missed" and e["phase"] in (3, 4)]

        # 5. Phase 5 requests in the learning condition.
        p5 = [e for e in L if e["phase"] == 5]
        req = [e for e in p5 if e["help"]]
        row["phase5"] = {
            "episodes": len(p5), "with_request": len(req),
            "request_reasons": dict(collections.Counter(e["help"]["reason"] for e in req)),
            "request_t_min": min(e["help"]["t"] for e in req), "request_t_max": max(e["help"]["t"] for e in req),
            "seconds_left_min": min(e["deadline"] - e["help"]["t"] for e in req),
            "seconds_left_max": max(e["deadline"] - e["help"]["t"] for e in req),
            "goal_met": sum(e["label"] != "missed" for e in p5), "joined": sum(e["joined"] for e in p5),
            "report_correct": sum(e["report_correct"] for e in p5),
            "report_correct_without_join": sum(e["report_correct"] and not e["joined"] for e in p5),
        }

        # 6. Requests by condition, phase and reason, with whether a report trigger or report had already been issued.
        hr = collections.Counter()
        for e in row["episodes"]:
            if e["help"]:
                order = {True: "after report armed", False: "before report armed", None: "report never armed"}[
                    e["help_after_report_armed"]]
                hr[f"{e['arm']} phase {e['phase']} {e['help']['reason']} {order}"] += 1
        row["help_requests"] = dict(sorted(hr.items()))

        # 7. Flight model calls of the learning condition by phase, and the first episode from which none was made.
        audit = audits[s]
        Lpos = sorted((p for p in audit["positions"] if p["arm"] == "L"), key=lambda p: p["episode"])
        calls = collections.Counter()
        for p in Lpos:
            calls[p["phase"]] += p["model_calls"]
        last_call = max(p["episode"] for p in Lpos if p["model_calls"] > 0)
        row["L_flight_calls_by_phase"] = {str(k): v for k, v in sorted(calls.items())}
        row["L_last_episode_with_a_flight_call"] = last_call
        row["L_episodes_after_last_flight_call"] = sum(1 for p in Lpos if p["episode"] > last_call)
        row["unknown_usage_positions"] = [
            {k: p[k] for k in ("arm", "episode", "model_calls", "calls_with_unknown_usage")}
            for p in audit["positions"] if p.get("calls_with_unknown_usage")]

        # 8. Frozen library episodes in phase 3: joins and reports.
        row["F_episodes"] = [{k: p[k] for k in ("episode", "label", "joined", "report_correct")}
                             for p in sorted(audit["positions"], key=lambda p: p["episode"]) if p["arm"] == "F"]

        # 9. Adopted candidates and their trials.
        adopted = [c for c in audit["learning"]["learning_candidate_results"] if c["passed"]]
        row["adopted"] = [
            {"decision": c["decision"], "boundary_ordinal": c["boundary_ordinal"],
             "recent_trials": len(c["recent_trials"]),
             "recent_trials_whole_goal_met": sum(t["full_goal"] for t in c["recent_trials"]),
             "retention_trials": len(c["regression_trials"]),
             "later_L_decisions": c["later_L_exact_source_decisions"], "later_L_episodes": len(c["later_L_episodes"])}
            for c in adopted]
        row["trial_counts"] = audit["learning"]["trial_counts"]

        # 10. Matched starts of the learning condition flown before its first heading procedure.
        matched = [e for e in L if schedule[e["episode"]]["c"]]
        row["matched_L_before_first_heading_procedure"] = [
            {k: e[k] for k in ("episode", "label")} for e in matched if e["episode"] in cold]
        # 11. Learning cost and adoptions by phase, and the phases of the episodes each candidate was tried on.
        # Known subtotals per attempt from the audit record, so a timed-out call does not hide the answered calls
        # beside it; unknown calls are counted apart.
        lc = collections.defaultdict(collections.Counter)
        for a in audit["learning"]["learning_attempts"]:
            ph = a["boundary_episode"].split("-p")[1][0]
            # The first audit record has no separate known-subtotal fields; it has no unknown usage, so its totals are
            # complete and serve as the known subtotals.
            known_in = a.get("tokens_in_known_subtotal", a["tokens_in"])
            known_out = a.get("tokens_out_known_subtotal", a["tokens_out"])
            lc[ph].update({"model_calls": a["model_calls"], "tokens_in_known_subtotal": known_in,
                           "tokens_out_known_subtotal": known_out,
                           "calls_with_unknown_usage": a["calls_with_unknown_usage"]})
        row["learning_cost_by_phase"] = {ph: dict(c) for ph, c in sorted(lc.items())}
        ad = collections.Counter()
        tried_on = collections.defaultdict(collections.Counter)
        for c in audit["learning"]["learning_candidate_results"]:
            ph = "5" if c["boundary_ordinal"] > 80 else str((c["boundary_ordinal"] - 1) // 20 + 1)
            if c["passed"]:
                ad[f"phase {ph} {c['decision']}"] += 1
            for t in c["recent_trials"]:
                tried_on[f"boundary in phase {ph}"][f"trial episode in phase {t['episode'].split('-p')[1][0]}"] += 1
        row["adoptions_by_phase"] = dict(sorted(ad.items()))
        row["candidate_trial_episode_phases"] = {k: dict(v) for k, v in sorted(tried_on.items())}

        # 12. Flight calls and tokens by model role on the matched starts, from the gateway records.
        role_cost = {}
        for arm in ("L", "C", "C-full"):
            acc = collections.defaultdict(collections.Counter)
            for d in episode_dirs(s, arm):
                if not schedule[d.name]["c"]:
                    continue
                for p in records(d / "gateway.log"):
                    if p.get("kind") == "model_start":
                        acc[p["request"]["role"]]["calls"] += 1
                    elif p.get("kind") == "model_end":
                        r = p["response"]
                        role = json.loads(p["request_id"])[3]
                        acc[role]["tokens_in"] += r.get("tokens_in") or 0
                        acc[role]["tokens_out"] += r.get("tokens_out") or 0
            role_cost[arm] = {k: dict(v) for k, v in sorted(acc.items())}
        row["matched_flight_cost_by_role"] = role_cost
        out["per_stream"][s] = row

    # Totals across the four streams.
    t = out["totals"]
    for arm in ("L", "C", "C-full"):
        acc = collections.Counter()
        for s in STREAMS:
            acc.update(out["per_stream"][s]["model_requests"][arm])
        t[f"model_requests_{arm}"] = dict(sorted(acc.items()))
        acc = collections.Counter()
        for s in STREAMS:
            acc.update(out["per_stream"][s]["small_model_heading_replies"][arm])
        t[f"small_model_heading_replies_{arm}"] = dict(acc)
    acc = collections.Counter()
    for s in STREAMS:
        acc.update(out["per_stream"][s]["L_misses"])
    t["L_misses"] = dict(sorted(acc.items()))
    t["L_met_with_help"] = {
        "episodes": sum(out["per_stream"][s]["L_met_with_help"]["episodes"] for s in STREAMS),
        "help_strictly_after_report_armed": sum(
            out["per_stream"][s]["L_met_with_help"]["help_strictly_after_report_armed"] for s in STREAMS),
    }
    acc = collections.Counter()
    for s in STREAMS:
        acc.update(out["per_stream"][s]["help_requests"])
    t["help_requests"] = dict(sorted(acc.items()))
    t["phase5"] = {
        "request_t_min": min(out["per_stream"][s]["phase5"]["request_t_min"] for s in STREAMS),
        "request_t_max": max(out["per_stream"][s]["phase5"]["request_t_max"] for s in STREAMS),
        "seconds_left_min": min(out["per_stream"][s]["phase5"]["seconds_left_min"] for s in STREAMS),
        "seconds_left_max": max(out["per_stream"][s]["phase5"]["seconds_left_max"] for s in STREAMS),
    } | {k: sum(out["per_stream"][s]["phase5"][k] for s in STREAMS)
         for k in ("episodes", "with_request", "goal_met", "joined", "report_correct", "report_correct_without_join")}
    t["adopted_with_a_trial_missing_whole_goal"] = sum(
        1 for s in STREAMS for a in out["per_stream"][s]["adopted"]
        if a["recent_trials_whole_goal_met"] < a["recent_trials"])
    t["trial_episodes_component_pass_whole_goal_missed"] = sum(
        out["per_stream"][s]["trial_counts"]["adopted_recent_component_pass_but_whole_goal_not_met"] for s in STREAMS)
    t["adopted_without_later_use_at_last_boundary"] = sum(
        1 for s in STREAMS for a in out["per_stream"][s]["adopted"]
        if a["later_L_episodes"] == 0 and a["boundary_ordinal"] == 100)
    t["adopted_without_later_use"] = sum(
        1 for s in STREAMS for a in out["per_stream"][s]["adopted"] if a["later_L_episodes"] == 0)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["totals"], indent=1))


if __name__ == "__main__":
    main()
