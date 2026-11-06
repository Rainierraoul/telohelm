"""Read preserved scored records; never run a flight, candidate, model or generator.

Usage: Python 3.10 on Linux: python3 <this file> [terminal-record-directory]
The frozen wheel supplies only context reconstruction and saved-log attribution.
The companion reviewers independently verify all originals, chains, requests,
trials and source bindings. This reader adds outcome and public-view joins.
Mac libm/Python versions can give different final float bits. Keep those strict
failures, and use the separately hash-bound native Linux context report to
resolve them rather than applying a numerical tolerance or changing a label.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
SOURCE = "28a5acd769452ef4eb44fdaa370481abce3c51b7"
WHEEL = "bb2c6e779b7526c85fb6ced29bd0db01f0d30a88a16e2252f2f058cab3ce2f8c"


def read(p):
    return json.loads(Path(p).read_text())


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def payloads(path):
    with Path(path).open() as f:
        for line in f:
            item = json.loads(line)
            if isinstance(item.get("payload"), dict):
                yield item["payload"]


def main():
    import zipfile
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "cluster-records/scored-4301-20261029/terminal"
    study = base / "originals/scored-4301-4304-20261029/4301"
    wheel, = (base / "originals/telohelm-study-scored-4301-28a5acd7-2649b430").rglob("*.whl")
    assert digest(wheel) == WHEEL
    members = ("runtime/attribution.py", "runtime/trusted_log.py", "taskpacks/tanker/context.py",
               "taskpacks/tanker/public.py", "taskpacks/tanker/pack.py")
    with zipfile.ZipFile(wheel) as z:
        for member in members:
            pinned = subprocess.check_output(["git", "show", f"{SOURCE}:paper2/telohelm/src/telohelm/{member}"], cwd=REPO)
            assert pinned == z.read("telohelm/" + member), member
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(wheel))
    from telohelm.runtime.attribution import derive
    from telohelm.taskpacks.tanker.context import ContextBuilder
    from telohelm.taskpacks.tanker.public import TANKER
    from telohelm.taskpacks.tanker.pack import TANKER_GOAL
    from telohelm.examples import submitted_at
    from telohelm.taskpacks.tanker.examples import TANKER_EXAMPLES

    problems, positions, steps_checked, decisions_checked, contexts_checked = [], [], 0, 0, 0
    flight_requests_checked, learning_examples_checked, learning_feedback_checked = 0, 0, 0
    source_use = Counter()
    source_use_by_phase = defaultdict(Counter)
    labels = defaultdict(Counter)
    checks = Counter()

    def check(condition, name, detail):
        checks[name] += 1
        if not condition:
            problems.append({"check": name, "record": detail})

    schedule = read(study / "schedule.json")["schedule"]
    assert schedule["mode"] == "scored"
    rows = {r["episode"]: r for r in schedule["rows"]}
    ordinals = {r["episode"]: i + 1 for i, r in enumerate(schedule["rows"])}
    assert len(rows) == 100
    aliases = {"episode-" + hashlib.sha256(f"telohelm-example-alias|{e}".encode()).hexdigest()[:8]: e for e in rows}
    learning_requests, example_points = [], defaultdict(set)
    events = read(study / "arms/L/learning-events.json")
    boundaries = {e["attempt"]: e["ordinal"] for e in events if e["kind"] == "attempt_start"}
    reasons = {e["attempt"]: e["reasons"] for e in events if e["kind"] == "attempt_start"}
    for folder in sorted((study / "arms/L").glob("learning-attempt-*")):
        for e in payloads(folder / "gateway.log"):
            if e.get("kind") != "model_start":
                continue
            q = e["request"]
            body = json.loads(q["messages"][1][1])
            attempt = int(folder.name.rsplit("-", 1)[1])
            learning_requests.append((attempt, q, body))
            check(body["context_fields"] == sorted(TANKER.context_fields), "learner_field_list", q["request_id"])
            for example in body.get("examples", []):
                ep = aliases.get(example["episode"])
                check(ep is not None, "example_alias", q["request_id"])
                if ep is not None:
                    example_points[ep].add(example["second"])
                    check(ordinals[ep] <= boundaries[attempt], "no_future_example", q["request_id"])

    public_examples, example_action_views = {}, {}
    summaries_l = {}
    for arm, count in (("L", 100), ("C", 20), ("C-full", 20), ("F", 10)):
        state = read(study / f"{arm}-state.json")
        check(len(state["positions"]) == count, "scheduled_position_count", arm)
        for pos in state["positions"]:
            ep, row = pos["episode"], rows[pos["episode"]]
            path = study / pos["summary"]
            folder, saved = path.parent, read(path)
            out = saved["outcome"]
            where = f"{arm}/{ep}"
            check(pos["status"] == "complete" and pos["input_matches_schedule"], "closed_scheduled_input", where)
            check(digest(path) == pos["summary_sha256"], "summary_identity", where)
            check(saved["complete"] and saved["settled"] and out["settled"] and not saved["censored_by_step_limit"], "complete_settled", where)
            check(not out["flags"] and not out["alternatives"], "no_numerical_flags_or_alternative_endings", where)
            for key in ("goal_met", "joined", "report_correct", "end_reason", "ended_at", "action_steps"):
                check(saved[key] == out[key], "summary_outcome_" + key, where)
            check(saved["help_requests"] == len(out["help_requests"]), "help_total", where)
            reports = [r for r in out["reports"] if r["goal_version"] == out["goal"]["version"]]
            in_time = out["goal"]["deadline"] is None or (out["joined"] and out["ended_at"] <= out["goal"]["deadline"])
            one_correct = len(reports) == 1 and reports[0]["correct"]
            check(out["reports_under_goal"] == len(reports) and out["report_correct"] == bool(one_correct), "current_goal_report_label", where)
            check(out["in_time"] == in_time and out["goal_met"] == bool(out["joined"] and in_time and one_correct), "whole_goal_boolean_label", where)
            check(out["possible_goal_met"] == out["goal_met_options"] == [out["goal_met"]], "settled_goal_options", where)
            check(out["joined"] == (out["end_reason"] == "join"), "joined_ending", where)
            for report in reports:
                g, text = out["goal"], report["text"].lower()
                content = (row["scenario"]["control_callsign"].lower() in text and "aircraft 2" in text
                           and g["tanker"].lower() in text
                           and any(abs(float(n) - g["report_at_nm"]) <= 0.01 for n in re.findall(r"\d+(?:\.\d+)?", text)))
                check(math.isfinite(report["true_range"]) and math.isfinite(report["range_derivative_nm_s"]), "finite_report_measurements", where)
                correct = content and report["range_derivative_nm_s"] < 0 and abs(report["true_range"] - g["report_at_nm"]) <= 0.25
                check(report["content_ok"] == content and report["inbound"] == (report["range_derivative_nm_s"] < 0)
                      and report["correct"] == bool(correct) and report["correct_options"] == [bool(correct)], "report_rule_on_recorded_measurements", where)

            requests = defaultdict(list)
            for e in payloads(folder / "gateway.log"):
                if e.get("kind") == "model_start":
                    requests[e["request"]["action_step"]].append(e["request"])
            actions, world_steps, full_context, builder, final_goal = [], [], None, None, None
            decision_sources, goal_changes = {}, []
            for e in payloads(folder / "broker.log"):
                kind = e.get("kind")
                if kind == "briefing":
                    builder = ContextBuilder(e["briefing"])
                    final_goal = e["briefing"]["goal"]
                elif kind == "bundle":
                    bundle = e["bundle"]
                    check(set(bundle) == {"t", "goal", "own", "reports", "refusals"}, "public_bundle_schema", where)
                    full_context = builder.update(bundle)
                    contexts_checked += 1
                    if bundle["goal"] is not None:
                        final_goal = bundle["goal"]
                        if bundle["t"] > 0:
                            goal_changes.append(bundle["t"])
                    for radar in bundle["reports"]:
                        required = {"callsign", "x", "y", "measured_at", "received_at"}
                        check(required <= set(radar) <= required | {"help_reply"}
                              and ("help_reply" not in radar or radar["help_reply"] is True)
                              and radar["measured_at"] <= radar["received_at"] <= bundle["t"], "delivered_radar_only", where)
                    if arm == "L" and e["t"] in example_points[ep]:
                        public_examples[(ep, e["t"])] = {**full_context, "reports": full_context["reports"][-10:]}
                    for q in requests.get(e["action_step"], []):
                        user = q["messages"][1][1]
                        shown = json.loads(user.split("\nContext:\n", 1)[1]) if "\nContext:\n" in user else json.loads(user)["context"]
                        expected = {k: v for k, v in full_context.items() if k != "reports"}
                        expected["recent_reports"] = full_context["reports"][-5:]
                        check(set(expected) <= set(shown) and all(shown[k] == v for k, v in expected.items()), "flight_model_observed_context", q["request_id"])
                        check(set(shown) - set(expected) <= {"reply_delay_s", "last_action_refusal"}, "no_hidden_model_context_fields", q["request_id"])
                        check(shown.get("reply_delay_s") == q["modelled_latency_s"], "model_delay_visible", q["request_id"])
                        if "last_action_refusal" in shown:
                            refusal = shown["last_action_refusal"]
                            check(refusal in saved["model_action_refusals"] and refusal["t"] <= q["sim_second"]
                                  and refusal["goal_version"] == q["goal_version"], "refusal_is_prior_public_record", q["request_id"])
                        flight_requests_checked += 1
                elif kind == "decision":
                    decision_sources[(e["t"], e["decision"])] = e["source"]
                    check(e["context_sha256"] == hashlib.sha256(canonical(full_context).encode()).hexdigest(), "procedure_check_context_from_observations", where)
                    decisions_checked += 1
                    if e["source"] == "procedure":
                        source_use[e["method"]] += 1
                        source_use_by_phase[str(row["phase"])][e["method"]] += 1
                elif kind == "action":
                    actions.append(e)
                elif kind == "world_step":
                    world_steps.append(e)
                elif kind == "feedback":
                    f = e["feedback"]
                    check(f["goal_met"] == out["goal_met"] and f["failed_part"] == TANKER_GOAL.failed(out), "broker_feedback_goal_part", where)
                elif kind == "ending":
                    check(e["end_reason"] == out["end_reason"] and e["ended_at"] == out["ended_at"]
                          and e["action_steps"] == out["action_steps"], "protected_ending", where)
            check(out["goal"] == final_goal, "final_observed_goal", where)
            if arm == "L":
                action_map = {e["t"]: e for e in actions}
                step_map = {e["t"]: e for e in world_steps}
                for t in example_points[ep]:
                    example_action_views[(ep, t)] = {
                        "submitted": submitted_at(t, action_map, decision_sources),
                        "applied": step_map[t]["accepted"],
                        **TANKER_EXAMPLES.details(t, out, action_map, decision_sources, step_map, goal_changes)}
            check(len(actions) == len(world_steps) == len(out["actions"]) == saved["action_steps"], "accepted_action_sequence_length", where)
            for action, step, result in zip(actions, world_steps, out["actions"]):
                check(action["t"] == result["t"] and action["action"] == result["actions"]
                      and step["accepted"] == result["accepted"], "saved_action_world_verdict", where)
                steps_checked += 1
            ends = [e for e in payloads(folder / "episode.log") if e.get("kind") == "episode_end"]
            check(len(ends) == 1 and all(ends[0][k] == out[k] for k in ("goal_met", "settled", "end_reason")), "protected_episode_outcome", where)
            attributed = derive(folder / "broker.log", folder / "broker.heads", ep)
            check(attributed["complete"] and attributed["K"] == saved["seconds"] == saved["action_steps"]
                  and attributed["shares"] == saved["share_of_seconds"], "heading_attribution_rederived", where)
            label = "met_with_help" if saved["goal_met"] and saved["help_requests"] else "met_alone" if saved["goal_met"] else "missed"
            labels[f"{arm}/phase{row['phase']}"][label] += 1
            position = {"arm": arm, "episode": ep, "phase": row["phase"], "label": label,
                        "joined": saved["joined"], "report_correct": saved["report_correct"], "help_requests": saved["help_requests"],
                        "end_reason": saved["end_reason"], "ended_at": saved["ended_at"], "steps": saved["action_steps"],
                        "heading_seconds": attributed["counts"], "summary_sha256": digest(path)}
            positions.append(position)
            if arm == "L":
                summaries_l[ep] = saved

    for attempt, request, body in learning_requests:
        for example in body.get("examples", []):
            ep = aliases[example["episode"]]
            saved = summaries_l[ep]
            check(example["context"] == public_examples[(ep, example["second"])], "learner_example_observed_context", request["request_id"])
            failed = None if saved["goal_met"] else TANKER_GOAL.failed(saved["outcome"])
            check(example["outcome"] == {"goal_met": saved["goal_met"], **({"failed": failed} if failed else {})}, "learner_example_public_feedback", request["request_id"])
            expected = {"episode": example["episode"], "second": example["second"],
                        "context": public_examples[(ep, example["second"])], "outcome": example["outcome"],
                        **example_action_views[(ep, example["second"])]}
            check(example == expected, "learner_example_action_and_radio_fields", request["request_id"])
            learning_examples_checked += 1
        expected_feedback = [{**r, "episode": next(alias for alias, ep in aliases.items() if ep == r["episode"])}
                             for r in reasons[attempt]]
        check(body["feedback"] == expected_feedback, "learner_feedback_exact_attempt_reasons", request["request_id"])
        for feedback in body["feedback"]:
            ep = aliases.get(feedback["episode"])
            check(ep is not None and ordinals[ep] <= boundaries[attempt], "feedback_from_completed_own_episode", request["request_id"])
            if ep is not None:
                check(feedback["failed"] == "floor" or feedback["failed"] == TANKER_GOAL.failed(summaries_l[ep]["outcome"]), "learner_failed_part_or_recorded_floor", request["request_id"])
            learning_feedback_checked += 1

    result = {"scope": "CPU saved-record outcome, public-view and attribution audit of scored stream 4301 only",
              "python_version": sys.version,
              "pinned_source": SOURCE, "wheel_sha256": WHEEL, "positions": positions, "checks": dict(checks),
              "problems": problems, "action_steps_checked": steps_checked, "decisions_checked": decisions_checked,
              "public_contexts_reconstructed": contexts_checked, "flight_model_requests_checked": flight_requests_checked,
              "learning_examples_checked": learning_examples_checked, "learning_feedback_checked": learning_feedback_checked,
              "labels_by_arm_phase": dict(labels), "procedure_decision_counts_by_source": dict(source_use),
              "procedure_decision_counts_by_phase": dict(source_use_by_phase),
              "limits": ["Boolean/report-rule labels are checked on recorded world measurements; trajectories and true ranges are not recomputed.",
                         "Attribution concerns the heading decision, not every accepted output or continuous control.",
                         "Hash-protected logs do not prove writer correctness; source, native admission, cost and trial bindings are separate review layers.",
                         "No unflown stream scenario, generator, world, candidate, model or test was executed."]}
    out = base / "root-checks/outcome-public-view.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("positions", "checks", "procedure_decision_counts_by_source", "procedure_decision_counts_by_phase")}, indent=2))
    return bool(problems)


if __name__ == "__main__":
    raise SystemExit(main())
