"""The tanker's trusted simulator and component-trial rules, separate from its public contract."""
from __future__ import annotations

from ....runtime.environment import Environment
from ..public import TANKER
from . import episode as E


def create_episode(scenario):
    return E.Episode(scenario)


def component_trial(decision: str | None, sha: str | None, outcome: dict, decisions: list[dict]) -> dict:
    """Decision D27, item 1: whether one shadow flight admits a candidate for its own decision. Unknown or
    incomplete evidence never passes, and the whole outcome is kept beside the verdict by the caller. Only evidence
    recorded under the goal being judged counts, and a decision record without a goal tag counts for nothing
    (Codex's TRIAL147).

    A heading candidate passes if the flight ended naturally with a settled outcome, joined the goal's tanker in time,
    asked no help, and the candidate's own heading, tagged with that goal, was accepted by the world at least once. A
    report candidate passes if the outcome is settled, exactly one report was made under the current goal and every
    reading of it is correct, no help was asked, and that report came from the candidate under that goal: directly
    in the second it was made, or through a trigger the candidate set and nothing later cleared (a null setting, a
    newer setting by another method, or a direct report) before it fired."""
    refused = {r["t"] for r in outcome.get("refusals") or []}
    gv = (outcome.get("goal") or {}).get("version")
    tagged = [p for p in decisions if p.get("goal_version") is not None and p.get("goal_version") == gv]
    used = [p for p in tagged if p.get("decision") == decision and p.get("method") == sha and p.get("part")
            and p["t"] not in refused]
    base = {"decision": decision, "candidate": sha, "goal_version": gv, "used_seconds": 0, "passed": False,
            "reason": None}
    if outcome.get("end_reason") is None:
        return {**base, "reason": "episode did not end naturally"}
    if outcome.get("settled") is not True:
        return {**base, "reason": "outcome not settled"}
    if outcome.get("help_requests"):
        return {**base, "reason": "help was asked"}
    if decision == "heading":
        used = [p for p in used if "heading_deg" in p["part"]]
        base["used_seconds"] = len(used)
        if not used:
            return {**base, "reason": "the candidate's heading was never used under the judged goal"}
        if not (outcome.get("joined") and outcome.get("in_time")):
            return {**base, "reason": "did not join the goal's tanker in time"}
        return {**base, "passed": True}
    if decision == "report":
        reports = [r for r in outcome.get("reports") or [] if r.get("goal_version") == gv]
        if len(reports) != 1:
            return {**base, "reason": f"{len(reports)} reports under the current goal"}
        rep = reports[0]
        if not (outcome.get("report_correct") and outcome.get("report_correct_options") == [True]):
            return {**base, "reason": "the report is not correct under every reading"}
        same_goal = [p for p in tagged if p.get("decision") == "report" and p.get("part") and p["t"] not in refused]
        owner = None
        if rep.get("via_trigger"):
            for p in sorted((p for p in same_goal if p["t"] < rep["t"]), key=lambda p: p["t"]):
                if "report_trigger_nm" in p["part"]:
                    owner = p.get("method") if p["part"]["report_trigger_nm"] is not None else None
                if "report" in p["part"]:
                    owner = None               # a direct report supersedes any trigger
        else:
            direct = [p for p in same_goal if p["t"] == rep["t"] and "report" in p["part"]]
            owner = direct[-1].get("method") if direct else None
        base["used_seconds"] = sum(1 for p in used if {"report", "report_trigger_nm"} & set(p["part"]))
        if owner != sha:
            return {**base, "reason": "the report did not come from the candidate under the judged goal"}
        return {**base, "passed": True}
    return {**base, "reason": f"no component trial for decision {decision!r}"}



TANKER_ENVIRONMENT = Environment(TANKER.pack_id, TANKER.schema_version, create_episode, component_trial,
                                 E.HORIZON + 1, tools=None)
