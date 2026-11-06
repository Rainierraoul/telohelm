"""The tanker's existing example selection, shortening, public radio details and exact learner instruction.

Moved from study2/examples.py without changing the task rule. No reference labels or disturbance knowledge.
"""
from __future__ import annotations

from typing import Any
from ...contracts.example_view import ExamplePolicy; from ...contracts.example_view import METHOD; from ...contracts.example_view import applied_at

HEADING_POINT_S = 120
REPORTS_SHOWN = 10

RULES = ("examples: up to 8 moments from your own completed episodes in this stream, failures first (newest episode "
         "first), then one episode that met the goal. Each names its episode by the same opaque name as feedback, the "
         "second, and the context your procedures saw at that second, except that its radar history is shortened to "
         f"the last {REPORTS_SHOWN} reports (live procedures receive up to 60). submitted is the action sent that "
         "second and which method issued each part; a heading that is not listed was held from earlier. applied says "
         "whether the world accepted it. report_transmitted, when present, says the radio sent a report that second "
         "and, if a trigger sent it, when and by which method the trigger was set. outcome is the episode's result as "
         "fed back. No example says whether its action was right.")


def points(decisions: list[dict], failed: str | None, outcome: dict, goal: Any) -> tuple[list[int], list[str]]:
    """The seconds chosen for one episode, in increasing order, and any omissions."""
    seconds = {p["t"] for p in decisions}
    last = max(seconds)
    chosen, omitted = set(), []
    if failed is None or "heading" in goal.concerns(failed):
        if HEADING_POINT_S in seconds:
            chosen.add(HEADING_POINT_S)
        else:
            omitted.append(f"no decision at {HEADING_POINT_S} s")
        if failed is not None:
            chosen.add(last)
    if "report" in goal.concerns(failed):
        sent = [r["t"] for r in outcome.get("reports") or [] if r.get("t") in seconds]
        chosen.update(sent if sent else [last])
    return sorted(chosen), omitted


def transmission(t: int, by_trigger: bool, actions: dict, sources: dict, steps: dict, goal_changes=(),
                 fired=()) -> dict:
    """A report the radio sent at second t. If a trigger sent it, the trigger that was armed: when it was set, at what
    range and by which method, kept apart from the second it was sent (Codex's TURN 205). The armed trigger is rebuilt
    from the actions the world accepted, in order (Codex's TURN 207): a refused action changes nothing, an accepted
    setting arms or disarms, a direct report disarms, and a goal change or an earlier firing clears it. If a second's
    acceptance is not in the log, or nothing was armed, no trigger is named."""
    sent = {"by_trigger": by_trigger}
    if not by_trigger:
        return sent
    armed, known = None, True
    clears = sorted([c for c in goal_changes if c <= t] + [f for f in fired if f < t])
    for s in sorted(x for x in actions if x < t):
        while clears and clears[0] <= s:
            clears.pop(0)
            armed = None
        accepted = applied_at(s, steps)
        if accepted is None:
            known = False
        if accepted is not True:
            continue
        action = actions[s]["action"]
        if "report_trigger_nm" in action:
            armed = None if action["report_trigger_nm"] is None else s
        if "report" in action:
            armed = None                   # a direct report supersedes any trigger, even one set with it
    if any(c > (armed if armed is not None else t) for c in clears):
        armed = None
    if armed is None or not known:
        sent["trigger_unattributed"] = "the armed trigger cannot be established from the logs"
        return sent
    writer = (actions[armed].get("field_writers") or {}).get("report_trigger_nm")
    sent.update(trigger_set_at=armed, trigger_nm=actions[armed]["action"]["report_trigger_nm"],
                trigger_by=METHOD.get(sources.get((armed, writer)), "no method"))
    return sent


def project(ctx: dict) -> tuple[dict, dict]:
    shown = {k: v for k, v in ctx.items() if k != "reports"}
    shown["reports"] = ctx["reports"][-REPORTS_SHOWN:]
    return shown, {"reports_in_context": len(ctx["reports"]), "reports_shown": len(shown["reports"])}


def details(t: int, outcome: dict, actions: dict, sources: dict, steps: dict, goal_changes=()) -> dict:
    transmissions = {r["t"]: bool(r.get("via_trigger")) for r in outcome.get("reports") or []}
    if t not in transmissions:
        return {}
    fired = [r for r, by in transmissions.items() if by and r < t]
    return {"report_transmitted": transmission(t, transmissions[t], actions, sources, steps, goal_changes, fired)}


TANKER_EXAMPLES = ExamplePolicy(RULES, points, project, details)
