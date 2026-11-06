"""Labelling one-second heading cases by bounds (note B8, decision D10; Codex's B8.16.2 and B8.18.1).

At second t of an episode, a candidate heading c is held for the one second the interface commits to. Then:
- U_c: the reference pilot flies on from the state after c in the world itself. If it meets the whole goal,
  report included, with every outcome settled, U_c is its join second minus t; otherwise U_c is infinite.
- L_c: from the state after the prefix, the relaxed bound of bounds.py, plus the prefix's one second.
- U_* is the smallest U over the cases tried at t, including the reference pilot's own output, and L_* is the
  relaxed bound from the state at t.
The case is wrong if L_c − U_* > δ, right if U_c − L_* < δ/2, wrong for a deadline if the relaxation
excludes every time up to the deadline after the prefix, and unknown otherwise. An infinite or missing bound
never gives a right label. Values are for a physical oracle using the true state.
"""
from __future__ import annotations

import copy
import math
from typing import Any

import telohelm.taskpacks.tanker.world.episode as E
from .bounds import earliest_contact_bound

DELTA_S = 10.0
HORIZON_S = 1800


def _speed_bound(ep: E.Episode) -> float:
    return ep.receiver.tas + math.hypot(*ep.s.wind)


def _fly_on(ep: E.Episode, pilot: Any, bundle: dict[str, Any], steps: int, origin: int,
            version: int) -> tuple[float | None, str]:
    """Elapsed seconds from second origin to a settled success under goal version `version`.

    Returns (infinity, reason) if the witness fails, and (None, 'goal changed') if the goal changes at any point
    of the continuation: a success under a later goal is no witness for the original one (Codex's R22.5)."""
    for _ in range(steps):
        bundle = ep.step(pilot.act(bundle))
        if ep.goal.version != version:
            return None, "goal changed during the witness"
        if bundle is None:
            break
    out = ep.outcome()
    if out["settled"] and out["goal_met"] and ep.record.joined_at is not None:
        return float(ep.record.joined_at - origin), "met"
    return math.inf, "witness did not meet the goal" if out["settled"] else "witness unsettled"


def _relaxed(ep: E.Episode) -> dict[str, Any]:
    tanker = ep.tankers[ep.goal.tanker].orbit
    until = (ep.goal.deadline if ep.goal.deadline is not None else ep.t + HORIZON_S)
    return earliest_contact_bound((ep.receiver.x, ep.receiver.y), float(ep.t), _speed_bound(ep),
                                  tanker.position, tanker.v, float(until))


def label_heading(ep: E.Episode, pilot: Any, bundle: dict[str, Any], candidates: list[float],
                  delta: float = DELTA_S) -> dict[str, Any]:
    """Label each candidate heading at the episode's current second. ep, pilot and bundle are not changed."""
    t = ep.t
    goal = ep.goal
    base = _relaxed(ep)
    lower_star = (base["lower"] - t) if base["status"] == "bound" else (math.inf if base["status"] == "excluded" else None)

    def branch(heading: float | None) -> dict[str, Any]:
        e, p = copy.deepcopy(ep), copy.deepcopy(pilot)
        action = p.act(copy.deepcopy(bundle))
        if heading is not None:
            action = {**{k: v for k, v in action.items() if k != "heading_deg"}, "heading_deg": heading}
        nxt = e.step(action)
        if nxt is None:                                   # the episode ended inside the prefix
            out = e.outcome()
            if not out["settled"] or e.goal.version != goal.version:
                return {"upper": None, "lower": None, "prefix": "unsettled or changed"}
            if out["goal_met"]:
                return {"upper": float(e.record.joined_at - t), "lower": float(max(0, e.record.joined_at - 1 - t)),
                        "prefix": "met"}
            if out["end_reason"] == "deadline":
                return {"upper": math.inf, "lower": math.inf, "prefix": "deadline"}
            return {"upper": math.inf, "lower": None, "prefix": out["end_reason"]}
        if e.goal.version != goal.version:
            return {"upper": None, "lower": None, "prefix": "goal changed"}
        rel = _relaxed(e)
        lower = (rel["lower"] - t) if rel["status"] == "bound" else (math.inf if rel["status"] == "excluded" else None)
        upper, reason = _fly_on(e, p, nxt, HORIZON_S, origin=t, version=goal.version)
        if upper is None:
            return {"upper": None, "lower": None, "prefix": "open", "witness": reason}
        if reason == "witness unsettled":
            return {"upper": None, "lower": lower, "prefix": "open", "witness": reason}
        return {"upper": upper, "lower": lower, "prefix": "open", "relaxation": rel["status"], "witness": reason}

    reference = branch(None)
    results = {c: branch(c) for c in candidates}
    uppers = [r["upper"] for r in [reference, *results.values()] if r["upper"] is not None]
    upper_star = min(uppers) if uppers else math.inf
    labels = {}
    for c, r in results.items():
        if r["prefix"] == "deadline" or (r["prefix"] == "open" and r.get("relaxation") == "excluded"
                                         and goal.deadline is not None):
            label = "wrong (deadline)"
        elif r["lower"] is not None and math.isfinite(upper_star) and r["lower"] - upper_star > delta:
            label = "wrong"
        elif (r["upper"] is not None and lower_star is not None and math.isfinite(r["upper"])
              and math.isfinite(lower_star) and r["upper"] - lower_star < delta / 2):
            label = "right"
        else:
            label = "unknown"
        labels[c] = {"label": label, **r}
    return {"t": t, "upper_star": upper_star, "lower_star": lower_star, "reference": reference, "labels": labels}
