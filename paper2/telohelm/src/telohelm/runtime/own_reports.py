"""Knowing when Telohelm's own report trigger has fired (opt-in; paper2/notes/HANDOVER-SIGNALS.md).

The world sends a triggered report by itself, and nothing in the next bundle says so. Without this, the runtime kept
asking the report decision after its trigger had fired, and a procedure that set the trigger again, or reported
directly, made a second report under the same goal, which fails the goal (8 of the 57 failures in phases 3 and 4 of
the four step 4 runs; 5 of them after a correct first report).

The trigger's rule is public; Telohelm is told it in the learner's rules and the flight models' instructions: a set
trigger sends the report once, when the observed range is falling and at or below its distance; it cannot fire in its
arming second; a null setting or a direct report clears it; a new goal version drops it. The observed range is the
context's observed_range_nm, the same number the world uses. So this replica, told each second's context and each
action with the world's verdict, knows exactly when the trigger fires, in the world's order: the action first, then the
trigger check, both at the action's second.
"""
from __future__ import annotations

from typing import Any


class OwnTrigger:
    def __init__(self) -> None:
        self.armed: dict[str, Any] | None = None       # {"distance", "previous", "armed_at", "goal_version", "by"}
        self.sent: dict[int, dict[str, Any]] = {}       # goal version -> the first report made under it
        self.events: list[dict[str, Any]] = []

    def at(self, goal_version: int) -> None:
        """A new goal version drops a trigger armed under another (the world's step 2)."""
        if self.armed is not None and self.armed["goal_version"] != goal_version:
            self.events.append({"kind": "trigger_dropped", "goal_version": goal_version, **self.armed})
            self.armed = None

    def reported(self, goal_version: int) -> bool:
        return goal_version in self.sent

    def after(self, t: int, ctx: dict[str, Any], action: dict[str, Any], accepted: bool, by: str | None) -> None:
        """Second t's action and the world's verdict on it, then the world's trigger check at t. ctx is second t's
        context; by is the source of the part that set a trigger (procedure, small_model or reasoning_model)."""
        version = ctx["goal_version"]
        if accepted and action:
            if "report_trigger_nm" in action:
                d = action["report_trigger_nm"]
                self.armed = None if d is None else {"distance": float(d), "previous": ctx.get("observed_range_nm"),
                                                    "armed_at": t, "goal_version": version, "by": by}
            if "report" in action:
                self._sent(t, version, via_trigger=False)
                self.armed = None
        trig = self.armed
        if trig is None or trig["goal_version"] != version or t <= trig["armed_at"]:
            return
        rng = ctx.get("observed_range_nm")
        if rng is None:
            return
        decreasing = trig["previous"] is not None and rng < trig["previous"]
        trig["previous"] = rng
        if decreasing and rng <= trig["distance"]:
            self.armed = None
            self._sent(t, version, via_trigger=True)

    def _sent(self, t: int, version: int, via_trigger: bool) -> None:
        self.events.append({"kind": "own_report", "t": t, "goal_version": version, "via_trigger": via_trigger})
        self.sent.setdefault(version, {"t": t, "via_trigger": via_trigger})
