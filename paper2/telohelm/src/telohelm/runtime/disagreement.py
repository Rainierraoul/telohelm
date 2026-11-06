"""Noticing in flight that the evidence disagrees with the briefing, and handing over (opt-in; Codex's rulings in
paper2/notes/TURN229-REVIEW-CODEX.md; the analysis in paper2/notes/HANDOVER-SIGNALS.md).

Each second the watch measures the task pack's gap between the newest observation in the context and what the briefing
alone predicts for it (packs/evidence.py). When the gap goes above the pack's tolerance an incident opens, and while it
is open the decisions whose procedures read that evidence are withheld from their learned procedures before selection:
the runtime does not run them and climbs to a model instead, small model first, under the ladder's existing rationing,
latch and one-call reasoning limit. The model is given the gap and what it means. Withholding is kept apart from the
ladder's latch: an accepted model reply releases the latch as before, and nothing here clears the incident because an
output was well formed.

An incident closes when the gap has stayed within the tolerance for COOLDOWN_S, a second with no gap to measure
starting that count again, or when the goal version changes; the new goal's evidence then decides afresh. For a
decision that no model can be asked any more (the ladder would send it to the reasoning model and the reasoning model
has been refused it under this goal version), withholding is lifted for the rest of that incident and the lift
recorded, so a lasting gap never leaves a decision with no method at all; a later incident starts with no lifts and
judges exhaustion again (Codex's TURN231 item 3). Library entries are never retired or changed by this.

A standing order a withheld procedure left armed (for the tanker, the report trigger) is withdrawn: the withdrawal
stays pending, and is tried again each second, until the world accepts an action carrying it or the order is
superseded (it fired, another method replaced it, or the goal changed). Attempts and the world's verdicts are recorded
apart, and the model is told only the state the world has confirmed (TURN231 item 1).

Everything the watch did is in its events, which the episode loop writes to the trusted episode log, and its summary
goes into the episode's record and, through the coordinator, into the learner's feedback.
"""
from __future__ import annotations

from typing import Any, Mapping

COOLDOWN_S = 60


def models_exhausted(ladder: Any, decision: str) -> bool:
    """No model will be asked for this decision again under the current goal version: the ladder would send it to the
    reasoning model, and a reasoning call for it has already been refused."""
    return (ladder.role_for(decision, "") == "reasoning_model"
            and (decision, ladder.goal_version) in ladder.no_reasoning)


class Watch:
    def __init__(self, view: Any, briefing: Mapping):
        self.view, self.briefing = view, briefing
        self.open: dict[str, Any] | None = None
        self.incidents: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self._agreeing_since: int | None = None

    def update(self, ctx: Mapping) -> list[dict[str, Any]]:
        """Second ctx["t"]: measure the gap and open, update or close the incident. Returns this second's events."""
        start = len(self.events)
        t, version = ctx["t"], ctx["goal_version"]
        if self.open is not None and self.open["goal_version"] != version:
            self._close(t, "goal changed")
        gap = self.view.gap(self.briefing, ctx)
        if gap is not None and gap > self.view.tolerance:
            self._agreeing_since = None
            if self.open is None:
                self.open = {"incident": len(self.incidents) + 1, "goal_version": version, "opened_at": t,
                             "gap": gap, "largest": gap, "closed_at": None, "why_closed": None, "lifted": [],
                             "withdrawals": {}}
                self.incidents.append(self.open)
                self.events.append({"kind": "incident_opened", "t": t, "incident": self.open["incident"],
                                    "goal_version": version, "gap": gap, "tolerance": self.view.tolerance,
                                    "measured_at": ctx.get("report_measured_at")})
            else:
                self.open["gap"], self.open["largest"] = gap, max(self.open["largest"], gap)
        elif self.open is not None and gap is None:
            self._agreeing_since = None           # nothing to measure this second: agreement must start again
        elif self.open is not None:
            self.open["gap"] = gap
            self._agreeing_since = t if self._agreeing_since is None else self._agreeing_since
            if t - self._agreeing_since >= COOLDOWN_S:
                self._close(t, "agreed")
        return self.events[start:]

    def withholds(self, decision: str, goal_version: int) -> bool:
        o = self.open
        return (o is not None and o["goal_version"] == goal_version and decision in self.view.affects
                and all(lift["decision"] != decision for lift in o["lifted"]))

    def lift(self, decision: str, goal_version: int, t: int, why: str) -> dict[str, Any]:
        self.open["lifted"].append({"decision": decision, "t": t, "why": why})
        return self._event({"kind": "withholding_lifted", "t": t, "incident": self.open["incident"],
                            "decision": decision, "goal_version": goal_version, "why": why})

    # -- a standing order a withheld procedure left armed -------------------------------------------------------------
    def withdrawal(self, decision: str) -> dict[str, Any] | None:
        return None if self.open is None else self.open["withdrawals"].get(decision)

    def withdrawal_needed(self, decision: str, armed: Mapping, t: int) -> dict[str, Any]:
        self.open["withdrawals"][decision] = {"state": "pending", "armed": dict(armed), "since": t, "attempts": [],
                                              "accepted_at": None}
        return self._event({"kind": "standing_order_withdrawal_pending", "t": t, "incident": self.open["incident"],
                            "decision": decision, "armed": dict(armed)})

    def withdrawal_superseded(self, decision: str, t: int, why: str) -> dict[str, Any]:
        self.open["withdrawals"][decision]["state"] = "superseded"
        return self._event({"kind": "standing_order_withdrawal_superseded", "t": t,
                            "incident": self.open["incident"], "decision": decision, "why": why})

    def withdrawal_attempted(self, decision: str, part: Mapping, t: int) -> dict[str, Any]:
        self.open["withdrawals"][decision]["attempts"].append(t)
        return self._event({"kind": "standing_order_withdrawal_attempted", "t": t,
                            "incident": self.open["incident"], "decision": decision, "part": dict(part)})

    def withdrawal_verdict(self, decision: str, t: int, accepted: bool) -> dict[str, Any]:
        w = self.open["withdrawals"][decision]
        if accepted:
            w["state"], w["accepted_at"] = "withdrawn", t
        return self._event({"kind": "standing_order_withdrawn" if accepted else "standing_order_withdrawal_refused",
                            "t": t, "incident": self.open["incident"], "decision": decision})

    def evidence(self) -> dict[str, Any]:
        """What the flight model is told while its decision is withheld from the procedures: the gap, and for each
        standing order a withheld procedure left armed, only what the world has confirmed."""
        o, unit = self.open, self.view.unit.lower()
        orders = {d: ({"state": "withdrawn", "at": w["accepted_at"]} if w["state"] == "withdrawn"
                      else {"state": "armed, withdrawal pending"} if w["state"] == "pending"
                      else {"state": w["state"]})
                  for d, w in o["withdrawals"].items()}
        return {f"gap_{unit}": round(o["gap"], 3), f"largest_{unit}": round(o["largest"], 3),
                f"tolerance_{unit}": self.view.tolerance, "since_t": o["opened_at"], "standing_orders": orders,
                "meaning": self.view.meaning}

    def summary(self) -> dict[str, Any]:
        return {"tolerance": self.view.tolerance, "unit": self.view.unit,
                "incidents": [{**i, "lifted": list(i["lifted"]),
                               "withdrawals": {d: {**w, "attempts": list(w["attempts"])}
                                               for d, w in i["withdrawals"].items()}}
                              for i in self.incidents]}

    def _event(self, event: dict[str, Any]) -> dict[str, Any]:
        self.events.append(event)
        return event

    def _close(self, t: int, why: str) -> None:
        for decision, w in self.open["withdrawals"].items():
            if w["state"] == "pending":
                w["state"] = "unfinished"
                self._event({"kind": "standing_order_withdrawal_unfinished", "t": t, "incident": self.open["incident"],
                             "decision": decision, "why": why})
        self.open["closed_at"], self.open["why_closed"] = t, why
        self._event({"kind": "incident_closed", "t": t, "incident": self.open["incident"], "why": why})
        self.open, self._agreeing_since = None, None
