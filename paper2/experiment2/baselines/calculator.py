"""Paper 1's intercept calculator as an Experiment 2 condition (backlog note B9).

Paper 1's tool (scenarios/simpilot/tools.py, intercept_heading) aims at the tanker's predicted orbit
position by a fixed-point iteration: time to go is the distance to the predicted point divided by the
aircraft's airspeed, repeated 25 times. It assumes still air, a straight path at airspeed, a tanker on its
briefed orbit and a current tanker position. Here it reads only the briefing and the delivered bundles,
and is recomputed every second, so a realistic turn's error is corrected at the next second.

It declines, once and for the rest of the episode, as soon as one of its assumptions visibly fails:
wind above 10⁻⁹ NM/s, a report older than 10 s, a report off the briefed radius by more than 10⁻⁶ NM,
or an iteration that has not settled to within 0.5 s. A declined episode counts as not completed
(plan section 11). Reports use the shared report trigger, set at each new goal. It never asks for help.
Study 1's code is not imported or changed.
"""
from __future__ import annotations

import math
from typing import Any

ITERATIONS = 25
SETTLED_S = 0.5
WIND_LIMIT_NM_S = 1e-9
STALE_AGE_S = 10
RADIUS_ALLOWANCE_NM = 1e-6


class Paper1Calculator:
    def __init__(self, briefing: dict[str, Any]):
        self.tankers = {k["callsign"]: k for k in briefing["tankers"]}
        self.goal = briefing["goal"]
        self.latest: dict[str, dict[str, Any]] = {}
        self.declined: dict[str, Any] | None = None
        self.trigger_for: int | None = None

    def _decline(self, t: int, reason: str) -> dict[str, Any]:
        self.declined = {"t": t, "reason": reason}
        return {}

    def act(self, bundle: dict[str, Any]) -> dict[str, Any]:
        t = bundle["t"]
        if bundle["goal"] is not None:
            self.goal = bundle["goal"]
        for r in bundle["reports"]:
            held = self.latest.get(r["callsign"])
            if held is None or r["measured_at"] > held["measured_at"]:
                self.latest[r["callsign"]] = r
        if self.declined:
            return {}
        own = bundle["own"]
        h, trk = math.radians(own["heading_deg"]), math.radians(own["track_deg"])
        wind = (own["ground_speed_nm_s"] * math.sin(trk) - own["tas_nm_s"] * math.sin(h),
                own["ground_speed_nm_s"] * math.cos(trk) - own["tas_nm_s"] * math.cos(h))
        if math.hypot(*wind) > WIND_LIMIT_NM_S:
            return self._decline(t, "wind")
        tanker = self.tankers[self.goal["tanker"]]
        report = self.latest.get(self.goal["tanker"])
        if report is None or t - report["measured_at"] > STALE_AGE_S:
            return self._decline(t, "stale or missing report")
        cx, cy = tanker["centre"]
        r0, v = tanker["radius"], tanker["speed_nm_s"]
        if abs(math.hypot(report["x"] - cx, report["y"] - cy) - r0) > RADIUS_ALLOWANCE_NM:
            return self._decline(t, "orbit differs from the briefing")
        angle = math.atan2(report["x"] - cx, report["y"] - cy)

        def position(at: float) -> tuple[float, float]:
            a = angle + v * (at - report["measured_at"]) / r0         # clockwise, as briefed
            return cx + r0 * math.sin(a), cy + r0 * math.cos(a)

        speed = own["tas_nm_s"]
        px, py = position(t)
        tau = math.hypot(px - own["x"], py - own["y"]) / speed
        previous = math.inf
        for _ in range(ITERATIONS):
            px, py = position(t + tau)
            previous, tau = tau, math.hypot(px - own["x"], py - own["y"]) / speed
        if abs(tau - previous) > SETTLED_S:
            return self._decline(t, "intercept did not settle")
        action: dict[str, Any] = {"goal_version": self.goal["version"],
                                  "heading_deg": math.degrees(math.atan2(px - own["x"], py - own["y"])) % 360}
        if self.trigger_for != self.goal["version"]:
            action["report_trigger_nm"] = self.goal["report_at_nm"]
            self.trigger_for = self.goal["version"]
        return action


def condition_outcome(world_outcome: dict[str, Any], calculator: Paper1Calculator) -> dict[str, Any]:
    """The calculator condition's result kept apart from what happened physically. A declined episode is not
    completed by the condition, even if the held heading later joins and the trigger reports (B9.20.1)."""
    return {"declined": calculator.declined,
            "completed_by_condition": calculator.declined is None and bool(world_outcome["goal_met"]),
            "physical_goal_met": bool(world_outcome["goal_met"]), "settled": world_outcome.get("settled")}
