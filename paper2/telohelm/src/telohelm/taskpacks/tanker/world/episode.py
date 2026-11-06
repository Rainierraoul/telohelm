"""One Experiment 2 episode: events, reports, goals, the report trigger, help and scoring.

Follows DISTURBANCES.md draft 0.3, sections 6 to 9 and 12. The world-private
scenario stays inside this module; a condition sees only the briefing and the
per-second bundles. Model and prediction requests belong to the layer above.
"""
from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

from . import physics as P

HORIZON = 3600
JOIN_TOLERANCE = 0.25
REPORT_TOLERANCE = 0.25
STALE_AGE = 10
HELP_DELAY = 30
OWN_CALLSIGN = "aircraft 2"
REPORT_FLAG_MARGIN = 1e-5
INBOUND_FLAG_MARGIN = 1e-9  # NM/s, declared numerical sensitivity band, not a proved derivative error bound.
HELP_REASONS = frozenset({"missing_tool", "missing_or_stale_evidence", "unfamiliar_but_solvable", "rules_changed"})


@dataclass(frozen=True)
class Goal:
    version: int
    tanker: str
    report_at_nm: float
    deadline: int | None = None      # whole simulated seconds, inclusive


@dataclass(frozen=True)
class Tanker:
    callsign: str
    orbit: P.Orbit


@dataclass(frozen=True)
class Scenario:
    """World-private description of one episode."""
    episode_id: str
    receiver: P.ReceiverState
    wind: tuple[float, float]                         # NM/s, the velocity the air moves with
    tankers: tuple[Tanker, ...]
    goals: tuple[tuple[int, Goal], ...]               # (effective second, goal); the first is at 0
    control_callsign: str = "Wedgetail"
    delay_s: int = 0                                  # report delay, all tankers
    dropouts: tuple[tuple[str, int, int | None], ...] = ()   # (callsign, start, end or None for permanent)


@dataclass
class Trigger:
    distance: float
    goal_version: int
    previous_range: float | None = None
    armed_at: int = -1


@dataclass
class EpisodeRecord:
    actions: list[dict[str, Any]] = field(default_factory=list)
    reports: list[dict[str, Any]] = field(default_factory=list)
    help_requests: list[dict[str, Any]] = field(default_factory=list)
    refusals: list[dict[str, Any]] = field(default_factory=list)
    flags: list[dict[str, Any]] = field(default_factory=list)
    joined_at: int | None = None
    joined_tanker: str | None = None
    ended_at: int | None = None
    end_reason: str | None = None


class Episode:
    def __init__(self, scenario: Scenario):
        self.s = scenario
        self._initialise()

    def _initialise(self) -> None:
        scenario = self.s
        self.t = 0
        self.receiver = scenario.receiver
        self.tankers = {k.callsign: k for k in scenario.tankers}
        self.goal = scenario.goals[0][1]
        self.pending_goals = list(scenario.goals[1:])
        self.trigger: Trigger | None = None
        self.pending_help: list[tuple[int, Goal, dict[str, str]]] = []    # (reply time, goal at request, reason)
        self.queue: list[tuple[int, dict[str, Any]]] = []      # (delivery time, report)
        self.record = EpisodeRecord()
        self.done = False
        self._delivered: dict[str, dict[str, Any]] = {}
        self._join_alternatives: list[dict[str, Any]] = []

    # -- truth, for the world and for scoring only ------------------------------------------
    def _true_range(self, callsign: str, t: float) -> float:
        r = P.receiver_at(self.receiver, self.s.wind, t) if t >= self.receiver.t else None
        x, y = self.tankers[callsign].orbit.position(t)
        return math.hypot(r.x - x, r.y - y)

    def _lost(self, callsign: str, t: int) -> bool:
        return any(c == callsign and start <= t and (end is None or t < end) for c, start, end in self.s.dropouts)

    # -- the episode ---------------------------------------------------------------------------
    def reset(self) -> dict[str, Any]:
        """Time 0: the briefing, the first goal and one fresh report per tanker (section 6)."""
        self._initialise()
        briefing = {
            "own_callsign": OWN_CALLSIGN, "control_callsign": self.s.control_callsign,
            "tankers": [{"callsign": k.callsign, "centre": [k.orbit.cx, k.orbit.cy], "radius": k.orbit.r0,
                         "speed_nm_s": k.orbit.v, "direction": "clockwise",
                         "position_at_0": list(k.orbit.position(0.0))} for k in self.s.tankers],
            "goal": self._goal_dict(self.goal),
        }
        self._measure_reports(0, include_lost=False, initial=True)
        return {"briefing": briefing, "bundle": self._bundle(goal_changed=True)}

    def step(self, actions: dict[str, Any]) -> dict[str, Any] | None:
        """Apply the actions for second t, advance to t + 1 and return the next bundle, or None at the end."""
        if self.done:
            raise RuntimeError("episode has ended")
        self._apply_actions(actions)
        self._check_trigger()
        t0, t1 = self.t, self.t + 1
        goal_in_interval = self.goal
        # Step 1: advance and check the join over [t0, t1] against the goal in effect during it.
        decision, approach = P.join_assessment(lambda u: self._true_range(goal_in_interval.tanker, u), float(t0), float(t1))
        self.receiver = P.receiver_at(self.receiver, self.s.wind, float(t1))
        self.t = t1
        if decision in ("flagged", "unresolved"):
            kind = "join near tolerance" if decision == "flagged" else "join unresolved at the work cap"
            self.record.flags.append({"t": t1, "kind": kind, "tanker": goal_in_interval.tanker,
                                      "enclosure": asdict(approach)})
            # Save the ending branch before events/goals/actions at t1 can change its score or cost prefix.
            alternative = self._score(goal_in_interval, "join", t1, goal_in_interval.tanker)
            alternative["assumption"] = "earlier flagged joins are no join; this flagged interval joins"
            alternative["join_assessment"] = asdict(approach)
            self._join_alternatives.append(alternative)
        if decision == "join":
            return self._end("join", t1, goal_in_interval.tanker)
        if goal_in_interval.deadline is not None and t1 >= goal_in_interval.deadline:
            return self._end("deadline", t1)
        if t1 >= HORIZON:
            return self._end("horizon", t1)
        # Step 2: scheduled events, here goal changes.
        changed = False
        while self.pending_goals and self.pending_goals[0][0] <= t1:
            _, self.goal = self.pending_goals.pop(0)
            changed = True
            if self.trigger and self.trigger.goal_version != self.goal.version:
                self.trigger = None
        # Step 3: reports measured and delivered.
        self._measure_reports(t1)
        return self._bundle(goal_changed=changed)

    # -- internals -------------------------------------------------------------------------------
    def _goal_dict(self, g: Goal) -> dict[str, Any]:
        return {"version": g.version, "tanker": g.tanker, "report_at_nm": g.report_at_nm, "deadline": g.deadline}

    def _measure_reports(self, t: int, include_lost: bool = False, initial: bool = False) -> None:
        for k in self.s.tankers:
            if initial or not self._lost(k.callsign, t):
                x, y = k.orbit.position(float(t))
                deliver = t if initial else t + self.s.delay_s
                self.queue.append((deliver, {"callsign": k.callsign, "x": x, "y": y, "measured_at": t}))
        for when, goal, _ in list(self.pending_help):
            if when == t:
                x, y = self.tankers[goal.tanker].orbit.position(float(t))
                self.queue.append((t, {"callsign": goal.tanker, "x": x, "y": y, "measured_at": t, "help_reply": True}))
        self.pending_help = [h for h in self.pending_help if h[0] != t]

    def _bundle(self, goal_changed: bool) -> dict[str, Any]:
        due = [r for when, r in self.queue if when <= self.t]
        self.queue = [(when, r) for when, r in self.queue if when > self.t]
        due.sort(key=lambda r: (r["measured_at"], r["callsign"], bool(r.get("help_reply"))))
        for r in due:
            r["received_at"] = self.t
        # The trigger's evidence is the report with the latest measurement time. A later delivery of an older
        # measurement never replaces it. On equal measurement times the report held first is kept, and within one
        # second's deliveries a regular report comes before a help reply (both are exact, so the position is equal).
        self._delivered = getattr(self, "_delivered", {})
        for r in due:
            newest = self._delivered.get(r["callsign"])
            if newest is None or r["measured_at"] > newest["measured_at"]:
                self._delivered[r["callsign"]] = r
        rcv = self.receiver
        ground = (rcv.tas * math.sin(rcv.heading) + self.s.wind[0], rcv.tas * math.cos(rcv.heading) + self.s.wind[1])
        bundle = {
            "t": self.t,
            "own": {"x": rcv.x, "y": rcv.y, "heading_deg": math.degrees(rcv.heading) % 360.0,
                    "commanded_heading_deg": math.degrees(rcv.commanded) % 360.0, "tas_nm_s": rcv.tas,
                    "track_deg": math.degrees(math.atan2(ground[0], ground[1])) % 360.0,
                    "ground_speed_nm_s": math.hypot(*ground)},
            "reports": due,
            "goal": self._goal_dict(self.goal) if goal_changed else None,
            "refusals": [r for r in self.record.refusals if r["t"] == self.t - 1],
        }
        return bundle

    def _check_trigger(self) -> None:
        """Section 8, report trigger: newest delivered report, one shot, tied to its goal version."""
        trig = self.trigger
        if trig is None or trig.goal_version != self.goal.version or self.t <= trig.armed_at:
            return
        rng = self._observed_range()
        if rng is None:
            return
        decreasing = trig.previous_range is not None and rng < trig.previous_range
        trig.previous_range = rng
        if decreasing and rng <= trig.distance:
            self.trigger = None
            self._report(f"{self.s.control_callsign}, {OWN_CALLSIGN}, {self.goal.report_at_nm:g} NM from {self.goal.tanker}",
                         self.goal.version, via_trigger=True)

    def _observed_range(self) -> float | None:
        """Range from the aircraft's own position now to the newest delivered report of the goal's tanker."""
        newest = getattr(self, "_delivered", {}).get(self.goal.tanker)
        if newest is None:
            return None
        return math.hypot(self.receiver.x - newest["x"], self.receiver.y - newest["y"])

    def _action_error(self, actions: Any) -> str | None:
        if not isinstance(actions, dict):
            return "actions must be an object"
        if not actions:
            return None
        allowed = {"goal_version", "heading_deg", "report", "report_trigger_nm", "help"}
        if set(actions) - allowed:
            return "unknown action field"
        version = actions.get("goal_version")
        if type(version) is not int:
            return "goal version must be an integer"
        if version != self.goal.version:
            return "goal version out of date"
        def finite(v: Any) -> bool:
            try:
                return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            except OverflowError:
                return False
        if "heading_deg" in actions:
            h = actions["heading_deg"]
            if not finite(h) or not 0 <= h < 360:
                return "heading out of range"
        if "report" in actions:
            text = actions["report"]
            if not isinstance(text, str) or not text.strip() or len(text) > 200:
                return "report must be a nonempty string of at most 200 characters"
        if "report_trigger_nm" in actions:
            d = actions["report_trigger_nm"]
            if d is not None and (not finite(d) or not 0 < d <= 1000):
                return "trigger distance must be null or finite in (0, 1000]"
        if "help" in actions:
            h = actions["help"]
            if (not isinstance(h, dict) or set(h) != {"reason", "text"}
                    or not isinstance(h["reason"], str) or h["reason"] not in HELP_REASONS
                    or not isinstance(h["text"], str) or len(h["text"]) > 500):
                return "help must name one declared reason and text of at most 500 characters"
            if self.record.help_requests:
                return "at most one help request per episode"
        return None

    def _apply_actions(self, actions: Any) -> None:
        error = self._action_error(actions)
        # The ledger records submitted public data, never a hidden reason for refusal.
        self.record.actions.append({"t": self.t, "actions": deepcopy(actions), "accepted": error is None})
        if error is not None:
            self.record.refusals.append({"t": self.t, "reason": error})
            return
        if "heading_deg" in actions:
            self.receiver = P.ReceiverState(self.receiver.t, self.receiver.x, self.receiver.y, self.receiver.heading,
                                            math.radians(actions["heading_deg"]), self.receiver.tas, self.receiver.omega)
        if "report_trigger_nm" in actions:
            d = actions["report_trigger_nm"]
            self.trigger = None if d is None else Trigger(float(d), self.goal.version, self._observed_range(), self.t)
        if "report" in actions:
            self._report(actions["report"], self.goal.version, via_trigger=False)
            # Direct reporting supersedes any trigger for this goal, even if set in the same action.
            self.trigger = None
        if "help" in actions:
            h = deepcopy(actions["help"])
            self.record.help_requests.append({"t": self.t, "goal_version": self.goal.version, **h})
            self.pending_help.append((self.t + HELP_DELAY, self.goal, h))

    def _report(self, text: str, version: int, via_trigger: bool) -> None:
        g = self.goal
        t = float(self.t)
        rng = self._true_range(g.tanker, t)
        tx, ty = self.tankers[g.tanker].orbit.position(t)
        tvx, tvy = self.tankers[g.tanker].orbit.velocity(t)
        rvx = self.receiver.tas * math.sin(self.receiver.heading) + self.s.wind[0]
        rvy = self.receiver.tas * math.cos(self.receiver.heading) + self.s.wind[1]
        derivative = ((self.receiver.x - tx) * (rvx - tvx) + (self.receiver.y - ty) * (rvy - tvy)) / rng if rng else 0.0
        inbound = derivative < 0.0
        inbound_flag = abs(derivative) <= INBOUND_FLAG_MARGIN
        range_flag = abs(abs(rng - g.report_at_nm) - REPORT_TOLERANCE) <= REPORT_FLAG_MARGIN + P.NUMERIC_ALLOWANCE
        index = len(self.record.reports)
        for condition, kind in ((inbound_flag, "report inbound near zero"), (range_flag, "report range near tolerance")):
            if condition:
                self.record.flags.append({"t": self.t, "kind": kind, "goal_version": version,
                                          "report_index": index, "true_range": rng, "range_derivative_nm_s": derivative})
        lower = text.lower()
        numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", lower)]
        content_ok = (self.s.control_callsign.lower() in lower and OWN_CALLSIGN in lower and g.tanker.lower() in lower
                      and any(abs(n - g.report_at_nm) <= 0.01 for n in numbers))
        range_ok = abs(rng - g.report_at_nm) <= REPORT_TOLERANCE
        inbound_options = (False, True) if inbound_flag else (inbound,)
        range_options = (False, True) if range_flag else (range_ok,)
        correct_options = sorted({content_ok and i and d for i in inbound_options for d in range_options})
        self.record.reports.append({
            "t": self.t, "goal_version": version, "text": text, "via_trigger": via_trigger,
            "true_range": rng, "range_derivative_nm_s": derivative, "inbound": inbound, "content_ok": content_ok,
            "correct_options": correct_options,
            "correct": inbound and abs(rng - g.report_at_nm) <= REPORT_TOLERANCE and content_ok,
        })

    def _end(self, reason: str, t: int, tanker: str | None = None) -> None:
        self.done = True
        self.record.ended_at = t
        self.record.end_reason = reason
        if reason == "join":
            self.record.joined_at = t
            self.record.joined_tanker = tanker
        return None

    def _score(self, goal: Goal, reason: str | None, ended: int | None,
               joined_tanker: str | None) -> dict[str, Any]:
        mine = [r for r in self.record.reports if r["goal_version"] == goal.version]
        joined = reason == "join" and joined_tanker == goal.tanker
        in_time = goal.deadline is None or (joined and ended is not None and ended <= goal.deadline)
        one_correct = len(mine) == 1 and mine[0]["correct"]
        report_options = mine[0].get("correct_options", [mine[0]["correct"]]) if len(mine) == 1 else [False]
        return deepcopy({"goal": self._goal_dict(goal), "goal_met": bool(joined and in_time and one_correct),
                         "goal_met_options": sorted({bool(joined and in_time and c) for c in report_options}),
                         "joined": joined, "in_time": in_time, "reports_under_goal": len(mine),
                         "report_correct": bool(one_correct), "report_correct_options": report_options,
                         "end_reason": reason, "ended_at": ended, "event_cutoff_t": ended,
                         "action_steps": len(self.record.actions), "reports": self.record.reports,
                         "flags": self.record.flags, "help_requests": self.record.help_requests,
                         "refusals": self.record.refusals})

    def outcome(self) -> dict[str, Any]:
        """Nominal continuation plus all possible ending/ambiguous-report score options.

        A flag is never a settled result. Alternative ending records preserve their
        then-current goal and event prefix; later reports/goals cannot revise them.
        The upper runtime must truncate its call/cost ledger at event_cutoff_t too.
        """
        primary = self._score(self.goal, self.record.end_reason, self.record.ended_at, self.record.joined_tanker)
        alternatives = deepcopy(self._join_alternatives)
        return primary | {"settled": self.done and not self.record.flags, "alternatives": alternatives,
                          "possible_goal_met": sorted(set(primary["goal_met_options"]).union(
                              *(set(a["goal_met_options"]) for a in alternatives))),
                          "actions": deepcopy(self.record.actions)}
