"""The tanker join task's execution view (slice 2 of the task pack).

Slice 2(a), wired in on 23 October 2026, holds the goal's parts: the receiver must join the goal tanker in time,
which the heading decides, and make the correct report, which the report decision decides. The fields, labels and
decisions were moved unchanged from learning.py, runtime/episode_loop.py, runtime/broker.py and examples.py;
study2/tests/test_pack_goal_parts.py checks that the rules they give equal the inline ones they replaced. The
evidence view below serves the opt-in briefing check (runtime/disagreement.py). This module imports nothing from
the world, the monitor, the reference pilot or the prior.
"""
from __future__ import annotations

import math
from typing import Mapping

from ...contracts.decisions import DecisionRoles; from ...contracts.decisions import HelpEvidence
from ...contracts.evidence import EvidenceView
from ...contracts.goal_parts import GoalPart; from ...contracts.goal_parts import GoalParts
from .public import REPORT_TOLERANCE_NM
from .dynamics import NUMERIC_ALLOWANCE

TANKER_GOAL = GoalParts(parts=(GoalPart("joined", "join", {"heading"}),
                               GoalPart("report_correct", "report", {"report"})),
                        all_failed="both")


# -- Evidence that disagrees with the briefing (opt-in; paper2/notes/HANDOVER-SIGNALS.md, TURN229-REVIEW-CODEX.md) -----

GAP_TOLERANCE_NM = 1.0     # chosen after looking at the four step 4 runs on stream 4401: a declared assumption (D42)


def briefed_position(tanker: Mapping, t: float) -> tuple[float, float]:
    """Where the briefing's orbit puts a tanker at time t: its bearing from the centre, measured clockwise from north,
    moves at the briefed ground speed over the briefed radius, from the briefed position at time 0."""
    (cx, cy), r = tanker["centre"], tanker["radius"]
    px, py = tanker["position_at_0"]
    turn = 1.0 if tanker["direction"] == "clockwise" else -1.0
    bearing = math.atan2(px - cx, py - cy) + turn * tanker["speed_nm_s"] / r * t
    return cx + r * math.sin(bearing), cy + r * math.cos(bearing)


def briefing_gap(briefing: Mapping, ctx: Mapping) -> float | None:
    """How far the goal tanker's newest radar report is from where the briefing's orbit puts it at the report's
    measurement time, in NM; None before any report."""
    if ctx.get("report_x") is None or ctx.get("report_measured_at") is None:
        return None
    tanker = next(k for k in briefing["tankers"] if k["callsign"] == ctx["goal_tanker"])
    x, y = briefed_position(tanker, float(ctx["report_measured_at"]))
    return math.hypot(ctx["report_x"] - x, ctx["report_y"] - y)


TANKER_EVIDENCE = EvidenceView(
    gap=briefing_gap, tolerance=GAP_TOLERANCE_NM, unit="NM", affects={"heading", "report"},
    withdraw={"report": {"report_trigger_nm": None}},
    meaning=("The goal tanker's newest radar report is gap_nm from where the briefing's orbit puts the tanker at the "
             "report's measurement time. While the gap is above tolerance_nm the stored procedures are not used. "
             "standing_orders says whether a report trigger they left armed is still armed or has been withdrawn, as "
             "the world has confirmed; decide from the reports."))


# -- How the decisions behave each second (slice 2(b)) ------------------------------------------------------------------

TANKER_ROLES = DecisionRoles(
    order=("heading", "report"), flying="heading", once_per_goal={"report"},
    completion_fields={"report": {"report"}},
    needs={"heading": ("report_x", "report_y"), "report": ("report_x", "report_y", "observed_range_nm"), "help": ()})

# When help may be asked for stale evidence, and the public rules for each action field's value (slice 2(b)). Moved
# unchanged from runtime/escalation.py, which now applies them for any task.
TANKER_HELP = HelpEvidence(age_field="report_age_s", stale_after_s=10.0,
                           text="the goal tanker's newest report is missing or over 10 s old")


def tanker_part_error(part: Mapping) -> str | None:
    """The public action rules for the tanker's fields, as stated to the models and enforced by the world: a heading
    in [0, 360), a report of 1 to 200 characters, a trigger null or in (0, 1000]. None if they hold."""
    def finite(v) -> bool:
        try:
            return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        except OverflowError:
            return False
    if "heading_deg" in part and not (finite(part["heading_deg"]) and 0 <= part["heading_deg"] < 360):
        return "heading out of range"
    if "report" in part and not (isinstance(part["report"], str) and part["report"].strip()
                                 and len(part["report"]) <= 200):
        return "report must be a nonempty string of at most 200 characters"
    if "report_trigger_nm" in part:
        d = part["report_trigger_nm"]
        if d is not None and not (finite(d) and 0 < d <= 1000):
            return "trigger distance must be null or finite in (0, 1000]"
    return None


def tanker_model_precondition(decision: str, ctx: Mapping, part: Mapping) -> dict | None:
    """Refuse only a direct report contradicted by simultaneous public positions.

    The simulation observes own position at t and a tanker position at report_measured_at. Equal times give range
    at that second. Older observations are deliberately inconclusive: this rule assumes no unobserved motion.
    The existing arithmetic allowance is not a real sensor-error model. Closing and text correctness remain untested.
    """
    if decision != "report" or "report" not in part:
        return None
    fields = ("t", "report_measured_at", "x", "y", "report_x", "report_y", "report_at_nm")
    try:
        if any(isinstance(ctx.get(k), bool) or not isinstance(ctx.get(k), (int, float))
               or not math.isfinite(ctx[k]) for k in fields):
            return None
    except OverflowError:
        return None
    if ctx["report_measured_at"] != ctx["t"]:
        return None
    distance = math.hypot(ctx["report_x"] - ctx["x"], ctx["report_y"] - ctx["y"])
    if not math.isfinite(distance) or abs(distance - ctx["report_at_nm"]) <= REPORT_TOLERANCE_NM + NUMERIC_ALLOWANCE:
        return None
    return {"reason": "report_range_contradicts_current_observation",
            "evidence": {"range_nm": distance, "requested_nm": ctx["report_at_nm"],
                         "measurement_t": ctx["report_measured_at"], "delivery_t": ctx["t"],
                         "tolerance_nm": REPORT_TOLERANCE_NM, "numeric_allowance_nm": NUMERIC_ALLOWANCE}}
