"""The tanker join task's public contract (slice 1 of the task pack; study2/packs/__init__.py).

Every value here was moved unchanged from where Telohelm read it before (learning.py, local_model.py,
runtime/procedures.py, runtime/delivery.py, hidden_set/store.py), so what Telohelm is told is byte for byte the same
(decision D42; study2/tests/test_designer_freeze.py). The tolerances are the shared contract's own numbers; a test
checks that they equal the world's (study2/tests/test_pack_public_contract.py). This module imports nothing from the
world, the monitor, the reference pilot or the prior.
"""
from __future__ import annotations

from typing import Mapping

from ...contracts import PublicContract

JOIN_TOLERANCE_NM = 0.25
REPORT_TOLERANCE_NM = 0.25
CONTROLLER = "Wedgetail"
OWN_CALLSIGN = "aircraft 2"

ACTIONS = (
    '- decision "heading": {"heading_deg": <number, at least 0 and below 360>}.\n'
    '- decision "report": {"report": "<text, at most 200 characters>"} to make the report now, or '
    '{"report_trigger_nm": <number above 0 and at most 1000, or null>} to set a trigger, or {} to do nothing. A set '
    "trigger sends the report once, in the correct wording, when the observed range is falling and at or below that "
    "distance; null clears it. Each accepted non-null trigger setting re-arms it, even at the same distance; it cannot "
    "fire in its arming second, and omitting this field leaves it armed until it fires, is cleared, or the goal version "
    "changes.\n")

GOAL = (
    f"The goal is met when the receiver joins goal_tanker (comes within {JOIN_TOLERANCE_NM} NM of it) by the deadline, "
    "if there is one, and exactly one report has been made under the current goal_version, and that report is "
    "correct. A report is correct when, at the moment it is made, the receiver is closing on goal_tanker, the true "
    f"range to it is within {REPORT_TOLERANCE_NM} NM of report_at_nm, and the text names {CONTROLLER}, {OWN_CALLSIGN} and "
    f"goal_tanker and contains a number within 0.01 of report_at_nm, for example \"{CONTROLLER}, {OWN_CALLSIGN}, 20 NM from "
    "SHELL 1\". The true range is not in the context; observed_range_nm is the range from the receiver now to the "
    "tanker's position in its newest radar report.\n")


def tools(limits: Mapping) -> str:
    """The prediction tool and the time a procedure call has, with the limits the core and the tool's own
    implementation set."""
    return (
        "A procedure may call predict(args), which flies the receiver forward on the estimates it is given and returns "
        "where it would be and how close it would come to the tanker. It knows only its arguments: it predicts from the orbit and "
        "wind it is given, and nothing else. args is a dict with exactly these fields: goal_version (an "
        "integer, the context's current goal_version; another is refused); start: {x_nm, y_nm, heading_deg, tas_nm_s}, "
        "the receiver now, in nautical miles, degrees true and nautical miles a second; headings_deg: 1 to "
        f"{limits['max_prediction_seconds']} commanded headings in degrees, one for each coming second, flown with the same "
        "30-degree-bank turns as the aircraft; wind_nm_s: [x, y] in nautical miles a second; orbit: {cx_nm, cy_nm, "
        "radius_nm, speed_nm_s, angle_rad}, a clockwise circular orbit with the tanker at (cx_nm + radius_nm * "
        "sin(angle_rad), cy_nm + radius_nm * cos(angle_rad)) now. It returns {\"ok\": true, \"result\": "
        "{\"positions_nm\": [[x, y] at the end of each second], \"closest_approach_nm\": d, \"closest_at_s\": t}, "
        "\"remaining\": n} or {\"ok\": false, \"refusal\": <code>}. d is a distance the predicted paths reach at t seconds "
        f"from now; the search for the smallest distance stops within {limits['gap_nm']:g} NM of it or after "
        f"{limits['max_evaluations']:,} evaluations, so d can be above the true smallest distance. An episode allows {limits['prediction_allowance']} predictions in all, and one "
        f"procedure call at most {limits['max_requests_per_call']} requests. A procedure call's own process has "
        f"{limits['cpu_seconds']} seconds of processor time; the predictions' processor time is measured apart. The whole "
        f"call, predictions included, has {limits['wall_seconds']:g} seconds; a prediction is not stopped part way, so a "
        "call that runs over is found when the prediction returns.\n")


TRIAL_RULE = (
    "heading procedure passes only if, in every one, the aircraft joins the goal's tanker in time without help using "
    "its headings; a report procedure passes only if, in every one, it makes the one correct report itself, directly "
    "or by its trigger.")

FLIGHT_SYSTEM = """You fly a receiver aircraft that must join a tanker and make one radio report on the way.
Positions are in nautical miles (x east, y north); headings are degrees true, 0 north, 90 east, clockwise.
The tanker flies a circular orbit: centre (tanker_centre_x, tanker_centre_y), radius tanker_radius_nm, at
tanker_speed_nm_s. Radar reports of the tanker's position may stop; report_measured_at is the time of the newest one,
and t is the time now. wind_x_nm_s and wind_y_nm_s move the receiver. tas_nm_s is the receiver's airspeed.
Answer with one JSON object and nothing else."""

FLIGHT_ASK = {
    "heading": 'Choose the heading that best closes on the tanker where it will be. Reply {"heading_deg": <number>}.',
    # The world's rule (episode.REPORT_TOLERANCE); this said 0.15 until Codex's MVP119.REPORT, 13 October.
    "report": (f"If the receiver is closing on the tanker and the observed range is within {REPORT_TOLERANCE_NM} NM of "
               f'report_at_nm, reply {{"report": "{CONTROLLER}, {OWN_CALLSIGN}, <report_at_nm> NM from <goal_tanker>"}}; '
               "otherwise reply {}. The report is scored on the true range, which observed_range_nm estimates, and "
               "only one report counts."),
}

TANKER = PublicContract(
    pack_id="tanker-join",
    schema_version=1,
    decisions=("heading", "report"),
    decision_fields={"heading": frozenset({"heading_deg", "help"}), "report": frozenset({"report", "report_trigger_nm"})},
    pair_decisions=("heading", "report"),
    context_fields=frozenset({
        "t", "x", "y", "heading_deg", "commanded_heading_deg", "track_deg", "tas_nm_s", "ground_speed_nm_s",
        "wind_x_nm_s", "wind_y_nm_s",
        "goal_version", "goal_tanker", "report_at_nm", "deadline",
        "tanker_centre_x", "tanker_centre_y", "tanker_radius_nm", "tanker_speed_nm_s",
        "report_x", "report_y", "report_measured_at", "report_age_s",
        "observed_range_nm", "observed_range_rate_nm_s", "reports",
    }),
    output_fields=frozenset({"heading_deg", "report", "report_tanker", "help_reason"}),
    learner_purpose="fly a receiver aircraft to join a tanker and make one radio report on the way",
    learner_actions=ACTIONS,
    learner_goal=GOAL,
    learner_tools=tools,
    learner_trial_rule=TRIAL_RULE,
    learner_attribute_example="ctx.x or out.heading_deg",
    learner_decision_choices='"heading" or "report"',
    flight_system=FLIGHT_SYSTEM,
    flight_ask=FLIGHT_ASK,
)
