"""Tests for the Experiment 2 episode runner on made-up scenarios. Nothing here is a result."""
from __future__ import annotations

import math
import pytest

import telohelm.taskpacks.tanker.world.episode as E
import telohelm.taskpacks.tanker.world.physics as P


def scenario(**kw):
    rcv = P.ReceiverState(0.0, 0.0, 0.0, math.radians(90), math.radians(90), P.kt(450.0), P.turn_rate(450.0))
    tankers = (E.Tanker("SHELL 1", P.Orbit(40.0, 10.0, 8.0, P.kt(280.0), math.radians(200))),
               E.Tanker("SHELL 2", P.Orbit(40.0, 60.0, 8.0, P.kt(280.0), 0.0)))
    base = dict(episode_id="test", receiver=rcv, wind=(0.0, 0.0), tankers=tankers,
                goals=((0, E.Goal(1, "SHELL 1", 20.0)),))
    base.update(kw)
    return E.Scenario(**base)


def pursue(bundle, latest, goal, extra=None):
    """A test-only pilot: head for the latest delivered report of the named tanker."""
    for r in bundle["reports"]:
        latest[r["callsign"]] = r
    target = latest[goal["tanker"]]
    own = bundle["own"]
    heading = math.degrees(math.atan2(target["x"] - own["x"], target["y"] - own["y"])) % 360
    action = {"goal_version": goal["version"], "heading_deg": heading}
    action.update(extra or {})
    return action


def run(sc, first_extra=None, max_steps=3600):
    ep = E.Episode(sc)
    start = ep.reset()
    goal, latest = start["briefing"]["goal"], {}
    bundle = start["bundle"]
    bundles = [bundle]
    extra = first_extra
    for _ in range(max_steps):
        if bundle["goal"]:
            goal = bundle["goal"]
            extra = {"report_trigger_nm": goal["report_at_nm"]}
        bundle = ep.step(pursue(bundle, latest, goal, extra))
        extra = None
        if bundle is None:
            break
        bundles.append(bundle)
    return ep, bundles


def test_briefing_has_a_fresh_report_for_each_tanker_even_with_delay():
    ep = E.Episode(scenario(delay_s=30))
    start = ep.reset()
    assert {r["callsign"] for r in start["bundle"]["reports"]} == {"SHELL 1", "SHELL 2"}
    assert all(r["measured_at"] == 0 and r["received_at"] == 0 for r in start["bundle"]["reports"])
    assert "position_at_0" in start["briefing"]["tankers"][0]


def test_delay_and_dropout_at_measurement():
    ep = E.Episode(scenario(delay_s=5, dropouts=(("SHELL 1", 10, 15),)))
    ep.reset()
    seen = []
    for _ in range(25):
        b = ep.step({})
        seen += [(b["t"], r["measured_at"]) for r in b["reports"] if r["callsign"] == "SHELL 1"]
    assert all(t - m == 5 for t, m in seen)
    measured = {m for _, m in seen}
    assert not measured & set(range(10, 15))
    assert {9, 15} <= measured


def test_permanent_dropout():
    ep = E.Episode(scenario(dropouts=(("SHELL 1", 30, None),)))
    ep.reset()
    last = 0
    for _ in range(200):
        b = ep.step({})
        last = max([last] + [r["measured_at"] for r in b["reports"] if r["callsign"] == "SHELL 1"])
    assert last == 29


def test_pursuit_joins_and_trigger_reports_once_inbound():
    ep, _ = run(scenario(), first_extra={"report_trigger_nm": 20.0})
    out = ep.outcome()
    assert out["joined"] and out["end_reason"] == "join"
    reports = ep.record.reports
    assert len(reports) == 1 and reports[0]["via_trigger"] and reports[0]["inbound"]
    assert abs(reports[0]["true_range"] - 20.0) <= 0.25
    assert out["goal_met"]


def test_goal_change_disarms_trigger_and_scores_the_new_goal():
    sc = scenario(goals=((0, E.Goal(1, "SHELL 1", 20.0)), (60, E.Goal(2, "SHELL 1", 15.0))))
    ep, _ = run(sc, first_extra={"report_trigger_nm": 20.0})
    out = ep.outcome()
    assert [r["goal_version"] for r in ep.record.reports] == [2]
    assert out["goal_met"]


def test_deadline_ends_episode_unmet():
    sc = scenario(goals=((0, E.Goal(1, "SHELL 1", 20.0, deadline=100)),))
    ep, _ = run(sc, first_extra={"report_trigger_nm": 20.0})
    out = ep.outcome()
    assert out["end_reason"] == "deadline" and out["ended_at"] == 100 and not out["goal_met"]


def test_help_reply_is_fresh_and_arrives_after_30_seconds():
    ep = E.Episode(scenario(dropouts=(("SHELL 1", 5, None),)))
    ep.reset()
    for _ in range(10):
        ep.step({})
    ep.step({"goal_version": 1, "help": {"reason": "missing_or_stale_evidence", "text": "missing or stale evidence"}})
    replies = []
    for _ in range(40):
        b = ep.step({})
        replies += [r for r in b["reports"] if r.get("help_reply")]
    assert len(replies) == 1
    assert replies[0]["measured_at"] == replies[0]["received_at"] == 10 + E.HELP_DELAY
    ep.step({"goal_version": 1, "help": {"reason": "missing_tool", "text": "again"}})
    assert ep.record.refusals[-1]["reason"] == "at most one help request per episode"


def test_old_goal_version_and_bad_heading_are_refused_and_reported():
    sc = scenario(goals=((0, E.Goal(1, "SHELL 1", 20.0)), (3, E.Goal(2, "SHELL 2", 20.0))))
    ep = E.Episode(sc)
    ep.reset()
    for _ in range(3):
        ep.step({})
    b = ep.step({"goal_version": 1, "heading_deg": 10.0})
    assert b["refusals"] and b["refusals"][0]["reason"] == "goal version out of date"
    b = ep.step({"goal_version": 2, "heading_deg": 400.0})
    assert b["refusals"][0]["reason"] == "heading out of range"


def test_second_report_under_same_goal_fails_the_goal():
    ep = E.Episode(scenario())
    ep.reset()
    ep.step({"goal_version": 1, "report": "Wedgetail, aircraft 2, 20 NM from SHELL 1"})
    ep.step({"goal_version": 1, "report": "Wedgetail, aircraft 2, 20 NM from SHELL 1"})
    assert ep.outcome()["reports_under_goal"] == 2 and not ep.outcome()["report_correct"]


# --- B4 boundaries, after Codex's review of 07:59 on 10 October (C5) ----------------------------

def fly(sc, extra_at=None, max_steps=3600):
    """Pursue the named tanker with no trigger, adding extra actions at chosen seconds. Returns the episode
    and the true range to the goal's tanker at each second before the step from it."""
    extra_at = extra_at or {}
    ep = E.Episode(sc)
    start = ep.reset()
    goal, latest, ranges = start["briefing"]["goal"], {}, {}
    bundle = start["bundle"]
    for _ in range(max_steps):
        goal = bundle["goal"] or goal
        ranges[ep.t] = ep._true_range(goal["tanker"], float(ep.t))
        bundle = ep.step(pursue(bundle, latest, goal, extra_at.get(ep.t)))
        if bundle is None:
            break
    return ep, ranges


def join_second():
    ep, ranges = fly(scenario())
    assert ep.record.end_reason == "join"
    return ep.record.joined_at, ranges


def test_join_at_the_deadline_counts_and_one_second_later_does_not():
    J, _ = join_second()
    at = fly(scenario(goals=((0, E.Goal(1, "SHELL 1", 20.0, deadline=J)),)))[0]
    assert at.record.end_reason == "join" and at.outcome()["in_time"]
    early = fly(scenario(goals=((0, E.Goal(1, "SHELL 1", 20.0, deadline=J - 1)),)))[0]
    assert early.record.end_reason == "deadline" and early.record.ended_at == J - 1 and not early.outcome()["joined"]


def test_report_in_the_second_before_the_join_counts():
    J, ranges = join_second()
    d = ranges[J - 1]
    report = {"report": f"Wedgetail, aircraft 2, {d:.4f} NM from SHELL 1"}
    ep, _ = fly(scenario(goals=((0, E.Goal(1, "SHELL 1", round(d, 4))),)), {J - 1: report})
    assert ep.record.joined_at == J and [r["t"] for r in ep.record.reports] == [J - 1]
    assert ep.outcome()["goal_met"]


def test_trigger_cannot_fire_in_the_second_that_ends_the_episode():
    J, _ = join_second()
    ep, _ = fly(scenario(), {J - 1: {"report_trigger_nm": 1000.0}})
    assert ep.record.joined_at == J and ep.record.reports == []


def test_trigger_set_at_t_compares_with_the_range_observed_at_t():
    ep = E.Episode(scenario())
    ep.reset()
    ep.step({"goal_version": 1, "report_trigger_nm": 1000.0})        # set at 0; bundle 1 precedes the check
    assert ep.record.reports == []
    ep.step({})                                                    # actions/check at 1
    assert [r["t"] for r in ep.record.reports] == [1] and ep.record.reports[0]["via_trigger"]


def test_trigger_without_an_earlier_range_records_one_before_it_can_fire():
    ep = E.Episode(scenario())
    ep.reset()
    ep.trigger = E.Trigger(1000.0, 1, None, armed_at=0)       # as if no report had been delivered when it was set
    ep.step({})
    assert ep.record.reports == []
    ep.step({})
    assert ep.record.reports == []
    ep.step({})
    assert [r["t"] for r in ep.record.reports] == [2]


def test_goal_change_in_the_second_a_trigger_would_fire_disarms_it():
    sc = scenario(goals=((0, E.Goal(1, "SHELL 1", 20.0)), (1, E.Goal(2, "SHELL 1", 15.0))))
    ep = E.Episode(sc)
    ep.reset()
    ep.step({"goal_version": 1, "report_trigger_nm": 1000.0})
    assert ep.trigger is None and ep.record.reports == []


def test_an_older_delivery_never_replaces_a_fresher_help_report():
    ep = E.Episode(scenario(delay_s=60))
    ep.reset()
    ep.step({"goal_version": 1, "help": {"reason": "missing_or_stale_evidence", "text": "missing or stale evidence"}})     # asked at 0; fresh reply measured and delivered at 30
    held = []
    for _ in range(80):
        b = ep.step({})
        held.append((b["t"], ep._delivered["SHELL 1"]["measured_at"]))
    by_t = dict(held)
    # Regular reports measured at 1 to 21 arrive from 61 to 81, after the help reply measured at 30.
    assert all(by_t[t] == 30 for t in range(30, 82))


def test_equal_measurement_times_keep_the_regular_report():
    ep = E.Episode(scenario())
    ep.reset()
    ep.step({"goal_version": 1, "help": {"reason": "missing_tool", "text": "check"}})                      # asked at 0; reply measured at 30, as is a regular report
    for _ in range(29):
        ep.step({})
    newest = ep._delivered["SHELL 1"]
    assert newest["measured_at"] == 30 and not newest.get("help_reply")


# TURN 14: actual event order, strict refusal and numerical sensitivity records.

def test_current_actions_replace_or_cancel_a_trigger_before_it_fires():
    for replacement in (0.01, None):
        ep = E.Episode(scenario())
        ep.reset()
        ep.step({"goal_version": 1, "report_trigger_nm": 1000.0})
        assert ep.record.reports == []
        ep.step({"goal_version": 1, "report_trigger_nm": replacement})
        assert ep.record.reports == []
        assert (ep.trigger is None) == (replacement is None)


def test_current_direct_report_supersedes_due_trigger_and_same_action_trigger():
    ep = E.Episode(scenario())
    ep.reset()
    ep.step({"goal_version": 1, "report_trigger_nm": 1000.0})
    # A newly arrived provider reply is applied by its adapter as these current actions.
    ep.step({"goal_version": 1, "report": "Wedgetail, aircraft 2, 20 NM from SHELL 1",
             "report_trigger_nm": 1000.0})
    assert len(ep.record.reports) == 1 and not ep.record.reports[0]["via_trigger"]
    assert ep.record.reports[0]["t"] == 1 and ep.trigger is None
    ep.step({})
    assert len(ep.record.reports) == 1


@pytest.mark.parametrize('bad', [
    {"heading_deg": 10}, {"goal_version": True, "heading_deg": 10},
    {"goal_version": 1, "heading_deg": 30, "report_trigger_nm": float('nan')},
    {"goal_version": 1, "heading_deg": float('inf')},
    {"goal_version": 1, "report_trigger_nm": 'bad'},
    {"goal_version": 1, "report_trigger_nm": True},
    {"goal_version": 1, "report_trigger_nm": 0},
    {"goal_version": 1, "report_trigger_nm": 1001},
    {"goal_version": 1, "heading_deg": 30, "report": ''},
    {"goal_version": 1, "report": 7},
    {"goal_version": 1, "report": 'x' * 201},
    {"goal_version": 1, "help": 'missing evidence'},
    {"goal_version": 1, "help": {"reason": 'other', "text": ''}},
    {"goal_version": 1, "help": {"reason": 'missing_tool', "text": 'x' * 501}},
    {"goal_version": 1, "help": {"reason": [], "text": ''}},
    {"goal_version": 1, "unknown": 'x'}, [], None,
])
def test_invalid_action_refuses_whole_action_without_changing_command(bad):
    ep = E.Episode(scenario())
    ep.reset()
    before = ep.receiver.commanded
    b = ep.step(bad)
    assert ep.receiver.commanded == before
    assert len(b['refusals']) == 1
    assert not ep.record.reports and not ep.record.help_requests and ep.trigger is None
    assert not ep.record.actions[-1]['accepted']


def test_large_integer_heading_is_refused_without_overflow():
    ep = E.Episode(scenario())
    ep.reset()
    assert ep.step({'goal_version': 1, 'heading_deg': 10**1000})['refusals']


def test_second_help_refuses_whole_action_even_if_heading_is_valid():
    ep = E.Episode(scenario())
    ep.reset()
    h = {'reason': 'missing_tool', 'text': 'test'}
    ep.step({'goal_version': 1, 'help': h})
    b = ep.step({'goal_version': 1, 'heading_deg': 30, 'help': h})
    assert b['refusals'][0]['reason'] == 'at most one help request per episode'
    assert math.degrees(ep.receiver.commanded) == 90
    assert len(ep.record.help_requests) == 1


def test_reset_matches_fresh_world_and_clears_all_pending_state():
    sc = scenario(delay_s=5)
    ep = E.Episode(sc)
    initial = ep.reset()
    ep.step({'goal_version': 1, 'report_trigger_nm': 1000,
             'help': {'reason': 'missing_tool', 'text': 'test'}})
    ep.step({'goal_version': 1, 'heading_deg': 30})
    ep.record.flags.append({'kind': 'test'})
    assert ep.reset() == initial == E.Episode(sc).reset()
    assert ep.t == 0 and not ep.done
    assert not ep.record.actions and not ep.record.flags and not ep.record.help_requests
    assert not ep.queue and not ep.pending_help and ep.trigger is None
    assert ep._join_alternatives == []


def boundary_world(range_nm=20.0, outward_rate=1e-7):
    h = math.pi + math.asin(outward_rate / P.kt(450))
    r = P.ReceiverState(0, 73-range_nm, 47, h, h, P.kt(450), P.turn_rate(450))
    k = E.Tanker('SHELL 1', P.Orbit(65, 47, 8, P.kt(280), math.pi/2))
    return scenario(receiver=r, tankers=(k,))


def test_instantaneous_outbound_is_not_inbound_when_forward_sample_decreases():
    ep = E.Episode(boundary_world())
    ep.reset()
    assert ep._true_range('SHELL 1', .001) < ep._true_range('SHELL 1', 0)
    ep.step({'goal_version': 1, 'report': 'Wedgetail, aircraft 2, 20 NM from SHELL 1'})
    r = ep.record.reports[0]
    assert r['range_derivative_nm_s'] == pytest.approx(1e-7, abs=1e-16)
    assert not r['inbound'] and not r['correct'] and r['correct_options'] == [False]


def test_report_boundary_and_tangency_record_only_physically_possible_correct_options():
    ep = E.Episode(boundary_world(20.25, outward_rate=0))
    ep.reset()
    ep.step({'goal_version': 1, 'report': 'Wedgetail, aircraft 2, 20 NM from SHELL 1'})
    assert {f['kind'] for f in ep.record.flags} == {'report inbound near zero', 'report range near tolerance'}
    assert ep.record.reports[0]['correct_options'] == [False, True]
    # Wrong naming cannot become correct by reversing a numeric boundary interpretation.
    bad = E.Episode(boundary_world(20.25, outward_rate=0)); bad.reset()
    bad.step({'goal_version': 1, 'report': 'not the required content'})
    assert bad.record.reports[0]['correct_options'] == [False]


def test_flagged_join_alternative_keeps_old_goal_and_joint_report_options(monkeypatch):
    sc = scenario(goals=((0, E.Goal(1, 'SHELL 1', 20)), (1, E.Goal(2, 'SHELL 2', 15))))
    ep = E.Episode(sc); ep.reset()
    ep.record.reports.append({'goal_version': 1, 'correct': True, 'correct_options': [False, True]})
    a = P.Approach(.250002, .249998, .5, 3, 'resolved')
    monkeypatch.setattr(P, 'join_assessment', lambda *args: ('flagged', a))
    ep.step({})
    assert ep.goal.version == 2
    out = ep.outcome(); alternative = out['alternatives'][0]
    assert not out['settled'] and out['goal']['version'] == 2
    assert alternative['goal']['version'] == 1 and alternative['ended_at'] == 1
    assert alternative['goal_met_options'] == [False, True]
    assert alternative['action_steps'] == 1 and alternative['help_requests'] == []
    # Later wrong reports do not revise the earlier possible ending.
    ep.record.reports.append({'goal_version': 1, 'correct': False, 'correct_options': [False]})
    assert ep.outcome()['alternatives'][0] == alternative
    assert ep.outcome()['possible_goal_met'] == [False, True]


def test_unresolved_join_is_not_a_settled_failure(monkeypatch):
    ep = E.Episode(scenario(goals=((0, E.Goal(1, 'SHELL 1', 20, deadline=1)),))); ep.reset()
    a = P.Approach(1, 0, .5, 3, 'unresolved')
    monkeypatch.setattr(P, 'join_assessment', lambda *args: ('unresolved', a))
    assert ep.step({}) is None
    out = ep.outcome()
    assert out['end_reason'] == 'deadline' and not out['settled']
    assert out['alternatives'][0]['end_reason'] == 'join'
    assert out['alternatives'][0]['join_assessment']['status'] == 'unresolved'
