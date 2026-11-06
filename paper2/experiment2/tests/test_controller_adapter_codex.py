"""Synthetic interface/report/help tests; no development or scored episodes."""
import math

import pytest

from paper2.experiment2.adaptive_controller import AdaptiveController; from paper2.experiment2.adaptive_controller import AdapterSettings; from paper2.experiment2.adaptive_controller import InterfaceError
from paper2.experiment2.adaptive_controller.guidance import Plan; from paper2.experiment2.adaptive_controller.guidance import SearchSettings


def briefing():
    return {'own_callsign': 'RECEIVER', 'control_callsign': 'CONTROL',
            'goal': {'version': 1, 'tanker': 'A', 'report_at_nm': 20, 'deadline': None},
            'tankers': [{'callsign': name, 'centre': [x, 47], 'radius': 8,
                         'speed_nm_s': 280/3600, 'direction': 'clockwise', 'position_at_0': [x, 55]}
                        for name, x in [('A', 65), ('B', 165)]]}


def bundle(t=0, *, heading=0, y=35, x=65, goal=None, reports=(), refusals=()):
    return {'t': t, 'own': {'x': x, 'y': y, 'heading_deg': heading,
                           'commanded_heading_deg': heading, 'tas_nm_s': .125,
                           'track_deg': heading, 'ground_speed_nm_s': .125},
            'goal': goal, 'reports': list(reports), 'refusals': list(refusals)}


def fixed_plan():
    return Plan(90, 180, 30, (), True, True, 4, 61, 60, False, 3, 1)


@pytest.fixture
def controller(monkeypatch):
    obj = AdaptiveController()
    obj.reset(briefing())
    monkeypatch.setattr(obj.search, 'choose', lambda *a, **kw: fixed_plan())
    return obj


def test_report_uses_current_inbound_geometry_and_public_content(controller):
    action = controller.act(bundle())
    assert action == {'goal_version': 1, 'heading_deg': 90,
                      'report': 'CONTROL, RECEIVER, A at 20 NM inbound'}
    h = controller.trace[0]['report_hypotheses']
    assert all(v['range_nm'] == 20 and v['range_rate_nm_s'] == -.125 for v in h)


def test_outbound_current_motion_prevents_report_despite_optimistic_plan(controller):
    assert 'report' not in controller.act(bundle(heading=180))
    assert controller.trace[0]['plan']['predicted_report_possible']


@pytest.mark.parametrize('y', [34.8, 35.2, 30])
def test_report_guard_is_strict_inside_public_tolerance(controller, y):
    assert 'report' not in controller.act(bundle(y=y))


def test_finite_hypothesis_disagreement_prevents_report(controller, monkeypatch):
    monkeypatch.setattr(controller.estimator, 'forecast', lambda *a: [
        {'change_nm': 0, 'x': 65, 'y': 55, 'vx': 0, 'vy': 0},
        {'change_nm': 1, 'x': 65, 'y': 56, 'vx': 0, 'vy': 0}])
    assert 'report' not in controller.act(bundle())


def test_zero_range_has_no_inbound_clearance(controller, monkeypatch):
    monkeypatch.setattr(controller.estimator, 'forecast', lambda *a: [
        {'change_nm': 0, 'x': 65, 'y': 35, 'vx': 0, 'vy': 0}])
    assert 'report' not in controller.act(bundle())
    assert controller.trace[0]['report_hypotheses'][0]['range_rate_nm_s'] is None


def test_one_report_per_version_then_new_goal_can_report(controller):
    assert 'report' in controller.act(bundle())
    assert 'report' not in controller.act(bundle(1))
    goal = dict(briefing()['goal'], version=2, tanker='B')
    action = controller.act(bundle(2, goal=goal, x=165))
    assert action['goal_version'] == 2 and ' B ' in action['report']
    assert 'report' not in controller.act(bundle(3, x=165))


def test_atomic_refusal_rolls_back_only_last_submitted_report(controller):
    assert 'report' in controller.act(bundle())
    action = controller.act(bundle(1, refusals=[{'t': 0, 'reason': 'invalid heading'}]))
    assert 'report' in action
    assert controller.trace[-1]['previous_action_refused']


def test_unrelated_later_refusal_does_not_roll_back_earlier_report(controller):
    controller.act(bundle())
    controller.act(bundle(1))
    assert 'report' not in controller.act(bundle(2, refusals=[{'t': 1, 'reason': 'invalid heading'}]))


def test_stale_evidence_one_help_attempt_across_goal_change_and_refusal(controller):
    assert 'help' not in controller.act(bundle(10, y=0))
    first = controller.act(bundle(11, y=0))
    assert first['help']['reason'] == 'missing_or_stale_evidence'
    goal = dict(briefing()['goal'], version=2, tanker='B')
    assert 'help' not in controller.act(bundle(12, goal=goal, y=0,
                                             refusals=[{'t': 11, 'reason': 'invalid action'}]))
    assert controller.trace[-1]['help_decision'] == 'already_attempted'


@pytest.mark.parametrize('remaining,requests', [(29, False), (30, False), (31, True)])
def test_help_requires_reply_before_deadline(controller, remaining, requests):
    goal = dict(briefing()['goal'], version=2, deadline=11+remaining)
    assert ('help' in controller.act(bundle(11, goal=goal, y=0))) == requests


def test_pending_old_target_help_does_not_revert_current_goal(controller):
    controller.act(bundle(11, y=0))
    goal = dict(briefing()['goal'], version=2, tanker='B')
    controller.act(bundle(12, goal=goal, y=0))
    theta = (280/3600)*41/8
    report = {'callsign': 'A', 'x': 65+8*math.sin(theta), 'y': 47+8*math.cos(theta),
              'measured_at': 41, 'received_at': 41, 'help_reply': True}
    action = controller.act(bundle(41, reports=[report], y=0))
    assert action['goal_version'] == 2
    assert controller.estimator.targets['A'].latest.measured_at == 41
    assert 'help' not in action


def test_deadline_bundle_emits_no_action_or_search(controller, monkeypatch):
    monkeypatch.setattr(controller.search, 'choose', lambda *a, **kw: pytest.fail('deadline search'))
    goal = dict(briefing()['goal'], version=2, deadline=10)
    assert controller.act(bundle(10, goal=goal)) == {}
    assert controller.trace[-1]['decision'] == 'deadline_reached'


def test_duplicate_time_and_invalid_bundle_do_not_submit_or_rollback(controller):
    controller.act(bundle())
    before = controller.trace
    with pytest.raises(InterfaceError, match='increasing'):
        controller.act(bundle())
    with pytest.raises(InterfaceError, match='preceding'):
        controller.act(bundle(1, refusals=[{'t': 7, 'reason': 'bad'}]))
    invalid = bundle(1, goal=dict(briefing()['goal'], report_at_nm=10))
    with pytest.raises(InterfaceError, match='new version'):
        controller.act(invalid)
    assert controller.trace == before and controller.estimator.t == 0
    assert 'report' not in controller.act(bundle(1))


def test_reset_clears_help_reports_estimates_trace_and_search(controller, monkeypatch):
    controller.act(bundle())
    controller.act(bundle(11, y=0))
    controller.reset(briefing())
    monkeypatch.setattr(controller.search, 'choose', lambda *a, **kw: fixed_plan())
    assert controller.trace == [] and controller.cpu_ns == 0 and controller.search.searches == 0
    assert 'report' in controller.act(bundle())
    assert 'help' in controller.act(bundle(11, y=0))


def test_returned_actions_and_trace_cannot_mutate_policy(controller):
    action = controller.act(bundle())
    action['report'] = 'changed'
    trace = controller.trace
    trace[0]['actions'].clear()
    assert controller.trace[0]['actions']['report'].startswith('CONTROL')
    assert 'report' not in controller.act(bundle(1))
    assert controller.cpu_ns >= sum(v['cpu_ns'] for v in controller.trace) > 0


@pytest.mark.parametrize('kwargs', [{'report_guard_nm': .25}, {'report_guard_nm': False},
                                     {'inbound_guard_nm_s': 0}, {'stale_after_s': True},
                                     {'stale_after_s': -1}])
def test_invalid_settings_refused(kwargs):
    with pytest.raises(InterfaceError):
        AdapterSettings(**kwargs)


def test_real_estimator_search_adapter_path_on_synthetic_bundle():
    obj = AdaptiveController(SearchSettings(horizon_s=60, heading_spacing_deg=90, second_leg_at_s=30))
    obj.reset(briefing())
    action = obj.act(bundle(y=0))
    assert 0 <= action['heading_deg'] < 360 and action['goal_version'] == 1
    assert 'report' not in action and 'help' not in action
    assert obj.search.searches == 1 and obj.estimator.forecast_calls > 1
    assert obj.cpu_ns >= obj.search.cpu_ns
    assert obj.trace[0]['plan']['horizon_s'] == 60


def test_decimal_report_distance_and_policy_settings_are_audited(controller):
    goal = dict(briefing()['goal'], version=2, report_at_nm=20.1234567)
    action = controller.act(bundle(goal=goal, y=55-goal['report_at_nm']))
    assert '20.1234567 NM' in action['report'] and 'e-' not in action['report']
    trace = controller.trace[0]
    assert trace['adapter_settings']['report_guard_nm'] == .20
    assert trace['search_settings']['sample_s'] == 1
    assert trace['coordinate_allowance_nm'] == 1e-6
