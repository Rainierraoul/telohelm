"""Independent estimator checks from hand-computable public observations.

No world, pilot, scenario, certificate, original collection or scored stream.
"""
from copy import deepcopy
import math

import pytest
from paper2.experiment2.adaptive_controller import EpisodeEstimator; from paper2.experiment2.adaptive_controller import InterfaceError


def briefing():
    return {"own_callsign": "RECEIVER", "control_callsign": "CONTROL",
            "goal": {"version": 1, "tanker": "A", "report_at_nm": 20, "deadline": None},
            "tankers": [{"callsign": "A", "centre": [65, 47], "radius": 8,
                         "speed_nm_s": 280 / 3600, "direction": "clockwise", "position_at_0": [65, 55]}]}


def bundle(t, reports=(), goal=None):
    # Northbound air velocity 0.125 and an eastward wind 0.01 NM/s.
    return {"t": t, "own": {"heading_deg": 0, "tas_nm_s": .125,
                             "track_deg": math.degrees(math.atan2(.01, .125)),
                             "ground_speed_nm_s": math.hypot(.01, .125)},
            "reports": list(reports), "goal": goal}


def report(measured, received, radius=8, angle=0):
    return {"callsign": "A", "measured_at": measured, "received_at": received,
            "x": 65 + radius * math.sin(angle), "y": 47 + radius * math.cos(angle)}


@pytest.fixture
def estimator():
    obj = EpisodeEstimator()
    obj.reset(briefing())
    obj.update(bundle(0))
    return obj


def test_wind_vector_uses_bundle_nm_per_second(estimator):
    assert estimator.wind_nm_s == pytest.approx((.01, 0), abs=1e-15)
    # A westbound aircraft tests heading/track wrap rather than a second unit conversion.
    b = bundle(1)
    b['own'] = {'heading_deg': 270, 'tas_nm_s': .125, 'track_deg': 270, 'ground_speed_nm_s': .115}
    estimator.update(b)
    assert estimator.wind_nm_s == pytest.approx((.01, 0), abs=1e-15)


def test_pre_onset_report_does_not_reveal_whether_change_will_occur(estimator):
    estimator.update(bundle(60, [report(60, 60, angle=.4)]))
    belief = estimator.targets['A']
    assert (belief.change_low, belief.change_high) == pytest.approx((-1.6, 1.6))
    assert belief.no_change_consistent
    hypotheses = estimator.forecast('A', 180)
    assert {round(h['change_nm'], 6) for h in hypotheses} == {-1.6, 0, 1.6}


def test_visible_radius_change_refines_conditional_prior(estimator):
    # At 120 s the declared half-cosine has completed one half of its +0.8 NM change.
    estimator.update(bundle(120, [report(120, 120, radius=8.4, angle=.9)]))
    belief = estimator.targets['A']
    assert belief.change_low == pytest.approx(.8 - 2e-6)
    assert belief.change_high == pytest.approx(.8 + 2e-6)
    assert not belief.no_change_consistent


def test_late_delivery_and_duplicate_measurement_are_not_fresh_evidence(estimator):
    estimator.update(bundle(180, [report(180, 180, radius=8.8, angle=1.2)]))
    estimator.update(bundle(181, [report(120, 181, radius=8.4, angle=.9),
                                  report(180, 181, radius=8.8, angle=1.2)]))
    belief = estimator.targets['A']
    assert belief.latest.measured_at == 180
    assert len(belief.observations) == 3  # briefing, 120, 180; repeat is not independent.
    assert (belief.change_low, belief.change_high) == pytest.approx((.8-1e-6, .8+1e-6))


def test_conflicting_duplicate_rejects_whole_bundle(estimator):
    before = deepcopy(estimator.targets)
    g = dict(briefing()['goal'], version=2, report_at_nm=10)
    with pytest.raises(InterfaceError, match='conflicting'):
        estimator.update(bundle(1, [report(0, 1, angle=.5)], goal=g))
    assert estimator.goal.version == 1
    assert estimator.t == 0
    assert estimator.targets == before


def test_constant_circle_forecast_agrees_with_analytic_quarter_turn(estimator):
    estimator.update(bundle(180, [report(180, 180, angle=0)]))
    quarter_turn = math.pi * 8 / (2 * (280 / 3600))
    predictions = estimator.forecast('A', 180 + quarter_turn)
    nominal = next(p for p in predictions if p['change_nm'] == 0)
    assert (nominal['x'], nominal['y']) == pytest.approx((73, 47), abs=1e-10)
    assert all(math.isfinite(p['estimated_angle_error_rad']) for p in predictions)
    assert estimator.forecast_calls == 1
    assert estimator.cpu_ns > 0


@pytest.mark.parametrize('measured,received,t', [(2,1,1), (0,0,1), (-1,1,1), (.5,1,1)])
def test_invalid_report_times_leave_state_unchanged(estimator, measured, received, t):
    with pytest.raises(InterfaceError):
        estimator.update(bundle(t, [report(measured, received)]))
    assert estimator.t == 0
    assert len(estimator.targets['A'].observations) == 1


def test_goal_versions_and_reset_are_episode_local(estimator):
    estimator.update(bundle(180, [report(180, 180, radius=8.8)]))
    estimator.forecast('A', 200)
    estimator.update(bundle(181, goal=dict(briefing()['goal'], version=2, deadline=400)))
    assert estimator.goal.version == 2
    assert estimator.targets['A'].latest.measured_at == 180
    with pytest.raises(InterfaceError, match='without a new version'):
        estimator.update(bundle(182, goal=dict(briefing()['goal'], version=2, deadline=401)))
    with pytest.raises(InterfaceError, match='regressed'):
        estimator.update(bundle(182, goal=briefing()['goal']))
    with pytest.raises(InterfaceError, match='backwards'):
        estimator.update(bundle(180))
    estimator.reset(briefing())
    assert estimator.t is None and estimator.wind_nm_s is None
    assert estimator.cpu_ns == estimator.forecast_calls == 0
    assert estimator.goal.version == 1
    assert len(estimator.targets['A'].observations) == 1
    assert estimator.targets['A'].no_change_consistent


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), True, '0.125'])
def test_invalid_navigation_numbers_rejected(estimator, bad):
    b = bundle(1)
    b['own']['tas_nm_s'] = bad
    with pytest.raises(InterfaceError):
        estimator.update(b)
    assert estimator.t == 0


def test_unknown_targets_and_past_forecasts_rejected(estimator):
    with pytest.raises(InterfaceError, match='unbriefed'):
        estimator.forecast('B', 2)
    estimator.update(bundle(3))
    with pytest.raises(InterfaceError, match='non-past'):
        estimator.forecast('A', 2)
    assert estimator.forecast_calls == 0


def test_invalid_new_brief_does_not_erase_previous_episode(estimator):
    b = briefing()
    b['tankers'][0]['position_at_0'] = [65, 60]
    with pytest.raises(InterfaceError):
        estimator.reset(b)
    assert estimator.t == 0 and estimator.goal.version == 1
    assert estimator.targets['A'].latest.y == 55


def test_hidden_extras_are_not_read(estimator):
    class Forbidden:
        def __float__(self):
            raise AssertionError('hidden information accessed')
        def __bool__(self):
            raise AssertionError('hidden information accessed')
    b = bundle(1)
    b.update(phase=Forbidden(), seed=Forbidden(), future=Forbidden(), certificate=Forbidden())
    estimator.update(b)
    assert estimator.targets['A'].no_change_consistent
