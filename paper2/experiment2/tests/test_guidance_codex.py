"""Independent synthetic geometry/interface tests; no episode or scored data."""
import math

import numpy as np
import pytest

from paper2.experiment2.adaptive_controller.estimation import EpisodeEstimator; from paper2.experiment2.adaptive_controller.estimation import Goal; from paper2.experiment2.adaptive_controller.estimation import InterfaceError
from paper2.experiment2.adaptive_controller.guidance import OwnState; from paper2.experiment2.adaptive_controller.guidance import SearchSettings; from paper2.experiment2.adaptive_controller.guidance import GuidanceSearch; from paper2.experiment2.adaptive_controller.guidance import propagate; from paper2.experiment2.adaptive_controller.guidance import prediction_metrics; from paper2.experiment2.adaptive_controller.guidance import chord_minimum


@pytest.mark.parametrize('heading,command,duration', [(0,90,100), (350,10,30),
                                                     (10,350,30), (90,270,150)])
def test_turn_and_completion_against_independent_sampled_velocity(heading, command, duration):
    # Integrate the separately constructed heading history, splitting at completion.
    own = OwnState(3, -2, heading, heading, .125)
    wind = (.012, -.007)
    difference = (command - heading) % 360
    direction = 1 if difference <= 180 else -1
    angle = min(difference, 360-difference)
    rate = math.degrees(9.80665 * math.tan(math.radians(30)) / (.125 * 1852))
    completion = angle/rate
    cuts = sorted({0, duration, min(completion, duration)})
    dx = dy = 0
    for a,b in zip(cuts,cuts[1:]):
        # Twice the grid density reduces the trapezoid's O(step**2) error;
        # this reference tolerance concerns quadrature, not a world error budget.
        t = np.linspace(a,b,40001)
        h = np.radians(heading + direction * np.minimum(t*rate, angle))
        dx += np.trapz(.125*np.sin(h)+wind[0],t)
        dy += np.trapz(.125*np.cos(h)+wind[1],t)
    actual = propagate(own, command, duration, wind)
    assert (actual.x, actual.y) == pytest.approx((3+dx,-2+dy), abs=2e-8)
    assert actual.heading_deg == pytest.approx((heading+direction*min(rate*duration,angle))%360)


def test_straight_motion_wind_and_partition_invariance():
    own = OwnState(0,0,90,90,.125)
    straight = propagate(own,90,40,(.01,.02))
    assert (straight.x,straight.y) == pytest.approx((5.4,.8),abs=1e-12)
    full = propagate(own,10,75,(.01,.02))
    split = propagate(propagate(own,10,20,(.01,.02)),10,55,(.01,.02))
    assert (full.x,full.y,full.heading_deg) == pytest.approx((split.x,split.y,split.heading_deg),abs=1e-12)


class FixedForecast:
    def __init__(self, deadline=None, points=((20,0),)):
        self.t=0; self.wind_nm_s=(0,0); self.goal=Goal(1,'A',10,deadline)
        self.points=points; self.calls=[]

    def forecast(self,callsign,t):
        assert callsign == self.goal.tanker
        self.calls.append(t)
        return [{'x':x,'y':y,'change_nm':j,'vx':0,'vy':0} for j,(x,y) in enumerate(self.points)]


def test_simple_geometry_preserves_report_before_contact_and_selects_east():
    search=GuidanceSearch(SearchSettings(180,5,10,90))
    estimator=FixedForecast()
    plan=search.choose(estimator,OwnState(0,0,90,90,.125))
    assert plan.first_heading_deg == 90
    assert plan.predicted_report_possible and plan.predicted_join_all_hypotheses
    assert plan.candidates <= 100 and plan.samples_per_candidate == 37
    assert plan.horizon_s == 180 and not plan.deadline_within_horizon
    assert search.cpu_ns >= plan.cpu_ns > 0 and search.searches == 1
    repeated=search.choose(estimator,OwnState(0,0,90,90,.125))
    assert repeated.rank == plan.rank


def test_miss_is_not_a_certificate_and_disagreement_is_retained():
    plan=GuidanceSearch(SearchSettings(180,5,10,90)).choose(
        FixedForecast(points=((20,0),(20,20))), OwnState(0,0,90,90,.125),True)
    assert plan.forecast_hypotheses == 2
    assert not plan.predicted_join_all_hypotheses


def test_deadline_caps_forecasts_and_keeps_partial_final_sample():
    estimator=FixedForecast(deadline=7)
    plan=GuidanceSearch().choose(estimator,OwnState(0,0,90,90,.125),True)
    assert estimator.calls == list(range(8))
    assert plan.horizon_s == 7 and plan.deadline_within_horizon
    assert plan.samples_per_candidate == 8


def test_expired_goal_rejects_and_charges_failed_search():
    search=GuidanceSearch()
    with pytest.raises(InterfaceError,match='deadline'):
        search.choose(FixedForecast(deadline=0),OwnState(0,0,90,90,.125))
    assert search.cpu_ns > 0 and search.searches == 0


@pytest.mark.parametrize('settings', [(601,1,10,120),(300,0,10,120),(300,5,7,120),
                                      (300,5,10,300),(300,5,10,121),(True,1,10,1)])
def test_search_settings_refuse_invalid_or_excessive_budget(settings):
    with pytest.raises(InterfaceError): SearchSettings(*settings)


def test_estimator_integration_reads_only_briefing_and_public_bundle():
    estimator=EpisodeEstimator()
    estimator.reset({'own_callsign':'aircraft 2','control_callsign':'CONTROL',
      'goal':{'version':1,'tanker':'SHELL 1','report_at_nm':25,'deadline':None},
      'tankers':[{'callsign':'SHELL 1','centre':[65,47],'radius':8,'speed_nm_s':280/3600,
                  'direction':'clockwise','position_at_0':[65,55]}]})
    own={'x':0,'y':0,'heading_deg':90,'commanded_heading_deg':90,
         'tas_nm_s':.125,'track_deg':90,'ground_speed_nm_s':.125}
    estimator.update({'t':0,'own':own,'reports':[],'goal':None})
    before=(estimator.t,estimator.goal,estimator.targets['SHELL 1'].latest)
    plan=GuidanceSearch().choose(estimator,OwnState.read(own))
    assert 0 <= plan.first_heading_deg < 360
    assert (estimator.t,estimator.goal,estimator.targets['SHELL 1'].latest)==before
    assert estimator.forecast_calls == 301


def test_invalid_navigation_never_enters_search():
    with pytest.raises(InterfaceError): OwnState(0,0,90,90,0)
    with pytest.raises(InterfaceError): OwnState(float('nan'),0,90,90,.125)


@pytest.mark.parametrize('empty', [True, False])
def test_empty_or_reordered_hypotheses_are_refused(empty):
    class BrokenForecast(FixedForecast):
        def forecast(self, callsign, t):
            points = super().forecast(callsign, t)
            return [] if empty else (points if t == 0 else list(reversed(points)))
    search = GuidanceSearch()
    with pytest.raises(InterfaceError, match='hypothesis identity'):
        search.choose(BrokenForecast(points=((20,0),(20,20))), OwnState(0,0,90,90,.125))
    assert search.cpu_ns > 0 and search.searches == 0


def test_one_second_report_mesh_resolves_a_five_second_alias():
    # D=10 crossing is at 82.5 s. Both adjacent 5-s samples miss the guard.
    def metrics(times):
        track = [OwnState(.125*t,0,90,90,.125) for t in times]
        target = [dict(x=20.3125,y=0,vx=0,vy=0) for _ in times]
        return prediction_metrics(track,target,times,10,(0,0),False)
    coarse = metrics(list(range(0,181,5)))
    fine = metrics(list(range(181)))
    assert not coarse[2] and fine[2]
    assert SearchSettings().sample_s == 1


def test_chord_finds_grazing_contact_missed_by_both_integer_endpoints():
    # Relative speed 0.2 NM/s, impact parameter .24 NM: circle entry at .15 s.
    assert math.hypot(.1,.24) > .25
    distance, at, contact = chord_minimum((-.1,.24),(.1,.24))
    assert (distance,at,contact) == pytest.approx((.24,.5,.15))
    track = [OwnState(-.1,.24,90,90,.125),OwnState(.025,.24,90,90,.125)]
    target = [dict(x=0,y=0,vx=-.075,vy=0),dict(x=-.075,y=0,vx=-.075,vy=0)]
    miss, nearest, report = prediction_metrics(track,target,[0,1],.26,(0,0),False)
    assert (miss,nearest) == pytest.approx((.24,.5))
    assert report  # t=0 is inbound and precedes entry; t=1 is outbound.


def test_report_after_first_contact_is_not_a_predicted_success():
    track = [OwnState(-1,0,90,90,1),OwnState(1,0,90,90,1)]
    target = [dict(x=0,y=0,vx=0,vy=0)]*2
    assert not prediction_metrics(track,target,[0,2],.5,(0,0),False)[2]
    assert prediction_metrics(track,target,[0,2],.5,(0,0),True)[2]


def test_instantaneous_outbound_overrides_previous_sample_closing():
    track = [OwnState(0,0,90,90,.125),OwnState(.125,0,90,90,.125)]
    target = [dict(x=10.1,y=0,vx=.2,vy=0),dict(x=10.125,y=0,vx=.2,vy=0)]
    assert 10 < 10.1  # endpoint ranges decrease, but instantaneous range rate is positive.
    assert not prediction_metrics(track,target,[0,1],10,(0,0),False)[2]


@pytest.mark.parametrize('start,end,expected', [((1,1),(1,1),(math.sqrt(2),0,None)),
                                              ((.1,0),(.1,0),(.1,0,0)),
                                              ((2,0),(3,0),(2,0,None))])
def test_chord_stationary_inside_and_receding(start,end,expected):
    distance, at, contact = chord_minimum(start,end)
    assert (distance,at) == pytest.approx(expected[:2])
    assert contact == expected[2]


@pytest.mark.parametrize('t', [60,100,180,230])
def test_forecast_velocity_matches_independent_position_derivative(t):
    from paper2.experiment2.adaptive_controller.estimation import TargetBelief; from paper2.experiment2.adaptive_controller.estimation import OrbitPrior; from paper2.experiment2.adaptive_controller.estimation import Observation
    # Nonzero radial motion inside the transition; exact constant orbit outside it.
    observation=Observation(0,0,0,8)
    belief=TargetBelief('A',(0,0),8,280/3600,OrbitPrior(),1e-6,observation,1,1)
    at=belief.forecasts(t)[0]
    before=belief.forecasts(t-1e-3)[0]; after=belief.forecasts(t+1e-3)[0]
    numerical=((after['x']-before['x'])/.002,(after['y']-before['y'])/.002)
    assert (at['vx'],at['vy']) == pytest.approx(numerical,abs=4e-7)
    assert math.hypot(at['vx'],at['vy']) == pytest.approx(280/3600,abs=1e-12)
