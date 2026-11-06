"""Fixed mathematical trajectories; no world/reference/controller or data draws."""
from dataclasses import asdict, replace
import json
import math
import pytest
from paper2.experiment2.contact_bounds import Settings; from paper2.experiment2.contact_bounds import contact_lower
from paper2.experiment2.monitor_labels import Original

O = Original("analytic-fixture",1,180, *(["a"*64]*5))


def fixed_tanker(t): return (10.0,0.0)
def fixed_receiver(t): return (0.0,0.0)


def test_static_contact_time_computed_independently():
    # Speed-ball reaches radius at (10-.25)/.1=97.5 seconds.
    e = contact_lower(O,fixed_tanker,fixed_receiver,.1,0,300,prefix_seconds=0)
    assert e.state=="contact_not_excluded"
    assert 97.48 <= e.elapsed_lower_seconds <= 97.5
    assert not e.admissible
    assert all(c.lower_margin>0 for c in e.cells if c.disposition=="excluded")
    json.dumps(asdict(e),allow_nan=False)


def test_committed_prefix_and_elapsed_origin_are_not_reset():
    # Prefix moves west by .1 NM; the free continuation must then cover 9.85 NM.
    # True relaxed earliest time is 1+(10.1-.25)/.1 = 99.5 elapsed seconds.
    prefix = lambda t: (-.1*(t-180),0)
    e = contact_lower(O,fixed_tanker,prefix,.1,0,300)
    assert 99.48 <= e.elapsed_lower_seconds <= 99.5
    shifted = contact_lower(replace(O,second=0),fixed_tanker,lambda t:(-.1*t,0),.1,0,120)
    assert shifted.elapsed_lower_seconds==e.elapsed_lower_seconds


def test_approaching_target_speed_is_included_in_exclusion():
    # T(s)=10-.02*(s-180); ball .1*(s-180); root 9.75/.12=81.25.
    e = contact_lower(O,lambda t:(10-.02*(t-180),0),fixed_receiver,.1,.02,300,prefix_seconds=0)
    assert 81.23 <= e.elapsed_lower_seconds <= 81.25


def test_prefix_contact_prevents_a_later_relaxation_bound():
    e = contact_lower(O,lambda t:(.25,0),fixed_receiver,.1,0,300)
    assert e.state=="contact_not_excluded" and e.elapsed_lower_seconds==0
    assert {c.phase for c in e.cells}=={"fixed_prefix"}


def test_closed_deadline_exclusion_is_only_conditional_contact_evidence():
    e = contact_lower(O,fixed_tanker,fixed_receiver,.1,0,190,prefix_seconds=0)
    assert e.state=="deadline_contact_excluded" and e.elapsed_lower_seconds==10
    assert e.qualification=="conditional_motion_speed_and_error_assumptions" and not e.admissible
    # Exactly on a contact boundary at the deadline may not be excluded.
    boundary = contact_lower(O,lambda t:(1.25,0),fixed_receiver,.1,0,190,prefix_seconds=0)
    assert boundary.state=="contact_not_excluded" and boundary.elapsed_lower_seconds<=10


def test_zero_length_interval_does_not_hide_initial_contact():
    e = contact_lower(O,lambda t:(.25,0),fixed_receiver,.1,0,180,prefix_seconds=0)
    assert e.state=="contact_not_excluded" and e.elapsed_lower_seconds==0


def test_cap_retains_first_unresolved_edge_and_partial_work():
    e = contact_lower(O,fixed_tanker,fixed_receiver,.1,0,300,prefix_seconds=0,
                      settings=Settings(evaluation_cap=1))
    assert e.state=="evaluation_cap" and e.elapsed_lower_seconds==0
    assert e.evaluations==1 and dict(e.motion_calls)=={"tanker":1,"prefix":1}
    assert e.cells[-1].disposition=="cap"


@pytest.mark.parametrize("bad",[float("nan"),float("inf"),True,"0"])
def test_motion_error_retains_attempted_queries(bad):
    e = contact_lower(O,lambda t:(bad,0),fixed_receiver,.1,0,300)
    assert e.state=="error" and e.elapsed_lower_seconds==0
    assert e.evaluations==1 and dict(e.motion_calls)=={"tanker":1,"prefix":0}
    assert e.cells[-1].error


def test_provider_exception_does_not_disappear_or_advance_the_bound():
    def broken(t): raise RuntimeError("invented curve failure")
    e = contact_lower(O,broken,fixed_receiver,.1,0,300)
    assert e.state=="error" and e.elapsed_lower_seconds==0
    assert "invented curve failure" in e.cells[-1].error


@pytest.mark.parametrize("kw",[dict(prefix_seconds=30),dict(prefix_seconds=True),
    dict(settings=Settings(evaluation_cap=0)),dict(settings=Settings(evaluation_cap=True)),
    dict(settings=Settings(resolution_seconds=0)),dict(settings=Settings(distance_allowance_nm=-1)),
    dict(settings=Settings(contact_radius_nm=-1)),dict(settings=Settings(resolution_seconds=float("nan")))])
def test_invalid_settings_refuse_before_any_motion_query(kw):
    with pytest.raises(ValueError):
        contact_lower(O,lambda t:pytest.fail("queried"),fixed_receiver,.1,0,300,**kw)


@pytest.mark.parametrize("S,V,deadline",[(-1,0,300),(.1,-1,300),(True,0,300),(.1,0,180),(.1,0,3601),(.1,0,float("inf"))])
def test_invalid_units_time_and_speed_refuse_before_queries(S,V,deadline):
    with pytest.raises(ValueError):
        contact_lower(O,lambda t:pytest.fail("queried"),fixed_receiver,S,V,deadline)


@pytest.mark.parametrize("second",[3601,10**1000])
def test_out_of_episode_original_refuses_before_queries(second):
    with pytest.raises(ValueError):
        contact_lower(replace(O,second=second),lambda t:pytest.fail("queried"),
                      fixed_receiver,.1,0,3600,prefix_seconds=0)
