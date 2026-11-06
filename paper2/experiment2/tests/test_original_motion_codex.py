"""Invented analytic fixtures, no sampled world or corpus cases."""
from dataclasses import replace, FrozenInstanceError
import math
import pytest
from paper2.experiment2.original_motion import Receiver; from paper2.experiment2.original_motion import CircularTanker; from paper2.experiment2.original_motion import FrozenMotion; from paper2.experiment2.original_motion import ChangingTanker; from paper2.experiment2.original_motion import ChangingMotion
from paper2.experiment2.monitor_labels import Original


def original(second=12):
    h='a'*64
    return Original('invented-motion',1,second,h,h,h,h,h)


def motion(*,second=12,heading=0,command=0,wind=(0,0),speed=.1,omega=.2):
    return FrozenMotion.bind(original(second),Receiver(second,0,0,heading,speed,omega),
                             CircularTanker(0,0,2,0,0),wind,command)


def test_straight_original_clock_and_detached_wind():
    wind=[.001,-.002]
    m=motion(wind=wind)
    wind[0]=1
    assert m.receiver_at(12)==(0,0)
    assert m.receiver_at(22)==pytest.approx((.01,.98))
    assert m.wind_nm_s==(.001,-.002)
    with pytest.raises(FrozenInstanceError): m.commanded_heading_deg=90


@pytest.mark.parametrize('command,sign',[(90,1),(270,-1)])
def test_quarter_turn_and_straight_continuation(command,sign):
    m=motion(command=command)
    tau=(math.pi/2)/.2
    assert m.receiver_at(12+tau)==pytest.approx((sign*.5,.5))
    assert m.receiver_at(12+tau+10)==pytest.approx((sign*1.5,.5))


def test_half_turn_tie_and_no_early_end():
    m=motion(command=180)
    assert m.receiver_at(12+math.pi/.2)==pytest.approx((1,0),abs=1e-14)
    tie=motion(command=math.degrees(math.pi+5e-10))
    assert tie.receiver_at(13)[0]>0
    outside=motion(command=math.degrees(math.pi+2e-9))
    assert outside.receiver_at(13)[0]<0


def test_absolute_tanker_clock_cardinal_positions():
    m=FrozenMotion.bind(original(0),Receiver(0,0,0,0,.1,.2),
                        CircularTanker(3,4,2,.05,0),(0,0),0)
    assert m.tanker_at(0)==pytest.approx((3,6))
    assert m.tanker_at(math.pi/.05)==pytest.approx((5,4))
    assert m.tanker_speed_bound_nm_s==.05


def test_fingerprint_retains_original_goal_command_and_speed():
    m=motion()
    def bound(o,r,c):return FrozenMotion.bind(o,r,m.tanker,m.wind_nm_s,c).fingerprint
    assert bound(m.original,m.receiver,0)==m.fingerprint
    assert bound(replace(m.original,goal_version=2),m.receiver,0)!=m.fingerprint
    assert bound(m.original,m.receiver,90)!=m.fingerprint
    assert bound(m.original,replace(m.receiver,tas_nm_s=.11),0)!=m.fingerprint
    assert m.admissible is False and m.qualification.startswith('conditional')


def test_original_elapsed_contact_bound_and_primitive_counts():
    m=motion(second=100,heading=math.pi/2,command=90)
    m=FrozenMotion.bind(m.original,m.receiver,CircularTanker(10,0,1,0,math.pi),(0,0),90)
    ev=m.lower(300)
    # Stationary target (10,-1); nearest speed-ball contact solves (sqrt(101)-.25)/.1.
    after_distance=math.hypot(9.9,1)
    relaxed=1+(after_distance-.25)/.1
    assert relaxed-.011<=ev.elapsed_lower_seconds<=relaxed
    assert ev.original==m.original and ev.prefix_seconds==1 and not ev.admissible
    assert dict(ev.motion_calls)['tanker']==ev.evaluations


@pytest.mark.parametrize('kwargs',[{'command':360},{'command':True},{'command':float('nan')},
                                 {'speed':0},{'omega':0},{'wind':(.1,0)},{'heading':math.tau}])
def test_invalid_motion_domains_are_refused(kwargs):
    with pytest.raises(ValueError):motion(**kwargs)


@pytest.mark.parametrize('second',[11,3601,float('inf'),True])
def test_queries_do_not_reset_or_leave_original_clock(second):
    with pytest.raises(ValueError):motion().receiver_at(second)


def test_receiver_original_time_must_match():
    with pytest.raises(ValueError):
        FrozenMotion.bind(original(12),Receiver(11,0,0,0,.1,.2),CircularTanker(0,0,2,0,0),(0,0),0)


def changing(*,second=0,delta=.5,start=60,duration=120,speed=.05):
    return ChangingMotion.bind(original(second),Receiver(second,0,0,0,.1,.2),
                               ChangingTanker(3,4,2,speed,0,delta,start,duration),(0,0),0)


def test_zero_delta_reduces_to_constant_orbit_without_quadrature():
    m = changing(delta=0)
    constant = FrozenMotion.bind(m.original,m.receiver,CircularTanker(3,4,2,.05,0),(0,0),0)
    for s in (0,60,120,180,3600):
        q=m.tanker_query(s)
        assert q.point == pytest.approx(constant.tanker_at(s))
        assert q.integrand_evaluations == 0 and q.angular_error_estimate == 0


@pytest.mark.parametrize('delta',[-.5,.5])
def test_change_absolute_clock_and_independent_composite_simpson(delta):
    m=changing(delta=delta,second=100)
    # Invented half-cosine fixture: independent fixed-grid integration, not
    # production Orbit.position or the adaptive backend being checked.
    def independent_angle(s):
        n=4096; hi=min(s,180); dx=(hi-60)/n
        def f(u):
            r=2+delta*(1-math.cos(math.pi*(u-60)/120))/2
            rd=delta*math.pi/240*math.sin(math.pi*(u-60)/120)
            return math.sqrt(.05**2-rd**2)/r
        integral=dx/3*(f(60)+f(hi)+sum((4 if i%2 else 2)*f(60+i*dx) for i in range(1,n)))
        return .05*60/2+integral+(.05*(s-180)/(2+delta) if s>180 else 0)
    for s in (100,120,180,210):
        a=independent_angle(s)
        r=2+delta*(1-math.cos(math.pi*(s-60)/120))/2 if s<180 else 2+delta
        q=m.tanker_query(s)
        assert q.state == 'ok' and q.integrand_evaluations > 0
        assert q.point == pytest.approx((3+r*math.sin(a),4+r*math.cos(a)),abs=2e-11)
        assert q.angular_error_estimate >= 0 and q.position_error_estimate_nm == r*q.angular_error_estimate
    assert m.tanker_speed_bound_nm_s == .05
    assert m.receiver_at(100) == (0,0)  # Propagation starts at the original second, not zero.


def test_change_before_start_uses_analytic_position():
    q=changing().tanker_query(60)
    assert q.point == pytest.approx((3+2*math.sin(1.5),4+2*math.cos(1.5)))
    assert q.integrand_evaluations == 0


@pytest.mark.parametrize('kwargs',[{'delta':-2},{'delta':10},{'duration':0},
                                  {'duration':3600},{'start':-1},{'speed':0}])
def test_invalid_changing_domains_are_refused(kwargs):
    with pytest.raises(ValueError):changing(**kwargs)


def test_change_fingerprint_retains_all_change_fields_and_original():
    m=changing()
    assert m.fingerprint != changing(delta=.4).fingerprint
    assert m.fingerprint != changing(start=61).fingerprint
    assert m.fingerprint != changing(duration=121).fingerprint
    assert m.fingerprint != changing(second=1).fingerprint
    assert not m.admissible and m.qualification.startswith('conditional')
    with pytest.raises(TypeError):
        FrozenMotion.bind(m.original,m.receiver,m.tanker,(0,0),0)


def test_query_cap_and_lower_error_retain_partial_work():
    # Fault injection into the diagnostic path. replace is NOT a bound record
    # producer; the future verifier must rebind parameters and fingerprint.
    m=replace(changing(second=100),quadrature_evaluation_cap=1)
    q=m.tanker_query(101)
    assert q.state == 'evaluation_cap' and q.point is None and q.integrand_evaluations == 1
    ev=m.lower(200)
    assert ev.contact.state == 'error' and ev.contact.elapsed_lower_seconds == 0
    assert len(ev.tanker_queries)==1 and ev.integrand_evaluations==1 and not ev.admissible


def test_quadrature_nonconvergence_is_error_not_a_contact_value(monkeypatch):
    import scipy.integrate
    monkeypatch.setattr(scipy.integrate,'quad',lambda *a,**k:(0,1,{'neval':0},'invented nonconvergence'))
    m=changing(second=100)
    q=m.tanker_query(101)
    assert q.state=='error' and q.point is None and 'nonconvergence' in q.error
    ev=m.lower(200)
    assert ev.contact.state=='error' and ev.tanker_queries[0].state=='error'


def test_changing_lower_retains_original_elapsed_clock_and_all_backend_work():
    m=changing(second=100)
    ev=m.lower(160)
    assert ev.contact.original==m.original and ev.contact.prefix_seconds==1
    assert ev.contact.elapsed_lower_seconds >= 0
    assert ev.motion_fingerprint==m.fingerprint and not ev.admissible
    assert len(ev.tanker_queries)==dict(ev.contact.motion_calls)['tanker']
    assert ev.integrand_evaluations==sum(q.integrand_evaluations for q in ev.tanker_queries)
    assert all(q.second>=100 for q in ev.tanker_queries)
