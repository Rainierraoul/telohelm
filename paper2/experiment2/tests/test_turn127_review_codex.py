"""Invented numerical boundary and record-retention controls for TURN127. No empirical outcomes."""
import json
import math
from types import SimpleNamespace

import pytest

import paper2.experiment2.comparators as K
import paper2.experiment2.adaptive_controller.guidance as G
from .test_comparators import scenario; from .test_comparators import condition; from .test_comparators import log
import telohelm.taskpacks.tanker.world.episode as E


@pytest.mark.parametrize('heading', [50.0, 100.0])
def test_completed_turn_to_north_stays_in_public_heading_range_without_motion_change(heading):
    tas=.125
    own=G.OwnState(0,0,heading,heading,tas)
    result=G.propagate(own,0,120,(0,0))
    rate=9.80665*math.tan(math.pi/6)/(tas*1852)
    h=math.radians(heading);turn=h/rate
    assert result.heading_deg == 0 and result.commanded_heading_deg == 0
    assert result.x == pytest.approx(tas*(1-math.cos(h))/rate,abs=1e-12)
    assert result.y == pytest.approx(tas*math.sin(h)/rate+tas*(120-turn),abs=1e-12)


def test_tiny_negative_lead_angle_is_canonical_before_it_becomes_a_candidate():
    class Estimate:
        t=0
        goal=SimpleNamespace(deadline=None,report_at_nm=20,tanker='invented')
        wind_nm_s=(0,0)
        def forecast(self,*args):
            return [{'change_nm':0,'x':-1e-16,'y':10.,'vx':0.,'vy':0.}]
    search=G.GuidanceSearch(G.SearchSettings(horizon_s=4,heading_spacing_deg=90,second_leg_at_s=2))
    plan=search.choose(Estimate(),G.OwnState(0,0,0,0,.125))
    assert 0<=plan.first_heading_deg<360 and 0<=plan.second_heading_deg<360


@pytest.mark.parametrize('radians', [-1e-16, -math.pi/2, 0, math.pi, math.tau, math.radians(359.9999999999)])
def test_derived_heading_wrap_preserves_direction_and_does_not_snap_valid_neighbours(radians):
    h=G._heading_degrees(radians)
    assert 0<=h<360
    assert math.sin(math.radians(h))==pytest.approx(math.sin(radians),abs=1e-14)
    assert math.cos(math.radians(h))==pytest.approx(math.cos(radians),abs=1e-14)
    original=math.degrees(radians)%360
    assert h==(0.0 if original==360 else original)


def test_full_world_outcome_is_retained_in_summary_and_protected_ending(tmp_path,monkeypatch):
    # Fabricated alternative-output metadata tests lossless recording, not a claimed physical outcome/label.
    original=E.Episode.outcome;seen=[]
    def outcome(self):
        r=original(self)
        r.update(settled=False,possible_goal_met=[False,True],flags=[{'kind':'invented numerical ambiguity'}],
                 alternatives=[{'goal':{'version':1},'event_cutoff_t':1,'goal_met_options':[False,True]}])
        seen.append(r)
        return r
    monkeypatch.setattr(E.Episode,'outcome',outcome)
    result=K.run_pilot_episode(scenario(deadline=2),condition(),tmp_path/'e','invented')
    assert result['outcome']==seen[-1]
    assert json.loads((tmp_path/'e/summary.json').read_text())['outcome']==seen[-1]
    ending=[r for r in log(tmp_path/'e') if r['kind']=='episode_end'][-1]
    assert ending['outcome']==seen[-1]


def test_hold_only_fixed_and_telohelm_paths_share_public_bundles_and_natural_ending(tmp_path):
    import telohelm.runtime.broker as B; import telohelm.runtime.episode_loop as L
    from telohelm.runtime.ledger import Ledger
    from telohelm.runtime.library import Library
    from telohelm.runtime.trusted_log import TrustedLog; from telohelm.runtime.trusted_log import verify
    sc=scenario(deadline=12)
    fixed=K.run_pilot_episode(sc,condition(),tmp_path/'fixed','invented')
    ledger=Ledger()
    br=B.Broker(sc.episode_id,TrustedLog(tmp_path/'broker.log',tmp_path/'broker.heads','invented'),ledger,
                lambda _:pytest.fail('hold fixture must not predict'))
    telo=L.run_episode(sc,Library(tmp_path/'library'),L.NoModels(),
                       TrustedLog(tmp_path/'telo.log',tmp_path/'telo.heads','invented'),ledger,broker=br)
    assert fixed['complete'] and fixed['action_steps']==12
    for field in ['goal_met','joined','report_correct','end_reason','ended_at','settled','action_steps']:
        assert fixed[field]==telo['outcome'][field]
    br._log.close()
    assert verify(tmp_path/'broker.log',tmp_path/'broker.heads')['ok']
    events=[json.loads(l)['payload'] for l in (tmp_path/'broker.log').read_text().splitlines()]
    a=[r['bundle'] for r in log(tmp_path/'fixed') if r['kind']=='bundle']
    b=[r['bundle'] for r in events if r['kind']=='bundle']
    assert a==b and len(a)==12 and ledger.calls==[]
