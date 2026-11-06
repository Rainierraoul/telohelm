"""TURN141 CPU/invented controls; no provider, world flight, guidance search or reserved draw."""
from pathlib import Path
from copy import deepcopy
import hashlib,json,math
import pytest
import paper2.experiment2.streams as G; import telohelm.taskpacks.tanker.world.episode as E
from paper2.experiment2.adaptive_controller.estimation import EpisodeEstimator
import paper2.experiment2.adaptive_controller.qualification as Q


def candidate(): return G.draw(4401,1,0)


def test_controlled_draw_removes_unusable_interference_from_original_requirement(monkeypatch):
    fixed=candidate()
    monkeypatch.setattr(G,'draw',lambda *a,**k:fixed)
    unknown={'goal_met':True,'settled':False,'possible_goal_met':[False,True],'flags':['invented'],'alternatives':[]}
    good={'goal_met':True,'settled':True,'possible_goal_met':[True],'flags':[],'alternatives':[]}
    replies=iter([unknown,good]);monkeypatch.setattr(G.phase5,'fly',lambda *a,**k:next(replies))
    got=G.solvable_episode(4401,4,0)
    assert len(got.attempts)==2 and got.attempts[0]['solvable'] is False and got.attempts[1]['solvable'] is True
    assert got.attempts[0]['reason']=='reference outcome not settled' and got.attempts[0]['outcome']==unknown


def test_cap_retains_mixed_unusable_and_settled_failure_evidence(monkeypatch):
    fixed=candidate();failure={'goal_met':False,'settled':True,'possible_goal_met':[False],'flags':[],'alternatives':[]}
    def draw(stream,phase,index,attempt):
        if attempt==0:raise G.Unusable('no eligible new goal',{'t_g':61,'eligible_nm':[]})
        return fixed
    monkeypatch.setattr(G,'draw',draw);monkeypatch.setattr(G,'ATTEMPT_CAP',2)
    monkeypatch.setattr(G.phase5,'fly',lambda *a,**k:deepcopy(failure))
    with pytest.raises(G.GenerationError) as caught:G.solvable_episode(4401,4,0)
    assert [r['attempt'] for r in caught.value.attempts]==[0,1]
    assert [r['seed'] for r in caught.value.attempts]==[G.episode_seed(4401,4,0,i) for i in (0,1)]
    assert caught.value.attempts[0]['details']=={'t_g':61,'eligible_nm':[]}
    assert caught.value.attempts[1]['outcome']==failure
    assert caught.value.attempts[1]['reason']=='not solvable by the reference pilot'


def test_unsettled_nominal_failure_retains_unknown_disposition(monkeypatch):
    fixed=candidate();monkeypatch.setattr(G,'draw',lambda *a,**k:fixed)
    unknown={'goal_met':False,'settled':False,'possible_goal_met':[False,True],'flags':['invented'],'alternatives':[]}
    good={'goal_met':True,'settled':True,'possible_goal_met':[True],'flags':[],'alternatives':[]}
    replies=iter([unknown,good]);monkeypatch.setattr(G.phase5,'fly',lambda *a,**k:next(replies))
    got=G.solvable_episode(4401,4,0)
    assert got.attempts[0]['solvable'] is False and got.attempts[0]['outcome']==unknown
    assert got.attempts[0]['reason']=='reference outcome not settled'


def load_manifest(name):
    p=next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'paper2/controller'/name
    body=json.loads(p.read_text());saved=body.pop('body_sha256')
    assert hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==saved
    return body


def test_stateful_public_inputs_are_valid_without_search_and_cover_distinct_beliefs():
    body=load_manifest('stateful-timing-inputs.json');seq=body['sequences']
    assert body['measurement_authorized'] is False and len(seq)==6
    before=deepcopy(seq);summaries={}
    for case in seq:
        payload={k:v for k,v in case.items() if k!='input_sha256'}
        assert hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==case['input_sha256']
        obs=EpisodeEstimator();obs.reset(case['briefing'])
        assert case['timed_bundle_index']==len(case['bundles'])-1
        assert all(a['t']<b['t'] for a,b in zip(case['bundles'],case['bundles'][1:]))
        for bundle in case['bundles']:obs.update(bundle)
        assert all(r['measured_at']<=r['received_at']==bundle['t'] for r in bundle['reports'])
        b=obs.targets[obs.goal.tanker];summaries[case['id']]={'width':b.change_high-b.change_low,'latest':b.latest.measured_at,'goal':obs.goal.version,'tanker':obs.goal.tanker}
    assert summaries['delivered-at-zero-then-stale180']['width']==pytest.approx(3.2)
    assert summaries['narrowed-zero-change180']['width']<3e-6
    assert summaries['narrowed-positive-change180']['width']<3e-6
    assert summaries['older-after-newer-delivery190']['latest']==180
    assert summaries['old-target-help-after-goal-change']['tanker']=='B'
    assert summaries['goal-tanker-change180']['goal']==2
    assert seq==before
    # Estimator-only checks do not establish actual Adapter help/policy state or costs.


def test_numeric_manifests_expand_all_recipes_and_preserve_boundary_offsets():
    body=load_manifest('constructed-inputs.json');rows=body['inputs']
    assert len(rows)==46 and len({r['id'] for r in rows})==46
    assert {r['recipe'] for r in rows}=={r['id'] for r in Q.constructed_recipes()}
    assert all(r['labels'] is None and r['label_status']=='not measured or certified' for r in rows)
    report=[r for r in rows if r['recipe']=='report-boundary'];assert len(report)==12
    assert {round(r['input']['receiver']['y_nm']-8.-20.,5) for r in report}=={.19998,.2,.20002,.24998,.25,.25002}
    assert {r['input']['receiver']['heading_deg'] for r in report}=={0.,180.}
    contact=[r for r in rows if r['recipe']=='contact-boundary']
    assert {round(r['input']['receiver']['y_nm']-8.,5) for r in contact}=={.24998,.25,.25002}
    assert all(r['input']['goals'][0]['deadline_s']==10 for r in rows if r['recipe']=='deadline-boundary')
    assert len([r for r in rows if r['kind']=='forced-prefix-diagnostic'])==4
    for r in rows:
        for action in r['input']['prefix_actions']:
            if 'help' in action:
                assert set(action['help'])=={'reason','text'}
                assert action['help']['reason'] in E.HELP_REASONS and len(action['help']['text'])<=500
    # No physical/contact/report/help label is inferred from a targeted candidate geometry.
