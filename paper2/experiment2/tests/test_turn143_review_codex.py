"""TURN143 exact-source witnesses with invented providers/outcomes; no world flight."""
import json
from copy import deepcopy
import pytest
import telohelm.ladder as A; import telohelm.learning as L; import paper2.experiment2.run_stream as R; import paper2.experiment2.comparators as K
import paper2.experiment2.streams as S


def test_reference_unknown_label_is_recorded_and_no_longer_selects(tmp_path, monkeypatch):
    # Until 27 October an unknown reference label left an episode out of trials and the floor (GEN141). Since then
    # the label is recorded for analysis only and Telohelm's own record selects
    # (paper2/notes/PRE-FREEZE-DESIGN-CHANGES-CLAUDE.md, item 1).
    from .test_learning import coordinator; from .test_learning import finish; from .test_learning import Scripted; from .test_learning import passing
    monkeypatch.setattr(S.phase5, 'fly', lambda *a, **kw: {'settled': False, 'goal_met': True})
    co, _ = coordinator(tmp_path, Scripted(), fly=passing, solvable=L.reference_solvable)
    finish(co, 25)
    assert all(c.solvable is None and c.eligible for c in co.completed)
    assert sum(e['kind']=='floor_breach' for e in co.events) == 1
    starts = [e for e in co.events if e['kind']=='attempt_start']
    assert starts and all(len(e['trial_set'])==min(8, e['ordinal']) for e in starts)


def test_same_logical_build_request_replays_without_second_provider_call(tmp_path, monkeypatch):
    from .test_learning import proposer_with
    p, seen = proposer_with(tmp_path, monkeypatch, count_tokens=lambda m: 10)
    p.allowance = A.ReasoningAllowance(tmp_path/'allow.jsonl')
    meta={'attempt':1,'call':1,'episode':'original'}
    first=p([('user','invented')], meta)
    again=p([('user','invented')], meta)
    assert first==again
    assert len(seen.sent)==1
    assert len(p.allowance.allocations(p.allowance.key(p.condition,p.stream,'original')))==1


def test_allocation_identity_cannot_be_relabelled(tmp_path):
    a=A.ReasoningAllowance(tmp_path/'allow.jsonl'); k=a.key('telohelm','s','e')
    a.allocate(k,'same','running','provider')
    with pytest.raises(ValueError): a.allocate(k,'same','build','preflight_refused')


def test_pause_counts_distinct_provider_requests_once():
    rows=[{'request_id':f'r{i}','kind':'reasoning_model','episode':'e','scope':'running',
           'status':'cut_off' if i==0 else 'answered','recovers':None} for i in range(10)]
    result=A.pause_due(rows+[deepcopy(rows[0])], ['e'])
    assert result['started']==10 and result['bad']==1 and result['due'] is False


def test_pause_excludes_trial_replay_setup_provider_calls():
    rows=[{'request_id':'running','kind':'reasoning_model','episode':'e','scope':'running',
           'status':'answered','recovers':None}]
    rows += [{'request_id':s,'kind':'reasoning_model','episode':'e','scope':s,'status':'cut_off','recovers':None}
             for s in ('trial','replay','setup','planting')]
    result=A.pause_due(rows,['e'])
    assert (result['started'],result['bad'],result['due'])==(1,0,False)


@pytest.mark.parametrize('collector',['telohelm','comparator'])
def test_collectors_keep_generation_cap_attempts_before_propagating(tmp_path, monkeypatch, collector):
    attempts=[{'attempt':0,'seed':123,'reason':'reference outcome not settled','solvable':False,
               'outcome':{'goal_met':False,'settled':False,'possible_goal_met':[False,True]}}]
    def capped(*a,**kw):raise S.GenerationError('invented cap',deepcopy(attempts))
    monkeypatch.setattr(S,'solvable_episode',capped)
    out=tmp_path/collector
    with pytest.raises(S.GenerationError):
        if collector=='telohelm':R.run_stream(1,1,out,stream_seed=4401,phases=(1,),empty_library=True)
        else:K.run_pilot_stream(K.Condition('invented','invented',lambda:None),4401,(1,),1,out)
    path=out/'generation.jsonl'
    assert path.exists()
    records=[json.loads(l) for l in path.read_text().splitlines()]
    assert records[-1]['episode']=='s4401-p1-e00' and records[-1]['attempts']==attempts
    assert records[-1]['accepted'] is False


def test_successful_generation_record_is_saved_before_comparator_flight(tmp_path, monkeypatch):
    out=tmp_path/'accepted'
    s=S.draw(4401,1,0); attempts=[{'attempt':0,'seed':123,'solvable':True,'outcome':{'goal_met':True,'settled':True}}]
    monkeypatch.setattr(S,'solvable_episode',lambda *a,**kw:S.Generated(s,deepcopy(attempts)))
    def stop(*a,**kw):
        records=[json.loads(l) for l in (out/'generation.jsonl').read_text().splitlines()]
        assert records==[{'episode':s.episode_id,'attempts':attempts}]
        raise RuntimeError('invented stop before any flight')
    monkeypatch.setattr(K,'run_pilot_episode',stop)
    with pytest.raises(RuntimeError,match='invented stop'):
        K.run_pilot_stream(K.Condition('invented','invented',lambda:None),4401,(1,),1,out)


def test_review_artifact_hashes_bind_exact_source_and_new_harness():
    from pathlib import Path
    import hashlib
    from paper2.experiment2.adaptive_controller.qualification import digest
    root=next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file())
    body=json.loads((next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'paper2/reproduction/runtime-turn143.json').read_text())
    saved=body.pop('body_sha256')
    assert digest(body)==saved
    import subprocess
    for group in ('reviewed_source_sha256','new_code_sha256'):
        for path, sha in body[group].items():
            if group=='reviewed_source_sha256' and (next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / '.git').exists():
                raw=subprocess.check_output(['git','show',body['after_commit']+':'+path],cwd=root)
            elif (next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / '.git').exists():
                # Historical new-code pins use their publishing commit.
                # Current relocated behaviour is checked separately; no qualification is transferred.
                raw=subprocess.check_output(['git','show','ae5c27b:'+path],cwd=root)
            else:
                raw=(root/path).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==sha
    assert body['scientific_flights']==0 and body['reserved_seeds_generated']==[]
    assert body['actual_guidance_profile_executed'] is False
