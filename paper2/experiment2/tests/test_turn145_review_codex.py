"""Exact repair review and recovery/pause invariants with invented providers only."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
import telohelm.ladder as A


def test_changed_messages_are_refused_without_another_inference(tmp_path, monkeypatch):
    from .test_learning import proposer_with
    p, seen=proposer_with(tmp_path,monkeypatch,count_tokens=lambda m:10)
    p.allowance=A.ReasoningAllowance(tmp_path/'allow.jsonl')
    meta={'attempt':1,'call':1,'episode':'e'}
    p([('user','original')],meta)
    with pytest.raises(ValueError):p([('user','changed')],meta)
    assert len(seen.sent)==len(p.ledger.calls)==1


def test_fresh_proposer_recovers_original_charge_once(tmp_path, monkeypatch):
    from .test_learning import proposer_with
    p, seen=proposer_with(tmp_path,monkeypatch,count_tokens=lambda m:10)
    path=tmp_path/'allow.jsonl'; p.allowance=A.ReasoningAllowance(path)
    meta={'attempt':1,'call':1,'episode':'e'};messages=[('user','original')]
    first=p(messages,meta);original=p.ledger.calls[0]
    q, later=proposer_with(tmp_path,monkeypatch,count_tokens=lambda m:10)
    q.allowance=A.ReasoningAllowance(path)
    assert q(messages,meta)==first and q(messages,meta)==first and later.sent==[]
    assert q.allowance.remaining(q.allowance.key(q.condition,q.stream,'e'))==3
    roots=[c for c in q.ledger.calls if c.recovers is None]
    assert len(roots)==1
    assert (roots[0].request_id,roots[0].tokens_in,roots[0].tokens_out,roots[0].wall_s,roots[0].scope)==(
        original.request_id,original.tokens_in,original.tokens_out,original.wall_s,original.scope)


def test_cached_answer_cannot_cross_configuration_identity(tmp_path,monkeypatch):
    from .test_learning import proposer_with
    p, seen=proposer_with(tmp_path,monkeypatch,count_tokens=lambda m:10)
    p.allowance=A.ReasoningAllowance(tmp_path/'allow.jsonl')
    meta={'attempt':1,'call':1,'episode':'e'};messages=[('user','original')]
    p(messages,meta)
    q, later=proposer_with(tmp_path,monkeypatch,count_tokens=lambda m:10)
    q.condition='changed-condition'; q.max_output_tokens=123
    with pytest.raises(ValueError):q(messages,meta)
    assert later.sent==[]


def test_conflicting_duplicate_pause_rows_are_refused_independent_of_order():
    rows=[{'request_id':'same','episode':'e','scope':'running','kind':'reasoning_model','recovers':None,
           'status':'answered'}, {'request_id':'same','episode':'e','scope':'running','kind':'reasoning_model',
           'recovers':None,'status':'cut_off'}]
    for ordered in (rows,list(reversed(rows))):
        with pytest.raises(ValueError):A.pause_due(ordered,['e'])


def test_pause_missing_purpose_is_explicitly_unplaced():
    row={'request_id':'unknown-purpose','episode':'e','kind':'reasoning_model','recovers':None,'status':'cut_off'}
    result=A.pause_due([row],['e'])
    assert result['started']==0 and result['unplaced_requests']==['unknown-purpose']
    assert result['fraction'] is None


def test_estimator_history_grows_and_is_transactionally_copied(monkeypatch):
    import math
    import paper2.experiment2.adaptive_controller.estimation as E; import paper2.experiment2.adaptive_controller.qualification as Q
    w=Q.public_workloads()[0]; est=E.EpisodeEstimator();est.reset(w['briefing'])
    est.update(w['bundle']); sizes=[];original=E.deepcopy
    def recording(x):
        sizes.append(sum(len(b.observations) for b in x.values()))
        return original(x)
    monkeypatch.setattr(E,'deepcopy',recording)
    for t in range(1,61):
        b=deepcopy(w['bundle']);b['t']=t;theta=(280/3600)/8*t
        b['reports']=[{'callsign':'A','x':65+8*math.sin(theta),'y':47+8*math.cos(theta),
                       'measured_at':t,'received_at':t}]
        est.update(b)
    assert sizes[0]==1 and sizes[-1]==60 and len(est.targets['A'].observations)==61
    # Count/storage proof only, not an estimate of real timing or whole-episode cost.


def test_measured_campaign_retains_all_fixed_keys_costs_and_actual_exposures():
    import hashlib
    import paper2.experiment2.adaptive_controller.qualification as Q
    root=next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file())
    body=json.loads((next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'paper2/controller/feasibility-turn145.json').read_text())
    saved=body.pop('body_sha256')
    assert Q.digest(body)==saved
    manifest=json.loads((next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'paper2/controller/stateful-timing-inputs.json').read_text())
    sequences=manifest['sequences']; rows=body['rows']
    expected={(family,c['id'],w['id'],rep) for family,ws in [('first',Q.public_workloads()),('stateful',sequences)]
              for rep in range(3) for c in Q.policy_grid() for w in ws}
    keys=[(r['family'],r['cell'],r['workload'],r['repetition']) for r in rows]
    assert len(rows)==len(set(keys))==576 and set(keys)==expected
    assert all(r['error'] is None and type(r['cpu_ns']) is int and r['cpu_ns']>=0 for r in rows)
    maxima={c['id']:max(r['cpu_ns'] for r in rows if r['cell']==c['id']) for c in Q.policy_grid()}
    projected=sum(maxima.values())*20*3600/1e9/3600
    assert body['projection']['projected_cpu_h']==pytest.approx(projected)
    assert projected==pytest.approx(59.30446) and body['projection']['disposition']=='next-review-only; no flight clearance'
    assert body['projection']['actual_timed_cpu_ns']==sum(r['cpu_ns'] for r in rows)
    assert body['projection']['actual_stateful_setup_cpu_ns']==sum(r['setup_cpu_ns'] for r in rows[144:])
    assert body['process_cpu_ns_inclusive_campaign_and_recording'] >= sum(r['cpu_ns'] for r in rows)+sum(r['setup_cpu_ns'] for r in rows[144:])
    assert all(r['trace'][-1]['t']==next(s for s in sequences if s['id']==r['workload'])['bundles'][-1]['t'] for r in rows[144:])
    # Actual reported public actions/traces preserved; short history does not prove complete-episode cost.
