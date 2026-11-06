"""Invented CPU/loopback MVP review. No real model requests."""
import json
from types import SimpleNamespace
import pytest
import telohelm.local_model as LM; import paper2.experiment2.run_stream as R
from telohelm.runtime.models import ModelResponse
import telohelm.taskpacks.tanker.world.episode as E; import telohelm.taskpacks.tanker.world.phase5 as phase5
from paper2.telohelm.tests.test_local_adapter import ok_body
from paper2.telohelm.tests.test_transport import Invented; from paper2.telohelm.tests.test_transport import http
from .test_turn114_review_codex import events; from .test_turn114_review_codex import independently_check_episode


def test_default_world_end_and_censored_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(E, 'HORIZON', 4)
    full=R.run_stream(1,3,tmp_path/'full'); prefix=R.run_stream(1,3,tmp_path/'prefix',max_steps=2)
    assert full['summary']['completed']==1 and full['summary']['censored_by_step_limit']==0
    assert full['episodes'][0]['ended_at']==4 and not full['episodes'][0]['censored_by_step_limit']
    assert prefix['summary']['completed']==0 and prefix['summary']['censored_by_step_limit']==1
    assert prefix['episodes'][0]['end_reason'] is None and prefix['episodes'][0]['censored_by_step_limit']
    assert prefix['summary']['goal_met_of_completed']==prefix['summary']['joined_of_completed']==0


def test_mixed_usage_and_empty_zero():
    calls=[SimpleNamespace(tokens_in=11,tokens_out=None),SimpleNamespace(tokens_in=None,tokens_out=4),
           SimpleNamespace(tokens_in=3,tokens_out=2)]
    assert R._usage(calls)=={'tokens_in':None,'tokens_in_known_subtotal':14,'tokens_in_unknown_calls':1,
                            'tokens_out':None,'tokens_out_known_subtotal':6,'tokens_out_unknown_calls':1}
    assert R._usage([])['tokens_in']==R._usage([])['tokens_out']==0
    assert R._total([{'tokens_in':11},{'tokens_in':None}],'tokens_in') is None


def test_local_loopback_calls_and_independent_replay(tmp_path,monkeypatch):
    monkeypatch.setattr(R.seed,'stock',lambda library:[])
    body=ok_body(text='{"heading_deg":45}'); body['model']='invented-local'
    srv=Invented(lambda req:[(http(json.dumps(body).encode()),0)])
    try:
        done=R.run_stream(1,1,tmp_path/'out',max_steps=8,
                          local={'base_url':f'http://127.0.0.1:{srv.port}','model':'invented-local'})
    finally: srv.close()
    r=done['episodes'][0]; folder=tmp_path/'out'/r['episode']
    independently_check_episode(phase5.draw(1).world_a,r,folder)
    assert len(srv.requests)==r['model_calls']>0
    assert r['tokens_in']==11*len(srv.requests) and r['tokens_out']==4*len(srv.requests)
    assert all(e['reportable'] is False for e in events(folder,'gateway') if e['kind']=='model_end')


# Fixed in TURN 117 (Claude): MVP116.PARSE: greedy braces reject first complete JSON with trailing braces
@pytest.mark.parametrize('reply',['{"heading_deg":45} {"heading_deg":90}','{"heading_deg":45} trailing }'])
def test_first_complete_object(reply):
    assert LM.parse('heading',reply)=={'heading_deg':45}


# Fixed in TURN 117 (Claude): MVP116.UNKNOWN.COUNT: output-only unknown usage omitted from stream count
def test_output_only_unknown_count(tmp_path,monkeypatch):
    monkeypatch.setattr(R.seed,'stock',lambda library:[])
    monkeypatch.setattr(R.seed,'stand_in_model',lambda request:ModelResponse('answered','{"heading_deg":45}',
                                                                         tokens_in=11,tokens_out=None))
    done=R.run_stream(1,1,tmp_path/'out',max_steps=5); r=done['episodes'][0]; s=done['summary']
    assert r['model_calls']>0 and r['tokens_in_unknown_calls']==0 and r['tokens_out_unknown_calls']>0
    assert s['tokens_in']==11*r['model_calls'] and s['tokens_out'] is None
    assert s['calls_with_unknown_usage']==r['model_calls']
