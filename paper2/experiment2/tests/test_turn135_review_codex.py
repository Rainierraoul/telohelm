"""TURN135 independent invented accounting controls; no inference or scientific data."""
import json
from dataclasses import asdict
import pytest
import paper2.experiment2.accounting as A
import telohelm.runtime.models as M
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.trusted_log import TrustedLog
from paper2.telohelm.tests.test_turn65_review_codex import MSG


def row(id, purpose='running', step=0, **kwargs):
    return A.Row(id,'telohelm','e1',purpose,'model:reasoning_model',step,
                 measured=('wall_s','tokens_in','tokens_out'),**kwargs)


def test_transitive_recovery_first_tokens_and_unknown_original_time_only():
    rows=[row('original',wall_s=None,tokens_in=None,tokens_out=None,status='interrupted'),
          row('first',recovers='original',wall_s=.01,tokens_in=11,tokens_out=7,status='answered'),
          row('second',recovers='first',wall_s=.02,tokens_in=None,tokens_out=None,status='answered')]
    c=A.cut(rows,1)
    assert len(c['counted'])==1 and len(c['carriers'])==2
    v=A.resource_vector(c['counted']+c['carriers'])
    assert v['models']['reasoning_model']['calls']==1
    assert v['models']['reasoning_model']['tokens_in']=={'total':11.,'known_subtotal':11.,'unknown':0}
    assert A.total(rows,'wall_s')=={'total':None,'known_subtotal':0.,'unknown':1}
    assert A.cut(rows,0)['carriers']==[]


def test_missing_learning_cost_is_unavailable_not_free_and_d1_selects_existing_shares(tmp_path):
    assert 'proposer calls (no learning-ledger.jsonl)' in A.read_learning(tmp_path)['unmeasured']
    build=row('build',purpose='build',tokens_in=12,wall_s=3.)
    c=A.cut([build],2)
    assert c['counted']==[] and c['other_purpose']==[build]
    d=A.stream_allocation(['e0','e1','e2','e3'],dict(e0=1.,e1=2.,e2=3.,e3=4.),[build],'tokens_in')
    assert d['actual']==[1.,14.,3.,4.] and d['amortised']==[1.,6.,7.,8.]
    assert A.matched_subset(d['amortised'],[0,2])['total']==8.
    unknown=row('unknown',purpose='build',tokens_in=None)
    assert A.stream_allocation(['e0','e1'],dict(e0=1.,e1=2.),[unknown],'tokens_in')['available'] is False


def test_prospectively_saved_build_ledger_reader_carries_identity_and_unknown_usage(tmp_path):
    data={'request_id':'invented','kind':'reasoning_model','episode':'e1','scope':'build','action_step':0,
          'status':'timed_out','wall_s':None,'tokens_in':9,'tokens_out':None}
    (tmp_path/'learning-ledger.jsonl').write_text(json.dumps(data)+'\n')
    r=A.read_learning(tmp_path)
    assert (r['rows'][0].episode,r['rows'][0].purpose)==('e1','build')
    assert A.total(r['rows'],'tokens_in')['total']==9
    assert A.total(r['rows'],'tokens_out')['total'] is None
    assert r['unmeasured']


def test_gateway_build_is_preserved_through_normal_recovery(tmp_path):
    ledger=Ledger(); provider=M.ScriptedProvider(lambda _:M.ModelResponse('answered','invented',11,7))
    def gateway(name):
        d=tmp_path/name;d.mkdir()
        return M.ModelGateway('e1','telohelm','s',provider,TrustedLog(d/'log',d/'heads','invented'),
                              ledger,tmp_path/'journal',purpose='build')
    first=gateway('first'); first.submit('reasoning_model',MSG,256,1,0,0,0.)
    second=gateway('second'); second.submit('reasoning_model',MSG,256,1,0,0,0.)
    assert provider.calls==1 and sum(c.tokens_in or 0 for c in ledger.calls)==11
    assert all(c.scope=='build' for c in ledger.calls)


def test_whole_unsettled_and_historical_unavailable_are_separate():
    scheduled=['flagged','old','censored','failed','never']
    records={'flagged':{'complete':True,'outcome':{'settled':False,'goal_met':False,
                     'goal_met_options':[False],'alternatives':[{'goal_met_options':[True]}]}},
             'old':{'complete':True,'goal_met':True},'censored':{'complete':False,'outcome':{}},
             'failed':{'status':'policy_failed','complete':False,'outcome':None}}
    d=A.denominators(scheduled,records)
    assert d['goals']=={'settled_met':0,'settled_not_met':0,'unsettled':1,'nominal_met':0,'unavailable':1}
    assert d['counts']['censored']==d['counts']['policy_failed']==d['counts']['not_attempted']==1

def test_recovery_cannot_reclassify_original_build_as_running(tmp_path):
    provider=M.ScriptedProvider(lambda _:M.ModelResponse('answered','invented original build',11,7))
    def gateway(name,purpose):
        d=tmp_path/name; d.mkdir()
        ledger=Ledger()
        return M.ModelGateway('e1','telohelm','s',provider,TrustedLog(d/'log',d/'heads','invented'),
                              ledger,tmp_path/'journal',purpose=purpose),ledger
    first,original=gateway('first','build')
    first.submit('reasoning_model',MSG,256,1,0,0,0.)
    assert original.calls[0].scope=='build'
    resumed,later=gateway('resumed','running')
    try:
        out=resumed.submit('reasoning_model',MSG,256,1,0,0,0.)
    except M.GatewayError:
        assert provider.calls==1 and later.calls==[]
        return
    assert provider.calls==1
    assert (later.calls and all(c.scope=='build' for c in later.calls)
            and sum(c.tokens_in or 0 for c in later.calls)==11) or (out.get('status')=='refused' and later.calls==[])
