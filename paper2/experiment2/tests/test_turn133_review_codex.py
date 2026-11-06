"""TURN133 exact accounting review. Invented CPU records/providers only; strict unmet requirements retained."""
from dataclasses import asdict
import json
import pytest
import paper2.experiment2.accounting as A; import paper2.experiment2.run_stream as R; import paper2.experiment2.mvp_seed as mvp_seed
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.library import Library
import telohelm.runtime.models as M
from paper2.telohelm.tests.test_turn65_review_codex import gateway; from paper2.telohelm.tests.test_turn65_review_codex import submit
from .test_comparators import scenario
import telohelm.taskpacks.tanker.world.physics as P


def rows(ledger):
    # Same model-row mapping as read_telohelm: immutable original ledger metadata, no guessed usage.
    return [A.Row(c.request_id,c.condition,c.episode,c.scope,'model:'+c.kind,c.action_step,
                  tokens_in=c.tokens_in,tokens_out=c.tokens_out,wall_s=c.wall_s,status=c.status,
                  answered_step=c.answered_step,recovers=c.recovers,
                  measured=('wall_s','tokens_in','tokens_out')) for c in ledger.calls]


@pytest.mark.parametrize('replay',[False,True])
def test_actual_original_interruption_first_token_carrier_is_counted_once(gateway,tmp_path,replay):
    ledger=Ledger(); journal=tmp_path/'journal'; ticks=[0]
    def clock():
        ticks[0]+=1
        if ticks[0]==2: raise KeyboardInterrupt('invented end-clock stop')
        return 4.
    first,_,_,_=gateway(journal=journal,ledger=ledger,clock=clock,replay=replay)
    with pytest.raises(KeyboardInterrupt): submit(first)
    for _ in range(2):
        resumed,_,provider,_=gateway(journal=journal,ledger=ledger,replay=replay)
        assert submit(resumed)['recovered'] and provider.calls==0
    assert sum(c.tokens_in or 0 for c in ledger.calls)==7
    assert sum(c.tokens_out or 0 for c in ledger.calls)==2
    r=rows(ledger); c=A.cut(r,12)
    assert len(c['counted'])==1 and len(c['carriers'])==2
    assert A.total(c['counted']+c['carriers'],'wall_s')['total'] is None
    v=A.resource_vector(c['counted']+c['carriers'])
    assert v['models']['small_model']['calls']==1
    assert v['models']['small_model']['tokens_in']['known_subtotal']==7
    assert v['models']['small_model']['tokens_out']['known_subtotal']==2
    assert v['models']['small_model']['tokens_in']['total']==7
    assert v['models']['small_model']['tokens_out']['total']==2


def test_completed_original_has_full_time_once_and_no_second_inference(gateway,tmp_path):
    ledger=Ledger(); journal=tmp_path/'journal'; clock=iter([4.,9.])
    first,_,_,_=gateway(journal=journal,ledger=ledger,clock=lambda:next(clock))
    submit(first)
    resumed,_,provider,_=gateway(journal=journal,ledger=ledger)
    assert submit(resumed)['recovered'] and provider.calls==0
    rr=rows(ledger); v=A.resource_vector(rr)
    assert A.total(rr,'wall_s')['total']==5.
    assert v['models']['small_model']['calls']==1
    assert v['models']['small_model']['tokens_in']['total']==7


def telo(tmp_path):
    lib=Library(tmp_path/'library'); mvp_seed.stock(lib)
    sc=scenario(deadline=2)
    done=R.run_episode(sc,lib,tmp_path/'out','invented',2,{'small_model':2.,'reasoning_model':60.})
    return sc,done,A.read_telohelm(tmp_path/'out'/sc.episode_id)


def test_actual_flagged_native_ending_is_not_classified_as_settled(tmp_path,monkeypatch):
    original=P.join_assessment; calls=[0]
    def assessment(*args,**kwargs):
        d,a=original(*args,**kwargs); calls[0]+=1
        return ('flagged' if calls[0]==1 else d),a
    monkeypatch.setattr(P,'join_assessment',assessment)
    sc,done,read=telo(tmp_path)
    assert done['attribution']['complete']
    assert 'alternative 0' in done['attribution']['recorded_endings']
    d=A.denominators([sc.episode_id],{sc.episode_id:read['summary']})
    assert d['goals']['unsettled']==1
    assert d['goals']['settled_met']==d['goals']['settled_not_met']==0
    assert read['summary']['outcome']['alternatives'][0]['action_steps']==1


def test_actual_proposer_cost_is_build_and_not_before_earlier_flight_cut(tmp_path,monkeypatch):
    from .test_learning import proposer_with
    p,seen=proposer_with(tmp_path,monkeypatch)
    p([('user','invented rewrite')],{'attempt':1,'episode':'completed','call':1})
    assert len(p.ledger.calls)==1
    output=tmp_path/'learning-ledger.jsonl'; R._write_ledger(output,p.ledger.calls)
    saved=json.loads(output.read_text())
    assert saved['scope']=='build'


def test_fixed_reader_keeps_full_native_outcome_and_world_policy_costs_separable(tmp_path,monkeypatch):
    import paper2.experiment2.comparators as K
    original=P.join_assessment
    monkeypatch.setattr(P,'join_assessment',lambda *a,**k:('flagged',original(*a,**k)[1]))
    raw=K.run_pilot_episode(scenario(deadline=2),K.conditions()['calculator'],tmp_path/'fixed','invented')
    read=A.read_fixed(tmp_path/'fixed')
    assert read['summary']['outcome']==raw['outcome'] and not raw['settled']
    d=A.denominators([raw['episode']],{raw['episode']:read['summary']})
    assert d['goals']['unsettled']==1
    policy=[r for r in read['rows'] if r.process=='policy']
    world=[r for r in read['rows'] if r.process=='world']
    assert A.total(policy,'cpu_s')['total']==pytest.approx(raw['costs']['policy_cpu_ns']/1e9)
    assert A.total(world,'cpu_s')['total']==pytest.approx(raw['costs']['world_cpu_ns']/1e9)
    assert len(A.cut(read['rows'],0)['counted'])==1


def test_pool_counts_d1_and_missing_schedule_remain_explicit():
    one=A.method_seconds('calculator',{'calculator':1,'holding_none':1})
    three=A.method_seconds('calculator',{'calculator':1,'holding_none':5})
    pooled=A.pool_method_seconds([one,three])
    assert pooled['fractions']['calculator']==.25
    assert pooled['fractions']['calculator']!=(.5+1/6)/2
    d=A.denominators(['failed','never'],{'failed':{'status':'policy_failed','complete':False,'outcome':None}})
    assert d['counts']['policy_failed']==d['counts']['not_attempted']==1
    allocated=A.allocate_learning([1.,2.,0.,0.],{1:12.},attempted=2)
    assert allocated['observed']==[True,True,False,False]
    assert allocated['amortised']==[1.,6.,4.,4.] and allocated['raw_total']==15.
