"""Authored short flights and four saved public inputs; no real model or generated candidate execution.

The four public delivery contexts come from model-report-preconditions-codex-20261024.json; no hidden label,
true position or disturbance parameter is supplied to the runtime rule.
"""
from dataclasses import replace
import json
import math

import pytest

import telohelm.learning as G; import telohelm.local_model as LM; import paper2.experiment2.run_stream as RS
from telohelm.contracts.decisions import HelpEvidence
from telohelm.contracts.evidence import EvidenceView
from telohelm.taskpacks.tanker.public import TANKER
from telohelm.taskpacks.tanker.execution import tanker_model_precondition
from telohelm.taskpacks.tanker.pack import TANKER_PACK; from telohelm.taskpacks.tanker.pack import TANKER_REDUCED_PACK
import telohelm.runtime.broker as B; import telohelm.runtime.episode_loop as L; import telohelm.runtime.models as M
from telohelm.runtime.delivery import DelayedModels
from telohelm.runtime.escalation import FlightLadder
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.library import Library
from telohelm.runtime.model_actions import ModelActions; from telohelm.runtime.model_actions import learner_feedback
from telohelm.runtime.trusted_log import TrustedLog; from telohelm.runtime.trusted_log import verify
from paper2.telohelm.tests.test_episode_loop import scenario
import telohelm.taskpacks.tanker.world.episode as E

BAD = {"report":"Wedgetail, aircraft 2, 20 NM from SHELL 1"}
PUBLIC_DELIVERIES = [{'report_at_nm': 30.0, 'report_measured_at': 145, 'report_x': 55.297776577698066, 'report_y': 46.2261526002386, 't': 145, 'x': 11.545571564928997, 'y': 11.520906063955152, 'goal_version': 1}, {'report_at_nm': 30.0, 'report_measured_at': 316, 'report_x': 64.33173225599606, 'report_y': 56.68205731411249, 't': 316, 'x': 31.797792313944402, 'y': 22.355423284661875, 'goal_version': 1}, {'report_at_nm': 10.0, 'report_measured_at': 168, 'report_x': 78.02840102992212, 'report_y': 48.422674689626845, 't': 168, 'x': 20.092755037499646, 'y': 12.491283984212611, 'goal_version': 1}, {'report_at_nm': 15.0, 'report_measured_at': 260, 'report_x': 68.3387944818413, 'report_y': 42.300938206050986, 't': 260, 'x': 25.659481880935722, 'y': 17.508618984942984, 'goal_version': 1}]


@pytest.mark.parametrize('ctx', PUBLIC_DELIVERIES)
def test_the_four_saved_handover_reports_are_refused_from_public_delivery_inputs(ctx):
    refusal = tanker_model_precondition('report', ctx, BAD)
    assert refusal['reason']=='report_range_contradicts_current_observation'
    assert refusal['evidence']['range_nm'] > ctx['report_at_nm'] + 0.25


@pytest.mark.parametrize('change,refused', [({},False),({'report_x':50},True),
    ({'report_x':50,'report_measured_at':0},False),({'report_x':None},False),
    ({'report_x':True},False),({'report_x':float('inf')},False),
    ({'x':-1e308,'report_x':1e308},False),({'report_x':10.25},False),
    ({'report_x':10.25+5e-11},False),({'report_x':10.25+2e-10},True)])
def test_only_a_simultaneous_finite_range_contradiction_is_refused(change,refused):
    ctx=dict(t=20,report_measured_at=20,x=0,y=0,report_x=10,report_y=0,report_at_nm=10)
    assert (tanker_model_precondition('report',{**ctx,**change},BAD) is not None) is refused


@pytest.mark.parametrize('part',[{}, {'report_trigger_nm':10}, {'report_trigger_nm':None}, {'heading_deg':90}])
def test_waits_and_deferred_actions_are_not_changed_into_immediate_reports(part):
    assert tanker_model_precondition('report',PUBLIC_DELIVERIES[0],part) is None
    assert tanker_model_precondition('heading',PUBLIC_DELIVERIES[0],part) is None


def test_feedback_is_detached_current_goal_specific_and_bounded_for_learning():
    control=ModelActions(tanker_model_precondition)
    ctx=PUBLIC_DELIVERIES[0]
    for i in range(5):
        event=control.refuse('report',ctx,BAD,'small_model',f'invented-{i}')
    event['evidence']['range_nm']=-1
    given=control.context('report',dict(ctx))
    assert given['last_action_refusal']['request_id']=='invented-4'
    assert given['last_action_refusal']['evidence']['range_nm']>0
    given['last_action_refusal']['evidence']['range_nm']=-2
    assert control.events[-1]['evidence']['range_nm']>0
    assert 'last_action_refusal' not in control.context('heading',dict(ctx))
    record={'model_action_refusals':control.events,'outcome':{'goal':{'version':1}}}
    feedback=learner_feedback(record)['model_action_refusals']
    assert feedback['count']==5 and [e['request_id'] for e in feedback['latest']]==['invented-2','invented-3','invented-4']
    assert learner_feedback({**record,'outcome':{'goal':{'version':2}}})=={}
    assert 'last_action_refusal' not in control.context('report',{**ctx,'goal_version':2})
    assert len(control.events)==5


@pytest.mark.parametrize('bad',[{}, {'reason':'x','evidence':float('nan')},
    {'reason':'x','evidence':{'bad':float('nan')}}, {'reason':'x','evidence':{},'extra':1}])
def test_a_broken_host_precondition_is_not_hidden_as_a_model_error(bad):
    control=ModelActions(lambda d,c,p:bad)
    with pytest.raises(ValueError):control.refuse('report',PUBLIC_DELIVERIES[0],BAD,'small_model','id')


def flight(tmp_path,monkeypatch,*,latency=0,replies=None,base=None,pack=TANKER_PACK,view=None,
           first_trigger=False,check=True,steps=5,enabled=True,broker=True):
    base=base or scenario(); ledger=Ledger();asked=[];checked=[]
    responses=iter(replies or [BAD,{}])
    def answer(req):
        body=json.loads(req.messages[-1][1]);asked.append((req.role,body['context']))
        return M.ModelResponse('answered',json.dumps(next(responses,{})),10,1)
    gw=M.ModelGateway(base.episode_id,'fixture','controls',M.ScriptedProvider(answer),
                      TrustedLog(tmp_path/'gateway.log',tmp_path/'gateway.heads','fixture'),ledger,tmp_path/'journal')
    ladder=FlightLadder(gw,flight_limit=1,help_evidence=pack.help)
    models=DelayedModels(gw,ladder.role_for,
                         lambda d,c,r:[('user',json.dumps({'decision':d,'context':c,'reason':r}))],
                         {'small_model':latency,'reasoning_model':latency},LM.parse_with_wait,
                         may_issue=ladder.may_issue,wait_decisions={'report'})
    def choose(d,c,library,**kw):
        if d=='heading':return {'use':'authored-heading','output':{'heading_deg':c['heading_deg']},'tried':[],'checks':[]}
        if first_trigger and c['t']==0:
            return {'use':'authored-trigger','output':{'report_trigger_nm':20},'tried':[],'checks':[]}
        return {'climb':'missing_tool','tried':[]}
    def judge(d,c,p,library,**kw):
        checked.append((d,c['t'],p))
        return {'accepted':check,'checks':[]}
    monkeypatch.setattr(L,'choose',choose);monkeypatch.setattr(L,'check_output',judge)
    br=(B.Broker(base.episode_id,TrustedLog(tmp_path/'broker.log',tmp_path/'broker.heads','fixture'),ledger,
                 lambda args:None,condition='fixture',stream='controls') if broker else None)
    result=L.run_episode(base,Library(tmp_path/'library'),models,
                          TrustedLog(tmp_path/'episode.log',tmp_path/'episode.heads','fixture'),ledger,
                          max_steps=steps,ladder=ladder,broker=br,pack=pack,own_reports=True,
                          watch_view=view,model_preconditions=enabled)
    records=[json.loads(s)['payload'] for s in (tmp_path/'episode.log').read_text().splitlines()]
    return result,records,asked,checked,ledger


@pytest.mark.parametrize('latency',[0,2])
def test_refusal_never_acts_and_flows_to_the_next_model_request_and_broker(tmp_path,monkeypatch,latency):
    result,records,asked,checked,ledger=flight(tmp_path,monkeypatch,latency=latency,steps=7)
    refusal=result['model_action_refusals'][0]
    assert refusal['t']==latency and refusal['source']=='small_model'
    assert not any('report' in a for a in result['actions'])
    assert all('heading_deg' in a for a in result['actions'])
    assert asked[1][0]=='reasoning_model' and asked[1][1]['last_action_refusal']==refusal
    assert asked[1][1]['reply_delay_s']==latency
    assert not any(p==BAD for d,t,p in checked)  # precondition refusal comes before adopted checks
    decision=next(p for p in records if p['kind']=='decision' and p['decision']=='report' and p['t']==latency)
    assert decision['source']=='none' and decision['part'] is None
    events=[json.loads(s)['payload'] for s in (tmp_path/'broker.log').read_text().splitlines()]
    assert any(p['kind']=='model_action_refused' and p['request_id']==refusal['request_id'] for p in events)
    assert verify(tmp_path/'broker.log',tmp_path/'broker.heads')['ok']
    assert any(c.request_id==refusal['request_id'] for c in ledger.calls)  # refused action still costs its call
    assert len([c for c in ledger.calls if c.kind=='reasoning_model'])<=1


def test_a_correct_range_model_report_is_not_refused(tmp_path,monkeypatch):
    base=scenario(); x,y=base.tankers[0].orbit.position(0)
    target=math.hypot(x-base.receiver.x,y-base.receiver.y)
    base=replace(base,goals=((0,E.Goal(1,'SHELL 1',target)),))
    result,_,_,checked,_=flight(tmp_path,monkeypatch,base=base,replies=[BAD],steps=2)
    assert result['model_action_refusals']==[] and result['actions'][0]['report']==BAD['report']
    assert checked and checked[0][2]==BAD  # range alone does not certify this authored text or the full goal


def test_existing_checks_still_judge_a_nonrefuted_model_answer(tmp_path,monkeypatch):
    result,_,_,checked,_=flight(tmp_path,monkeypatch,replies=[{}],check=False,steps=2)
    assert checked[0][2]=={} and result['model_action_refusals']==[]
    assert any(e.get('why')=='check_rejected' for e in result['ladder'])


def test_the_callback_sees_only_the_selected_public_view(tmp_path,monkeypatch):
    pack=replace(TANKER_PACK,hidden={'report_measured_at'})
    result,_,_,_,_=flight(tmp_path,monkeypatch,pack=pack,replies=[BAD],steps=2)
    assert result['model_action_refusals']==[] and result['actions'][0]['report']==BAD['report']


def test_default_delivery_does_not_gain_the_new_gate(tmp_path,monkeypatch):
    result,_,asked,_,_=flight(tmp_path,monkeypatch,enabled=False,replies=[BAD],steps=2)
    assert 'model_action_refusals' not in result and result['actions'][0]['report']==BAD['report']
    assert 'reply_delay_s' not in asked[0][1]


def test_a_goal_change_discards_old_refusal_feedback(tmp_path,monkeypatch):
    base=replace(scenario(),goals=((0,E.Goal(1,'SHELL 1',20)),(1,E.Goal(2,'SHELL 1',15))))
    _,_,asked,_,_=flight(tmp_path,monkeypatch,base=base,replies=[BAD,{}],steps=3)
    assert asked[1][1]['goal_version']==2 and 'last_action_refusal' not in asked[1][1]


def test_an_old_goal_reply_is_discarded_before_a_precondition_can_refuse_it(tmp_path,monkeypatch):
    base=replace(scenario(),goals=((0,E.Goal(1,'SHELL 1',20)),(1,E.Goal(2,'SHELL 1',15))))
    result,log,asked,_,_=flight(tmp_path,monkeypatch,base=base,latency=2,replies=[BAD,{}],steps=5)
    reply=next(p for p in log if p['kind']=='model_reply')
    assert reply['refusal']=='out_of_date' and result['model_action_refusals']==[]
    assert asked[1][1]['goal_version']==2 and 'last_action_refusal' not in asked[1][1]


def test_a_preincident_wait_is_set_aside_without_a_spurious_refusal(tmp_path,monkeypatch):
    view=EvidenceView(lambda b,c:2 if c['t']>=1 else 0,1,'NM',{'report'},{},'Authored fixture.')
    result,log,asked,_,_=flight(tmp_path,monkeypatch,view=view,latency=2,replies=[{},{}],steps=5)
    decision=next(p for p in log if p['kind']=='decision' and p['decision']=='report' and p['t']==2)
    assert decision['reply_checks']['refused']=='asked_before_incident'
    assert result['model_action_refusals']==[] and 'last_action_refusal' not in asked[1][1]
    assert asked[1][1]['briefing_disagreement']['since_t']==1


def test_stale_evidence_help_is_not_suppressed_by_the_new_gate(tmp_path,monkeypatch):
    base=replace(scenario(),goals=((0,E.Goal(1,'SHELL 1',20,deadline=100)),))
    pack=replace(TANKER_PACK,help=HelpEvidence('always_missing',10,'Authored stale evidence.'))
    result,_,_,_,_=flight(tmp_path,monkeypatch,base=base,pack=pack,steps=1)
    assert result['model_action_refusals']
    assert any(e['kind']=='help_asked' and e['reason']=='missing_or_stale_evidence' for e in result['ladder'])


def test_a_refusal_cannot_block_withdrawing_an_old_trigger(tmp_path,monkeypatch):
    view=EvidenceView(lambda b,c:2 if c['t']>=1 else 0,1,'NM',{'report'},
                      {'report':{'report_trigger_nm':None}},'Authored fixture.')
    result,_,_,_,_=flight(tmp_path,monkeypatch,view=view,first_trigger=True)
    assert result['actions'][0]['report_trigger_nm']==20 and result['actions'][1]['report_trigger_nm'] is None
    assert result['disagreement']['incidents'][0]['withdrawals']['report']['state']=='withdrawn'
    assert result['disagreement']['incidents'][0]['closed_at'] is None


def test_refusal_feedback_reaches_the_learning_queue_without_privileged_outcomes(tmp_path):
    lib=Library(tmp_path/'library')
    co=G.Coordinator(lib,tmp_path/'learning',None,lambda s:True,None,attempt_every=5)
    event=ModelActions(tanker_model_precondition).refuse('report',PUBLIC_DELIVERIES[0],BAD,'small_model','id')
    outcome={'goal_met':False,'joined':False,'report_correct':False,'model_action_refusals':[event]}
    co.after_episode(scenario(),outcome,True)
    assert co.queue[0]['model_action_refusals']=={'count':1,'latest':[event]}
    assert 'true_range' not in json.dumps(co.queue)
    assert co.attempts==0


def test_new_report_messages_offer_existing_actions_and_public_feedback_only():
    ctx={**PUBLIC_DELIVERIES[0],'reply_delay_s':2,'last_action_refusal':{'reason':'public test'}}
    messages=LM.report_control_messages('report',ctx,'unfamiliar_but_solvable')
    system=messages[0][1]; body=json.loads(messages[1][1])
    assert TANKER.learner_goal in system and TANKER.learner_actions in system
    assert 'report_trigger_nm' in system and 'return {}' in system
    assert body['reason']=='unfamiliar_but_solvable' and body['context']['reply_delay_s']==2
    assert body['context']['last_action_refusal']=={'reason':'public test'}
    assert 'true_range' not in json.dumps(body)


@pytest.mark.parametrize('options',[{}, {'own_reports':True,'ladder':False}])
def test_mode_dependencies_fail_before_output_files(tmp_path,options):
    out=tmp_path/'absent'
    with pytest.raises(ValueError,match='needs the ladder'):
        RS.run_stream(1,1,out,max_steps=2,model_report_control=True,**options)
    assert not out.exists()


def test_episode_entry_does_not_silently_expand_the_flight_budget(tmp_path):
    lib=Library(tmp_path/'library')
    with pytest.raises(ValueError,match='one-call'):
        RS.run_episode(scenario(),lib,tmp_path/'absent','fixture',4,{'small_model':2,'reasoning_model':60},
                        flight={'model_report_control':True,'own_reports':True,'reasoning_limit':2})
    assert not (tmp_path/'absent').exists()


def test_runner_records_refusals_and_the_effective_option(tmp_path,monkeypatch):
    def answer(req):
        body=json.loads(req.messages[-1][1]);text=BAD if body['decision']=='report' else {'heading_deg':90}
        return M.ModelResponse('answered',json.dumps(text),10,1)
    monkeypatch.setattr(RS.seed,'stand_in_model',answer)
    done=RS.run_stream(1,1,tmp_path/'run',max_steps=4,empty_library=True,model_report_control=True,
                        own_reports=True,small_interval_s=0,stream_seed=4401,prepared_scenarios=(scenario(),))
    assert done['summary']['flight']['model_report_control'] and done['summary']['flight']['model_wait']
    episode=done['episodes'][0]
    assert episode['model_action_refusals'] and episode['report_correct'] is False
    saved=json.loads((tmp_path/'run'/episode['episode']/'summary.json').read_text())
    assert saved['model_action_refusals']==episode['model_action_refusals']
