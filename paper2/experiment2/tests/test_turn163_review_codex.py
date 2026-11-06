"""Fixed invented short world controls and pure component records; no generated stream or provider."""
import json
import pytest
import paper2.experiment2.run_stream as RS
import telohelm.runtime.models as M
from telohelm.runtime.library import Library
from paper2.telohelm.tests.test_episode_loop import scenario
from telohelm.runtime.escalation import FlightLadder
from .test_flight_ladder import _Gateway
from telohelm.ladder import ReasoningAllowance
from telohelm.runtime.chooser import check_output
from telohelm.runtime.sandbox import SandboxResult


def fly(tmp_path, monkeypatch, heading):
    def reply(request):
        decision=json.loads(request.messages[-1][1])['decision']
        if decision=='report': text='{}'
        elif request.role=='small_model': text=json.dumps(heading)
        else: text='{"heading_deg":90}'
        return M.ModelResponse('answered',text,3,1)
    monkeypatch.setattr(RS.seed,'stand_in_model',reply)
    out=tmp_path/'run';out.mkdir()
    s=scenario()
    # The scenario is a fixed hand-constructed unit fixture, never a generator draw.
    done=RS.run_episode(s,Library(tmp_path/'library'),out,'invented-unit',5,
                        {'small_model':2.0,'reasoning_model':2.0},None,
                        {'allowance':ReasoningAllowance(tmp_path/'allowance.jsonl'),'reasoning':None})
    done['unit_ledger']=[json.loads(row) for row in (out/s.episode_id/'ledger.jsonl').read_text().splitlines()]
    return done


def test_fixed_small_parse_refusal_escalates_and_reasoning_acts(tmp_path,monkeypatch):
    def reply(request):
        decision=json.loads(request.messages[-1][1])['decision']
        text='{}' if decision=='report' else ('unparseable' if request.role=='small_model' else '{"heading_deg":90}')
        return M.ModelResponse('answered',text,3,1)
    monkeypatch.setattr(RS.seed,'stand_in_model',reply)
    out=tmp_path/'run';out.mkdir()
    result=RS.run_episode(scenario(),Library(tmp_path/'library'),out,'invented-unit',5,
                           {'small_model':2.0,'reasoning_model':2.0},None,
                           {'allowance':ReasoningAllowance(tmp_path/'allowance.jsonl'),'reasoning':None})
    assert any(e['kind']=='escalated' and e['decision']=='heading' for e in result['record']['ladder'])
    assert result['record']['share_of_seconds']['reasoning_model']>0


def test_fixed_public_heading_refusal_escalates_next_eligible_second(tmp_path,monkeypatch):
    result=fly(tmp_path,monkeypatch,{'heading_deg':360})
    assert any(e['kind']=='escalated' and e['decision']=='heading' for e in result['record']['ladder'])
    assert any(r['kind']=='reasoning_model' and r['sim_second']<=3 for r in result['unit_ledger'])


def test_reply_for_other_decision_escalates_heading(tmp_path,monkeypatch):
    result=fly(tmp_path,monkeypatch,{'report_trigger_nm':25})
    assert any(e['kind']=='escalated' and e['decision']=='heading' for e in result['record']['ladder'])
    assert any(r['kind']=='reasoning_model' and r['sim_second']<=3 for r in result['unit_ledger'])


def test_joint_allowance_leaves_only_one_build_slot_after_three_flight_slots(tmp_path):
    a=ReasoningAllowance(tmp_path/'allowance.jsonl')
    key=a.key('telohelm','invented-unit','fixed')
    gw=_Gateway();ladder=FlightLadder(gw,a,key);ladder.at(1)
    for i in range(3):
        assert ladder.may_issue('reasoning_model','heading',i)
        gw.used['reasoning_model']+=1
    assert a.attempt_budget(key)==1
    assert a.allocate(key,'attempt1/call1','build','provider')==4
    assert not ladder.may_issue('reasoning_model','heading',3)
    assert ReasoningAllowance(a.path).attempt_budget(key)==0


def test_old_goal_reply_never_escalates_new_goal():
    ladder=FlightLadder(_Gateway());ladder.at(2)
    ladder.unusable('heading','small_model',1,'unparseable',3)
    assert ladder.role_for('heading','missing_tool')=='small_model' and not ladder.events


def test_role_router_uses_only_the_requested_stub_provider():
    from types import SimpleNamespace
    seen=[]
    class Provider:
        def __init__(self,role):self.role=role
        def complete(self,request):seen.append(self.role);return self.role
    r=RS.RoleRouter({role:Provider(role) for role in ('small_model','reasoning_model')})
    assert r.complete(SimpleNamespace(role='reasoning_model'))=='reasoning_model'
    assert r.complete(SimpleNamespace(role='small_model'))=='small_model'
    assert seen==['reasoning_model','small_model'] and r.reportable is False


def test_model_reply_check_keeps_reject_verdict_and_unknown_cost(tmp_path):
    lib=Library(tmp_path/'checks')
    sha=lib.add({'kind':'check','decision':'heading','conditions':[],
                 'source':"def check(ctx,out):\n    return 'reject'\n",'origin':{'invented':True}})
    lib.set_status(sha,'adopted','unit fixture')
    result=check_output('heading',{}, {'heading_deg':90},lib,
                        run=lambda *a,**k:SandboxResult('ok','reject',cpu_seconds=None))
    assert result['accepted'] is False
    assert result['checks'][0]['check']==sha and result['checks'][0]['verdict']=='reject'
    assert result['checks'][0]['cpu_seconds'] is None
