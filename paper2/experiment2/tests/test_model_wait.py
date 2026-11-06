"""Explicit model waiting with authored replies and short control-flow flights; no real model or learned candidate."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

import telohelm.local_model as LM; import paper2.experiment2.run_stream as RS
import telohelm.runtime.episode_loop as L; import telohelm.runtime.models as M
from telohelm.runtime.delivery import DelayedModels; from telohelm.runtime.delivery import Pending
from telohelm.runtime.escalation import FlightLadder
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.library import Library
from telohelm.runtime.trusted_log import TrustedLog
from telohelm.contracts.decisions import HelpEvidence
from telohelm.contracts.evidence import EvidenceView
from paper2.telohelm.tests.test_episode_loop import scenario
from paper2.telohelm.tests.test_delivery import joining_scenario
import telohelm.taskpacks.tanker.world.episode as E


def pending(text='{}', decision='report', version=1, status='answered'):
    return Pending(1, decision, 'small_model', 'invented-request', version, 0, 0, 2, 'invented-sha', status,
                   text, False)


def layer(parse=LM.parse_with_wait, wait=('report',)):
    return DelayedModels(None, None, None, {}, parse, wait_decisions=frozenset(wait))


def test_parser_keeps_waiting_separate_only_in_the_explicit_path():
    assert LM.parse('report', '{}') is None
    assert LM.parse_with_wait('report', 'reasoning </think> {}') == {}
    assert layer().part_for(pending(), 1) == ({}, False, None)
    assert layer(wait=()).part_for(pending(), 1) == (None, False, 'invalid_part')
    assert layer().part_for(pending(decision='heading'), 1) == (None, False, 'invalid_part')
    assert layer(parse=lambda d, text: None).part_for(pending(), 1) == (None, False, None)


@pytest.mark.parametrize('text,refusal', [('bad', 'unparseable'), ('{"unknown":1}', 'invalid_part'),
                                         ('{"report":NaN}', 'invalid_part')])
def test_waiting_does_not_accept_malformed_or_undeclared_fields(text, refusal):
    assert layer().part_for(pending(text=text), 1) == (None, False, refusal)


def test_old_goals_and_failed_calls_do_not_become_waits():
    assert layer(parse=lambda *a: pytest.fail('old reply must not be parsed')).part_for(
        pending(version=1), 2) == (None, True, 'out_of_date')
    assert layer().part_for(pending(status='failed'), 1) == (None, False, None)
    with pytest.raises(ValueError):
        DelayedModels(None, None, None, {}, LM.parse_with_wait, wait_decisions='report')


def fly(tmp_path, monkeypatch, *, latency=0, steps=5, replies=None, check=True, base=None, view=None,
        first_trigger=False, no_reasoning=False, stale=False, latched=False):
    base = base or scenario()
    ledger = Ledger()
    asked, checked = [], []
    answers = iter(replies) if replies is not None else None
    def answer(req):
        body = json.loads(req.messages[-1][1]); asked.append((body['t'], req.role, body['goal_version']))
        return M.ModelResponse('answered', next(answers) if answers is not None else '{}', 10, 1)
    gw = M.ModelGateway(base.episode_id, 'fixture', 'wait', M.ScriptedProvider(answer),
                        TrustedLog(tmp_path/'gateway.log', tmp_path/'gateway.heads', 'fixture'), ledger,
                        tmp_path/'journal')
    ladder = FlightLadder(gw, help_evidence=HelpEvidence('always_missing', 10, 'old fixture evidence') if stale
                          else RS.TANKER_PACK.help)
    if no_reasoning: ladder.no_reasoning.add(('report', 1))
    if latched: ladder.latched.add(('report', 1))
    dm = DelayedModels(gw, ladder.role_for,
                       lambda d, c, r: [('user', json.dumps({'t':c['t'], 'goal_version':c['goal_version']}))],
                       {'small_model':latency, 'reasoning_model':latency}, LM.parse_with_wait,
                       wait_decisions={'report'})
    def choose(d, ctx, library, **kwargs):
        if d == 'heading':
            return {'use':'authored-heading', 'output':{'heading_deg':ctx['heading_deg']}, 'tried':[], 'checks':[]}
        if first_trigger and ctx['t'] == 0:
            return {'use':'authored-trigger', 'output':{'report_trigger_nm':20}, 'tried':[], 'checks':[]}
        return {'climb':'missing_tool', 'tried':[]}
    def judge(d, ctx, out, library, **kwargs):
        checked.append((ctx['t'], dict(out)))
        return {'accepted':check, 'checks':[{'check':'authored-stand-in', 'accepted':check}]}
    monkeypatch.setattr(L, 'choose', choose)
    monkeypatch.setattr(L, 'check_output', judge)
    result = L.run_episode(base, Library(tmp_path/'library'), dm,
                           TrustedLog(tmp_path/'episode.log', tmp_path/'episode.heads', 'fixture'), ledger,
                           max_steps=steps, ladder=ladder, watch_view=view, own_reports=True)
    records = [json.loads(s)['payload'] for s in (tmp_path/'episode.log').read_text().splitlines()]
    return result, records, asked, checked


@pytest.mark.parametrize('latency', [0, 2])
def test_a_wait_is_checked_attributed_and_reconsidered_on_the_next_tick(tmp_path, monkeypatch, latency):
    result, log, asked, checked = fly(tmp_path, monkeypatch, latency=latency, no_reasoning=True)
    decisions = [p for p in log if p['kind']=='decision' and p['decision']=='report' and p['source']=='small_model']
    assert decisions and decisions[0]['t'] == latency
    assert decisions[0]['part'] == {} and decisions[0]['method'].startswith('model:')
    assert decisions[0]['reply_checks']['accepted'] is True and checked[0] == (latency, {})
    assert [a[0] for a in asked[:2]] == [0, latency + 1]  # never another request in the delivery second
    assert not any(e['kind']=='help_asked' for e in result['ladder'])
    assert not any('report' in a or 'report_trigger_nm' in a for a in result['actions'])
    assert not any(p['kind']=='decision' and p['decision']=='report' and p['source']=='procedure' for p in log)


def test_an_accepted_wait_releases_the_ladder_but_not_the_goal(tmp_path, monkeypatch):
    result, _, asked, _ = fly(tmp_path, monkeypatch, latched=True)
    assert asked[0][1] == 'reasoning_model' and asked[1][1] == 'small_model'
    assert any(e['kind']=='released' and e['decision']=='report' for e in result['ladder'])
    assert result['outcome']['goal_met'] is False and result['outcome']['reports'] == []


def test_a_check_rejecting_waiting_escalates_it_normally(tmp_path, monkeypatch):
    result, log, asked, checked = fly(tmp_path, monkeypatch, check=False)
    assert checked[0] == (0, {}) and asked[1][1] == 'reasoning_model'
    first = next(p for p in log if p['kind']=='decision' and p['decision']=='report')
    assert first['part'] is None and first['source']=='none'
    assert first['reply_checks']['accepted'] is False
    assert any(e.get('why')=='check_rejected' for e in result['ladder'])


def test_waiting_does_not_complete_the_report_or_suppress_later_reporting(tmp_path, monkeypatch):
    text = '{"report":"Wedgetail, aircraft 2, 20 NM from SHELL 1"}'
    result, log, asked, _ = fly(tmp_path, monkeypatch, replies=['{}', text])
    assert [a[0] for a in asked] == [0, 1]
    assert len(result['outcome']['reports']) == 1  # syntax accepted; no correctness claim for this authored report
    assert len([p for p in log if p['kind']=='decision' and p['decision']=='report']) == 2


def test_goal_change_invalidates_a_pending_wait_before_the_new_decision(tmp_path, monkeypatch):
    base = replace(scenario(), goals=((0,E.Goal(1,'SHELL 1',20)), (2,E.Goal(2,'SHELL 1',15))))
    _, log, asked, _ = fly(tmp_path, monkeypatch, base=base, latency=2)
    delivered = next(p for p in log if p['kind']=='model_reply')
    assert delivered['stale'] and delivered['refusal']=='out_of_date' and delivered['part'] is None
    assert asked[1] == (2, 'small_model', 2)


def test_a_wait_due_after_the_ending_remains_undelivered_and_charged(tmp_path, monkeypatch):
    result, log, asked, checked = fly(tmp_path, monkeypatch, base=joining_scenario(), latency=2)
    assert result['outcome']['end_reason']=='join' and len(asked)==1 and checked==[]
    assert not any(p['kind']=='model_reply' for p in log)
    assert any(p['kind']=='model_undelivered' for p in log)


def test_waiting_cannot_clear_disagreement_or_block_withdrawing_a_trigger(tmp_path, monkeypatch):
    view = EvidenceView(lambda briefing, ctx: 2 if ctx['t']>=1 else 0, 1, 'NM', {'report'},
                        {'report':{'report_trigger_nm':None}}, 'Authored time-based fixture, no disturbance inference.')
    result, log, asked, _ = fly(tmp_path, monkeypatch, view=view, first_trigger=True)
    assert result['actions'][0]['report_trigger_nm']==20
    assert result['actions'][1]['report_trigger_nm'] is None
    incident = result['disagreement']['incidents'][0]
    assert incident['closed_at'] is None and incident['withdrawals']['report']['state']=='withdrawn'
    assert any(p['kind']=='decision' and p.get('withheld') and p['part']=={} for p in log)


def test_a_wait_asked_before_an_incident_is_set_aside(tmp_path, monkeypatch):
    view = EvidenceView(lambda briefing, ctx: 2 if ctx['t']>=1 else 0, 1, 'NM', {'report'}, {}, 'Authored fixture.')
    _, log, asked, _ = fly(tmp_path, monkeypatch, view=view, latency=2)
    delivery = next(p for p in log if p['kind']=='model_reply')
    assert delivery['part']=={}
    decision = next(p for p in log if p['kind']=='decision' and p['decision']=='report' and p['t']==2)
    assert decision['reply_checks']['refused']=='asked_before_incident' and decision['part'] is None
    assert asked[1][0]==2


def test_waiting_does_not_suppress_stale_evidence_help(tmp_path, monkeypatch):
    base=replace(scenario(), goals=((0,E.Goal(1,'SHELL 1',20,deadline=100)),))
    result, _, _, _ = fly(tmp_path, monkeypatch, base=base, stale=True, steps=1)
    assert any(e['kind']=='help_asked' and e['reason']=='missing_or_stale_evidence' for e in result['ladder'])


def test_runner_opt_in_is_recorded_and_the_default_is_unchanged(tmp_path):
    baseline=RS.run_stream(1, 1, tmp_path/'old', max_steps=3, empty_library=True)
    enabled=RS.run_stream(1, 1, tmp_path/'new', max_steps=3, empty_library=True, model_wait=True)
    assert 'model_wait' not in baseline['summary']['flight']
    assert enabled['summary']['flight']['model_wait'] is True
