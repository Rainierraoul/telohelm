"""Independent invented TURN129 controls. No scientific development/scored episodes or model calls."""
from copy import deepcopy
import hashlib
import json
import math
import shutil

import pytest

import paper2.experiment2.comparators as K
from telohelm.runtime.trusted_log import verify
from .test_comparators import condition; from .test_comparators import log; from .test_comparators import scenario; from .test_comparators import summary
import telohelm.taskpacks.tanker.world.episode as E; import telohelm.taskpacks.tanker.world.physics as P


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def test_original_held_identity_and_counts_are_recoverable_without_policy_execution(tmp_path):
    goals = ((0, E.Goal(1, 'SHELL 1', 20., None)), (5, E.Goal(2, 'SHELL 1', 25., 10)))
    script = {1: {'goal_version': 1, 'heading_deg': 45.},
              3: {'goal_version': 2, 'heading_deg': 90.},
              4: {'goal_version': 1, 'heading_deg': 400., 'report': 'invented atomic refusal'},
              6: {'goal_version': 2, 'heading_deg': 90.}, 7: None}
    out = tmp_path / 'original'
    K.run_pilot_episode(scenario(goals=goals), condition(script=script), out, 'invented')
    recovered = tmp_path / 'recovered'
    shutil.copytree(out, recovered)  # recovered evidence; no policy rerun/resume claim
    assert verify(recovered / 'episode.log', recovered / 'episode.heads')['ok']
    events = log(recovered)
    bundles = {e['step']: e for e in events if e['kind'] == 'bundle'}
    decisions = {e['step']: e for e in events if e['kind'] == 'decision'}
    actions = summary(recovered)['outcome']['actions']
    shares, held_by = {'invented': 0, 'held': 0}, {}
    held = None
    for event in [e for e in events if e['kind'] == 'world_step']:
        step = event['step']; d = decisions[step]; a = actions[step]
        assert d['input_sha256'] == canonical_hash(bundles[step]['bundle'])
        assert d['t'] == d['available_t'] == a['t'] == step
        assert d['step'] == d['available_step'] == step
        assert event['accepted'] == a['accepted'] and d['part'] == a['actions']
        if a['accepted'] and 'heading_deg' in a['actions']:
            held = d
            shares['invented'] += 1
            assert event['heading'] == {'issued': d['decision_id']}
        else:
            shares['held'] += 1
            issuer = 'initial' if held is None else held['source']
            held_by[issuer] = held_by.get(issuer, 0) + 1
            original = {'decision_id': None, 'goal_version': None} if held is None else {
                'decision_id': held['decision_id'], 'goal_version': held['goal_tag']}
            assert event['heading'] == {'held': original, 'issuer': issuer}
    s = summary(recovered)
    assert s['method_seconds'] == {'shares': shares, 'held_by_issuer': held_by, 'seconds': 10,
                                   'fractions': {'invented': .2, 'held': .8}}
    assert shares == {'invented': 2, 'held': 8} and held_by == {'initial': 1, 'invented': 7}
    assert decisions[5]['current_goal_version'] == 2 and decisions[1]['goal_tag'] == 1
    assert s['outcome']['reports'] == []  # the combined bad-heading/report action applied neither field
    world = E.Episode(scenario(goals=goals)); world.reset()
    for row in actions:
        world.step(deepcopy(row['actions']))
    assert world.outcome() == s['outcome']  # exact action replay, including refusals, current goal and ending


@pytest.mark.parametrize('mode', ['zero_cap', 'reset_failure'])
def test_no_action_seconds_have_undefined_fractions_and_correct_outcome_status(tmp_path, mode):
    out = tmp_path / mode
    if mode == 'zero_cap':
        r = K.run_pilot_episode(scenario(), condition(), out, 'invented', max_steps=0)
        assert r['outcome_provisional'] and r['censored_by_step_limit'] and not r['complete']
        assert log(out)[-1]['outcome'] == r['outcome']
    else:
        class FailReset:
            def reset(self, briefing):
                raise ValueError('invented reset failure')
        with pytest.raises(K.PolicyFailed):
            K.run_pilot_episode(scenario(), K.Condition('invented', 'invented', FailReset), out, 'invented')
        r = summary(out)
        assert r['outcome'] is None and not r['complete']
        assert not any(e['kind'] == 'episode_end' for e in log(out))
        work = [json.loads(l) for l in (out / 'work.jsonl').read_text().splitlines()]
        assert len(work) == 1 and work[0]['scope'] == 'policy_reset' and work[0]['cpu_ns'] >= 0
    assert r['method_seconds']['seconds'] == 0 and r['method_seconds']['fractions'] is None
    assert sum(r['method_seconds']['shares'].values()) == 0


def trigger_world():
    receiver = P.ReceiverState(0., 0., 0., math.pi / 2, math.pi / 2, .125, P.turn_rate(450))
    tanker = E.Tanker('SHELL 1', P.Orbit(10., 0., 1., 0., 0.))
    return E.Scenario('invented-trigger', receiver, (0., 0.), (tanker,),
                      ((0, E.Goal(1, 'SHELL 1', 100., 4)),))


@pytest.mark.parametrize('mode,report_times', [('once', [1]), ('repeat', []), ('clear', [])])
def test_trigger_interface_rearming_same_second_omission_and_clear(tmp_path, mode, report_times):
    # Deliberately broad trigger checks mechanics, not report correctness or steering quality.
    action = {'goal_version': 1, 'report_trigger_nm': 100.}
    script = {0: action}
    if mode == 'repeat':
        script = {t: action for t in range(4)}
    elif mode == 'clear':
        script[1] = {'goal_version': 1, 'report_trigger_nm': None}
    r = K.run_pilot_episode(trigger_world(), condition(script=script), tmp_path / mode, 'invented')
    assert [e['t'] for e in r['outcome']['reports']] == report_times
    assert all(e['via_trigger'] for e in r['outcome']['reports'])


def test_native_alternative_prefix_keeps_original_goal_and_action_cutoff(tmp_path, monkeypatch):
    # Force an invented numerical branch; no assertion that this trajectory is physically ambiguous.
    original = P.join_assessment
    calls = 0
    def assessment(*args, **kwargs):
        nonlocal calls
        decision, approach = original(*args, **kwargs)
        calls += 1
        return ('flagged' if calls == 1 else decision), approach
    monkeypatch.setattr(P, 'join_assessment', assessment)
    goals = ((0, E.Goal(1, 'SHELL 1', 20., None)), (2, E.Goal(2, 'SHELL 1', 25., 4)))
    out = tmp_path / 'branch'
    r = K.run_pilot_episode(scenario(goals=goals), condition(), out, 'invented')
    a = r['outcome']['alternatives'][0]
    assert a['goal']['version'] == 1 and a['action_steps'] == a['event_cutoff_t'] == 1
    assert r['outcome']['goal']['version'] == 2 and r['action_steps'] == 4
    assert not r['settled'] and not r['outcome_provisional'] and r['complete']
    assert summary(out)['outcome'] == log(out)[-1]['outcome'] == r['outcome']
    work = [json.loads(l) for l in (out / 'work.jsonl').read_text().splitlines()]
    assert [e['t'] for e in work if e['scope'] == 'policy_act' and e['t'] < a['action_steps']] == [0]
    assert len([e for e in work if e['scope'] == 'policy_act']) == 4


@pytest.mark.parametrize('ending', ['join', 'deadline'])
def test_fixed_and_telohelm_paths_share_atomic_refusal_and_ending_before_goal_event(tmp_path, ending):
    import telohelm.runtime.broker as B; import telohelm.runtime.episode_loop as L
    from telohelm.runtime.ledger import Ledger
    from telohelm.runtime.library import Library
    from telohelm.runtime.trusted_log import TrustedLog
    receiver = P.ReceiverState(0., 0., 0., math.pi / 2, math.pi / 2, .125, P.turn_rate(450))
    tanker = E.Tanker('SHELL 1', P.Orbit(.6 if ending == 'join' else 10., 0., .01, 0., 0.))
    sc = E.Scenario('invented-ending-' + ending, receiver, (0., 0.), (tanker,),
                    ((0, E.Goal(1, 'SHELL 1', 100., 3)), (3, E.Goal(2, 'SHELL 1', 20., 6))))
    class ScriptedParts:
        def decide(self, decision, ctx, reason):
            if ctx['t'] == 0:
                part = {'heading_deg': 400.} if decision == 'heading' else {'report': 'invented atomic refusal'}
            elif ctx['t'] == 1 and decision == 'report':
                part = {'report_trigger_nm': 100.}
            else:
                part = {}
            return part, None  # invented interface answers; no actual model/provider request
    ledger = Ledger()
    br = B.Broker(sc.episode_id, TrustedLog(tmp_path / 'b.log', tmp_path / 'b.heads', 'invented'), ledger,
                  lambda _: pytest.fail('this fixture does not predict'))
    telo = L.run_episode(sc, Library(tmp_path / 'library'), ScriptedParts(),
                         TrustedLog(tmp_path / 't.log', tmp_path / 't.heads', 'invented'), ledger, broker=br)
    br._log.close()
    assert verify(tmp_path / 'b.log', tmp_path / 'b.heads')['ok']
    actions = telo['actions']
    class Replay:
        def reset(self, briefing):
            self.briefing = briefing
        def act(self, bundle):
            return deepcopy(actions[bundle['t']])
    fixed = K.run_pilot_episode(sc, K.Condition('replay', 'replay', Replay), tmp_path / 'fixed', 'invented')
    inputs = B.load_inputs(tmp_path / 'b.log', tmp_path / 'b.heads', sc.episode_id)
    events = log(tmp_path / 'fixed')
    assert [e['bundle'] for e in events if e['kind'] == 'bundle'] == [e['bundle'] for e in inputs['bundles']]
    assert fixed['outcome'] == telo['outcome']
    assert fixed['end_reason'] == ending and fixed['ended_at'] == fixed['action_steps'] == 3
    assert fixed['outcome']['goal']['version'] == 1  # termination precedes the pending t=3 goal change
    assert fixed['outcome']['refusals'] == [{'t': 0, 'reason': 'heading out of range'}]
    assert [r['t'] for r in fixed['outcome']['reports']] == [2]
    assert fixed['outcome']['reports'][0]['via_trigger'] and ledger.calls == []
