"""A genuine second simulator through ordinary qualification, with authored model replies."""
import json
from pathlib import Path
import socket

import pytest

pytest.importorskip('gymnasium')  # opt-in simulator; the verification run installs the pinned extra

from telohelm import demo_cartpole as demo
from telohelm.taskpacks import cartpole as C


def test_second_task_qualifies_and_acts_without_tanker_or_network(tmp_path, monkeypatch):
    from telohelm.taskpacks.tanker.world import episode as tanker_world
    from telohelm.taskpacks.tanker.context import ContextBuilder as TankerContext

    def forbidden(*args, **kwargs):
        pytest.fail('CartPole must not construct a tanker world/context or contact a model')
    monkeypatch.setattr(tanker_world, 'Episode', forbidden)
    monkeypatch.setattr(TankerContext, '__init__', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    result = demo.run(tmp_path / 'cartpole')
    assert result['passed'], result['checks']
    assert result['real_model_calls'] == 0
    assert len(result['learning_charges']) == 2
    assert all(c['tokens_in'] is None and c['tokens_out'] is None for c in result['learning_charges'])
    rows = result['episodes']
    assert [r['seed'] for r in rows] == [11, 12, 12]
    assert [r['goal_met'] for r in rows] == [False, True, False]
    assert rows[1]['procedure_steps'] == 128
    events = [json.loads(line)['event'] for line in
              (tmp_path / 'cartpole/learning/learning-events.jsonl').read_text().splitlines()]
    trials = [e for e in events if e['kind'] == 'attempt_result']
    assert [e['passed'] for e in trials] == [False, True]
    for event in trials:
        trial = event['detail']['trial'][0]
        assert trial['component']['used_steps'] > 0
        assert Path(trial['evidence_dir'], 'result.json').is_file()
        assert trial['component']['passed'] == event['passed']
    outcome = json.loads((tmp_path / 'cartpole/later/outcome.json').read_text())
    assert len(outcome['native_steps']) == 128
    assert outcome['truncated'] and not outcome['terminated']


@pytest.mark.parametrize('action', ({'push': True}, {'push': 2}, {'push': 0.0},
                                    {'push': 1, 'goal_version': 2}, {'heading_deg': 90}))
def test_bad_action_is_refused_and_last_push_held(action):
    episode = C.Episode(C.Scenario('bad-action', 11))
    episode.reset()
    try:
        episode.step(action)
        assert not episode.record.actions[-1]['accepted']
        assert episode.native_steps[-1]['push_applied'] == 0
        assert episode.outcome()['refusals']
    finally:
        episode.env.close()


def test_native_failure_has_priority_at_time_limit_and_cutoff_cannot_pass():
    episode = C.Episode(C.Scenario('edge', 11))
    try:
        episode.reset()
        assert episode.outcome()['end_reason'] is None
        assert not C.component_trial('control', 'sha', episode.outcome(), [])['passed']
        episode.t = C.TARGET_STEPS
        episode.terminated = episode.truncated = True
        outcome = episode.outcome()
        assert not outcome['goal_met'] and outcome['end_reason'] == 'fell'
        with pytest.raises(RuntimeError, match='ended'):
            episode.step({})
        assert not C.component_trial('control', 'sha', outcome, [])['passed']
    finally:
        episode.env.close()


def test_verdict_requires_actual_candidate_use_on_every_step():
    outcome = {'goal_met': True, 'settled': True, 'end_reason': 'survived',
               'terminated': False, 'truncated': True, 'action_steps': 128}
    records = [{'decision': 'control', 'method': 'candidate', 'goal_version': 1,
                't': t, 'part': {'push': 1}} for t in range(128)]
    assert C.component_trial('control', 'candidate', outcome, records)['passed']
    assert not C.component_trial('control', 'another', outcome, records)['passed']
    assert not C.component_trial('control', 'candidate', outcome, records[:-1])['passed']
    refused = {**outcome, 'refusals': [{'t': 5}]}
    assert not C.component_trial('control', 'candidate', refused, records)['passed']


def test_public_context_keeps_raw_readings_and_excludes_seed_and_world():
    from telohelm.learning import render_learner_rules, learner_limits
    briefing = {'goal': {'version': 1, 'survive_steps': 128}}
    bundle = {'t': 0, **dict(zip(C.READINGS, (0.1, 0.2, 0.3, 0.4))),
              'observation_age_s': 0, 'step_duration_s': 0.02}
    ctx = C.ContextBuilder(briefing).update(bundle)
    assert set(ctx) == C.PUBLIC.context_fields
    assert {k: ctx[k] for k in C.READINGS} == {k: bundle[k] for k in C.READINGS}
    text = render_learner_rules(C.PUBLIC, learner_limits())
    assert json.loads(C.PUBLIC.learner_decision_choices) == 'control'
    for forbidden in ('heading_deg', 'goal_tanker', 'Shell', 'Orbit', 'seed'):
        assert forbidden not in text
    assert 'pole_angular_velocity' in text and 'push' in text
    assert C.CARTPOLE_PACK.environment.tools == {}


def test_existing_records_are_not_overwritten(tmp_path):
    (tmp_path / 'keep').write_text('original')
    with pytest.raises(FileExistsError):
        demo.run(tmp_path)
    assert (tmp_path / 'keep').read_text() == 'original'
