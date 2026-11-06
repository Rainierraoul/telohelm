"""MVP validity review: independent event recount and world replay, not hardening.

Short invented development runs only; no empirical/scored evidence or tuning.
"""
import json
from collections import Counter

import pytest

import paper2.experiment2.run_stream as R
from telohelm.runtime.models import ModelResponse
from telohelm.runtime.trusted_log import verify
import telohelm.taskpacks.tanker.world.episode as E; import telohelm.taskpacks.tanker.world.phase5 as phase5


def events(folder, name):
    assert verify(folder / (name + '.log'), folder / (name + '.heads'))['ok']
    return [json.loads(line)['payload'] for line in (folder / (name + '.log')).read_text().splitlines()[1:]]


def independently_check_episode(scenario, record, folder):
    broker = events(folder, 'broker'); gateway = events(folder, 'gateway'); episode = events(folder, 'episode')
    actions = {e['action_step']: e for e in broker if e['kind'] == 'action'}
    headings = {e['action_step']: e for e in broker if e['kind'] == 'decision' and e['decision'] == 'heading'}
    steps = [e for e in broker if e['kind'] == 'world_step']
    counts = Counter({name: 0 for name in ['procedure', 'small_model', 'reasoning_model', 'holding_pending', 'holding_none']})
    world = E.Episode(scenario); world.reset()
    for n, step in enumerate(steps):
        assert step['action_step'] == n
        action = actions[n]
        if step['accepted'] and 'heading_deg' in action['action']:
            assert action['field_writers']['heading_deg'] == 'heading'
            category = headings[n]['source']
        else:
            category = 'holding_pending' if action['pending']['heading'] else 'holding_none'
        counts[category] += 1
        bundle = world.step(action['action'])
        assert world.record.actions[-1]['accepted'] == step['accepted']
        assert (bundle is None) == step['ended']
    out = world.outcome()
    for field in ['goal_met', 'joined', 'report_correct', 'end_reason', 'ended_at', 'action_steps']:
        assert record[field] == out[field]
    assert record['complete'] == (out['end_reason'] is not None)
    assert sum(counts.values()) == record['seconds'] == len(steps)
    for name, count in counts.items():
        assert record['share_of_seconds'][name] == count / len(steps)
    ends = [e for e in gateway if e['kind'] == 'model_end']
    starts = [e for e in gateway if e['kind'] == 'model_start']
    assert len(ends) == len(starts) == record['model_calls']
    assert len({e['request']['request_id'] for e in starts}) == len(starts)
    assert record['tokens_in'] == sum(e['response']['tokens_in'] for e in ends)
    assert record['tokens_out'] == sum(e['response']['tokens_out'] for e in ends)
    end = [e for e in episode if e['kind'] == 'episode_end'][-1]
    assert end['goal_met'] == record['goal_met'] and end['end_reason'] == record['end_reason']
    return dict(counts)


def test_short_stream_has_independent_world_cost_and_attribution_agreement(tmp_path, monkeypatch):
    monkeypatch.setattr(R.seed, 'stock', lambda library: [])  # invented empty library exercises model fallback
    out = tmp_path / 'out'; done = R.run_stream(2, 1, out, max_steps=8)
    total = Counter()
    for n, record in enumerate(done['episodes'], start=1):
        total.update(independently_check_episode(phase5.draw(n).world_a, record, out / record['episode']))
    s = done['summary']
    assert s['share_of_seconds']['incomplete']['counts'] == dict(total)
    assert s['share_of_seconds']['incomplete']['K'] == 16
    assert s['model_calls'] > 0 and s['model_calls'] == sum(r['model_calls'] for r in done['episodes'])
    assert s['tokens_in'] == sum(r['tokens_in'] for r in done['episodes'])
    assert s['tokens_out'] == sum(r['tokens_out'] for r in done['episodes'])


def test_natural_ending_is_separated_from_a_step_limited_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(E, 'HORIZON', 5)  # invented short world horizon, not a changed scored protocol
    out = tmp_path / 'out'; done = R.run_stream(1, 3, out, max_steps=5)
    record = done['episodes'][0]
    independently_check_episode(phase5.draw(3).world_a, record, out / record['episode'])
    assert record['complete'] and record['ended_at'] == 5
    assert done['summary']['share_of_seconds']['completed']['K'] == 5
    assert done['summary']['share_of_seconds']['incomplete']['K'] == 0


# Fixed in TURN 115 (Claude): MVP114.COST.UNKNOWN: missing model token usage becomes a zero total
def test_unknown_usage_is_not_presented_as_observed_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(R.seed, 'stock', lambda library: [])  # ensure a real fallback call in this fixture
    monkeypatch.setattr(R.seed, 'stand_in_model', lambda request: ModelResponse('timed_out'))
    out = tmp_path / 'out'; done = R.run_stream(1, 1, out, max_steps=5)
    record = done['episodes'][0]
    ends = [e for e in events(out / record['episode'], 'gateway') if e['kind'] == 'model_end']
    assert ends and all(e['response']['tokens_in'] is None for e in ends)
    assert record['tokens_in'] is None and record['tokens_out'] is None
    assert done['summary']['tokens_in'] is None and done['summary']['tokens_out'] is None
