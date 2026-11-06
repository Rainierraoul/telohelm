"""Interruption controls with invented outcomes and providers; no real model calls."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import telohelm.learning as G; import telohelm.learning_monitor as GM
import telohelm.runtime.library as LIB
from paper2.experiment2.run_stream import _write_ledger
from .test_learning import coordinator; from .test_learning import Scripted; from .test_learning import reply; from .test_learning import proposer_with


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_killing_the_next_proposal_keeps_full_prior_verdicts_and_trial_set(tmp_path):
    program = '\nimport os, sys\nfrom pathlib import Path\nfrom types import SimpleNamespace\nfrom paper2.experiment2.tests.test_learning import coordinator; from paper2.experiment2.tests.test_learning import reply\n\nclass StopOnSecond:\n    def __init__(self): self.n = 0\n    def __call__(self, messages, meta):\n        self.n += 1\n        if self.n == 2: os._exit(73)\n        return reply()\n\ndef fly(shadow, c, root):\n    return {"episode": c.episode, "goal_met": True, "help": 0,\n            "end_reason": "join", "private": "invented-private-evidence"}\n\nco, _ = coordinator(Path(sys.argv[1]), StopOnSecond(), fly=fly, trial_set=5)\nfor i in range(10):\n    co.after_episode(SimpleNamespace(episode_id=f"invented-{i}"),\n                     {"goal_met": False, "joined": False, "report_correct": False}, True)\n'
    result = subprocess.run([sys.executable, '-c', program, str(tmp_path)],
                            env={**os.environ, 'PYTHONPATH': str(Path(G.__file__).parents[1])})
    assert result.returncode == 73
    saved = read_rows(tmp_path / 'learning-events.jsonl')
    assert [r['event_number'] for r in saved] == list(range(1, len(saved) + 1))
    assert len({r['session'] for r in saved}) == 1
    events = [r['event'] for r in saved]
    verdict, = [e for e in events if e['kind'] == 'attempt_result']
    assert verdict['passed'] and verdict['detail']['trial'][0]['private'] == 'invented-private-evidence'
    assert events[-1]['kind'] == 'attempt_start' and events[-1]['attempt'] == 2
    assert events[-1]['trial_set'] == [f'invented-{i}' for i in range(5, 10)]
    assert not (tmp_path / 'learning-events.json').exists()  # No normal-end export occurred.


def test_monitor_instances_have_distinct_journal_sessions_without_erasing_history(tmp_path):
    first, _ = coordinator(tmp_path, Scripted())
    event = first._event('invented', detail={'private': 1})
    second, _ = coordinator(tmp_path, Scripted())
    second._event('another')
    saved = read_rows(tmp_path / 'learning-events.jsonl')
    assert saved[0]['event'] == event and saved[1]['event'] == second.events[0]
    assert saved[0]['session'] != saved[1]['session']
    assert [r['event_number'] for r in saved] == [1, 1]


def test_event_sync_failure_stops_before_it_is_published_in_memory(tmp_path, monkeypatch):
    co, _ = coordinator(tmp_path, Scripted())
    def fail(fd):
        raise OSError('invented full disk')
    monkeypatch.setattr(GM.os, 'fsync', fail)
    with pytest.raises(OSError, match='full disk'):
        co._event('invented')
    assert co.events == []


def test_completed_costs_exist_before_the_stream_ends_and_cache_does_not_double_charge(tmp_path, monkeypatch):
    p, sent = proposer_with(tmp_path, monkeypatch, count_tokens=lambda _: 1)
    messages, meta = [('user', 'invented')], {'episode': 'invented', 'attempt': 1, 'call': 1}
    p(messages, meta)
    path = tmp_path / 'learning-ledger.jsonl'
    first = path.read_bytes()
    assert len(read_rows(path)) == len(sent.sent) == 1
    q, sent_again = proposer_with(tmp_path, monkeypatch, count_tokens=lambda _: 1)
    q(messages, meta)
    assert not sent_again.sent and path.read_bytes() == first
    assert len(q.ledger.calls) == 1
    # A new refusal writes no charge and cannot remove an earlier one.
    q.count_tokens = lambda _: 10**6
    q(messages, {**meta, 'attempt': 2})
    assert path.read_bytes() == first


def test_conflicting_charge_is_refused_and_original_is_kept(tmp_path, monkeypatch):
    p, _ = proposer_with(tmp_path, monkeypatch, count_tokens=lambda _: 1)
    p([('user', 'invented')], {'episode': 'invented', 'attempt': 1, 'call': 1})
    path = tmp_path / 'learning-ledger.jsonl'; first = path.read_bytes()
    p.ledger.calls[0] = replace(p.ledger.calls[0], tokens_out=999)
    with pytest.raises(ValueError, match='conflicting charge'):
        p.persist_costs()
    assert path.read_bytes() == first


def test_final_ledger_replace_failure_preserves_the_previous_complete_file(tmp_path, monkeypatch):
    path = tmp_path / 'ledger.jsonl'; path.write_text('old complete record\n')
    def fail(source, target):
        raise OSError('invented interruption before rename')
    monkeypatch.setattr(LIB.os, 'replace', fail)
    with pytest.raises(OSError, match='before rename'):
        _write_ledger(path, [])
    assert path.read_text() == 'old complete record\n'


@pytest.mark.parametrize('settled', [True, False])
def test_each_trial_result_is_saved_before_return_and_matches_its_verdict(tmp_path, monkeypatch, settled):
    co, lib = coordinator(tmp_path, Scripted())
    sha = lib.add({'kind': 'procedure', 'decision': 'heading', 'conditions': [],
                   'source': 'def decide(ctx):\n    return {"heading_deg": 123}\n', 'origin': {'invented': True}})
    co.monitor._candidate = (sha, 'heading')
    outcome = {'goal': {'version': 1}, 'goal_met': False, 'joined': True, 'in_time': True,
               'settled': settled, 'end_reason': 'join', 'help_requests': [], 'reports': [], 'refusals': [],
               'invented_bound': 1 if settled else float('inf')}
    def invented_flight(scenario, shadow, models, log, ledger, **kwargs):
        log.append({'kind': 'decision', 'decision': 'heading', 'method': sha,
                    'goal_version': 1, 't': 0, 'part': {'heading_deg': 123}})
        return {'outcome': outcome, 'actions': [{'heading_deg': 123}]}
    monkeypatch.setattr(GM.L, 'run_episode', invented_flight)
    root = tmp_path / 'scratch'; root.mkdir()
    c = G.Completed(1, 'invented', SimpleNamespace(), False, 'both', True)
    result = co.monitor._fly(lib, c, root)
    saved = Path(result['evidence_dir']) / 'result.json'
    assert json.loads(saved.read_text()) == result
    assert result['component']['passed'] is settled and not result['goal_met']
    assert saved.stat().st_mode & 0o222 == 0


def test_unknown_model_usage_stays_unknown_in_the_saved_charge(tmp_path, monkeypatch):
    from telohelm.runtime.models import ScriptedProvider; from telohelm.runtime.models import ModelResponse
    p, _ = proposer_with(tmp_path, monkeypatch, count_tokens=lambda _: 1)
    p.provider = ScriptedProvider(lambda request: ModelResponse('timed_out', None))
    p([('user', 'invented')], {'episode': 'invented', 'attempt': 1, 'call': 1})
    row, = read_rows(tmp_path / 'learning-ledger.jsonl')
    assert row['status'] == 'timed_out' and row['tokens_in'] is None and row['tokens_out'] is None


def test_exit_inside_next_model_request_keeps_first_charge_and_explicit_unfinished_start(tmp_path):
    program = "\nimport os, sys\nfrom pathlib import Path\nimport telohelm.learning as G; import telohelm.local_model as LM\nfrom telohelm.runtime.models import ModelResponse; from telohelm.runtime.models import ScriptedProvider\ncalls = 0\ndef answer(request):\n    global calls\n    calls += 1\n    if calls == 2: os._exit(74)\n    return ModelResponse('answered', '{}', tokens_in=1, tokens_out=2)\nLM.provider = lambda *a, **kw: ScriptedProvider(answer)\np = G.GatewayProposer('invented', 'invented', Path(sys.argv[1]), 'invented', count_tokens=lambda _: 1)\nfor n in [1, 2]:\n    p([('user', 'invented')], {'episode': 'invented', 'attempt': 1, 'call': n})\n"
    done = subprocess.run([sys.executable, '-c', program, str(tmp_path)],
                          env={**os.environ, 'PYTHONPATH': str(Path(G.__file__).parents[1])})
    assert done.returncode == 74
    row, = read_rows(tmp_path / 'learning-ledger.jsonl')
    assert row['tokens_out'] == 2
    events = [r['payload'] for r in read_rows(tmp_path / 'learning-attempt-1/gateway.log')]
    assert sum(e['kind'] == 'model_start' for e in events) == 2
    assert sum(e['kind'] == 'model_end' for e in events) == 1
    assert len(read_rows(tmp_path / 'learning-answers.jsonl')) == 1
