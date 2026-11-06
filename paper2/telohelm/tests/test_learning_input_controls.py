"""Local fitting, current status and cache controls; scripted providers only."""
import json
import socket

import pytest

from telohelm import learning as G
from telohelm.demo_server import ServerConfig, proposer as server_proposer
from telohelm.runtime.library import Library
from telohelm.runtime.models import ModelResponse, ScriptedProvider

SOURCE = "def decide(ctx):\n    return {'heading_deg': 90}\n"


def coordinator(path, proposer):
    co = G.Coordinator(Library(path / 'library'), path, proposer, lambda scenario: True, None)
    co.completed = [G.Completed(5, 'baseline', None, False, 'both', True)]
    co._remember(1, 'procedure', 'heading', 'trial', ['the procedure failed its trial'], SOURCE)
    return co


def gateway(path, count):
    p = G.GatewayProposer('http://127.0.0.1:1', 'invented', path, 'input-control', count_tokens=count)
    p.provider = ScriptedProvider(lambda request: ModelResponse('answered', '{}', 1, 1))
    return p


@pytest.mark.parametrize('cap,kept', [(250, ['old', 'new']), (150, ['new'])])
def test_optional_source_is_dropped_before_public_examples(tmp_path, cap, kept):
    def count(messages):
        body = json.loads(messages[1][1])
        return 10 + 100 * len(body.get('examples', [])) + (
            1000 if 'source' in body['previous_attempts'][-1] else 0)
    p = gateway(tmp_path / 'gateway', count)
    p.input_cap = cap
    co = coordinator(tmp_path / 'co', p)
    co.examples = True
    co._attempt_examples = ([{'label': 'old'}, {'label': 'new'}],
                            [{'ordinal': 1, 'second': 0}, {'ordinal': 2, 'second': 0}])
    record = {}
    messages = co.messages([{'episode': 'baseline', 'failed': 'both'}], log=record)
    body = json.loads(messages[1][1])
    assert [e['label'] for e in body['examples']] == kept
    assert body['previous_attempts'][-1]['source_omitted']
    assert record['source_omitted_for_tokens'] and record['input_tokens'] <= cap
    assert co.history[-1]['source'] == SOURCE  # fitting never edits the remembered original


def test_fitting_keeps_source_when_it_fits(tmp_path):
    co = coordinator(tmp_path / 'co', gateway(tmp_path / 'gateway', lambda messages: 10))
    body = json.loads(co.messages([])[1][1])
    assert body['previous_attempts'][-1]['source'] == SOURCE
    assert 'source_omitted' not in body['previous_attempts'][-1]


def test_irreducible_overflow_keeps_rules_feedback_library_and_repair_tail(tmp_path):
    p = gateway(tmp_path / 'gateway', lambda messages: 999)
    p.input_cap = 100
    co = coordinator(tmp_path / 'co', p)
    tail = [('assistant', 'original candidate'), ('user', 'essential repair instruction')]
    messages = co.messages([{'episode': 'baseline', 'failed': 'both'}], tail)
    body = json.loads(messages[1][1])
    assert messages[0][1] == co.rules and messages[-2:] == tail
    assert body['feedback'] == [{'episode': 'baseline', 'failed': 'both'}]
    assert body['library'] == {} and 'context_fields' in body
    assert body['previous_attempts'][-1]['source_omitted']
    p(messages, {'episode': 'baseline', 'attempt': 2, 'call': 1})
    assert p.provider.calls == 0 and p.last_status == 'preflight_refused'


def test_counter_failure_remains_unknown_and_is_refused(tmp_path):
    def unavailable(messages):
        raise RuntimeError('offline counter unavailable')
    p = gateway(tmp_path / 'gateway', unavailable)
    co = coordinator(tmp_path / 'co', p)
    record = {}
    messages = co.messages([], log=record)
    assert json.loads(messages[1][1])['previous_attempts'][-1]['source'] == SOURCE
    assert record['input_tokens'] is None and record['token_fit_error'] == 'RuntimeError'
    p(messages, {'episode': 'baseline', 'attempt': 2, 'call': 1})
    assert p.provider.calls == 0 and not p.ledger.calls
    assert p.last_status == 'preflight_refused'


@pytest.mark.parametrize('value', [True, -1, 1.5, '3', None])
def test_malformed_count_cannot_be_coerced_into_permission_to_send(tmp_path, value):
    p = gateway(tmp_path / 'gateway', lambda messages: value)
    p([('user', 'a request')], {'episode': 'baseline', 'attempt': 1, 'call': 1})
    assert p.provider.calls == 0 and p.last_status == 'preflight_refused'
    row = json.loads((tmp_path / 'gateway/learning-attempt-1/preflight.jsonl').read_text())
    assert row['input_tokens'] is None and not row['sent'] and row['error'] == 'ValueError'


@pytest.mark.parametrize('candidates', [1, 2])
def test_unsent_attempt_cannot_raise_budget_because_an_older_call_timed_out(tmp_path, monkeypatch, candidates):
    p = gateway(tmp_path / 'gateway', lambda messages: 1)
    first_provider = ScriptedProvider(lambda request: ModelResponse('timed_out'))
    p.provider = first_provider
    p([('user', 'old')], {'episode': 'old', 'attempt': 0, 'call': 1})
    original = p.ledger.calls[-1]
    p.count_tokens = lambda messages: p.input_cap + 1
    co = coordinator(tmp_path / 'co', p)
    co.candidates = candidates
    co.budget = G.AdaptiveBudget()
    co.budget.observe('budget')  # one earlier miss; another false timeout would raise the level
    monkeypatch.setattr(socket, 'create_connection', lambda *a, **kw: pytest.fail('Unsent request used network'))
    co.attempt(co.completed[-1])
    assert co.budget.level == 0 and co.budget.misses == 1
    assert co.history[-1]['why'] == ['preflight_refused']
    assert p.ledger.calls == [original] and first_provider.calls == 1


def test_cached_status_belongs_to_its_own_request_and_cost_is_restored_once(tmp_path):
    path = tmp_path / 'gateway'
    p = gateway(path, lambda messages: 1)
    p.provider = ScriptedProvider(lambda request: ModelResponse('timed_out'))
    old, new = [('user', 'old')], [('user', 'new')]
    one, two = {'episode': 'baseline', 'attempt': 1, 'call': 1}, {'episode': 'baseline', 'attempt': 2, 'call': 1}
    p(old, one)
    p.count_tokens = lambda messages: p.input_cap + 1
    p(new, two)
    rows = [json.loads(line) for line in (path / 'learning-answers.jsonl').read_text().splitlines()]
    rows[-1].pop('call_status')  # historical rows without the added status still restore safely
    (path / 'learning-answers.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    q = gateway(path, lambda messages: 1)
    q(new, two)
    assert q.last_status == 'preflight_refused' and q.provider.calls == 0 and not q.ledger.calls
    q(old, one)
    q(old, one)
    assert q.last_status == 'timed_out' and len(q.ledger.calls) == 1 and q.provider.calls == 0


def test_contradictory_cached_status_is_refused_before_restoring_a_charge(tmp_path):
    path = tmp_path / 'gateway'
    p = gateway(path, lambda messages: 1)
    messages, meta = [('user', 'one')], {'episode': 'baseline', 'attempt': 1, 'call': 1}
    p(messages, meta)
    answer = path / 'learning-answers.jsonl'
    row = json.loads(answer.read_text());row['call_status'] = 'timed_out'
    answer.write_text(json.dumps(row) + '\n')
    q = gateway(path, lambda messages: 1)
    with pytest.raises(ValueError, match='contradictory saved status'):
        q(messages, meta)
    assert not q.ledger.calls and q.provider.calls == 0


def test_constructing_server_messages_uses_only_the_local_counter(tmp_path, monkeypatch):
    work = tmp_path / 'gateway';work.mkdir()
    p = server_proposer(work, ServerConfig('http://127.0.0.1:1', 262144, max_output_tokens=4096, input_cap=4096))
    co = coordinator(tmp_path / 'co', p)
    co.history[-1]['source'] = 'def decide(ctx):\n' + '    x=1\n' * 650 + "    return {'heading_deg': 90}\n"
    monkeypatch.setattr(socket, 'create_connection', lambda *a, **kw: pytest.fail('Fitting contacted the server'))
    messages = co.messages([])
    assert json.loads(messages[1][1])['previous_attempts'][-1]['source_omitted']
    assert not (work / 'token-count-gates.jsonl').exists()
