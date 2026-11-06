"""Owned loopback stand-in only: no model server, inference or GPU reservation."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading

import pytest

pytest.importorskip('gymnasium')

from telohelm import demo_cartpole as demo, token_count
from telohelm.demo_server import MODEL, ServerConfig

POLICY = "def decide(ctx):\n    return {'push': 1 if ctx.pole_angle + 0.5 * ctx.pole_angular_velocity > 0 else 0}\n"


@contextmanager
def standin(mode='answer'):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append({'path': self.path, 'body': body})
            status = 200
            if self.path == '/tokenize':
                count = token_count.count([(m['role'], m['content']) for m in body['messages']],
                                          body['chat_template_kwargs'])
                data = {'count': count + (1 if mode == 'wrong_count' else 0)}
                if mode == 'bad_count':
                    data = {'count': True}
                if mode == 'probe_failure':
                    status = 503
            else:
                assert self.path == '/v1/chat/completions'
                text = json.dumps({'kind': 'procedure', 'decision': 'control', 'conditions': [],
                                   'source': POLICY, 'reason': {'code': 'missing_tool', 'text': 'CPU stand-in'}})
                if mode == 'malformed':
                    text = 'This is not a candidate.'
                if mode == 'bad_trial':
                    text = json.dumps({'kind': 'procedure', 'decision': 'control', 'conditions': [],
                                       'source': "def decide(ctx):\n    return {'push': 0}\n",
                                       'reason': {'code': 'missing_tool', 'text': 'CPU negative control'}})
                data = {'id': 'cpu-standin', 'model': 'different-model' if mode == 'bad_model' else MODEL,
                        'choices': [{'finish_reason': 'length' if mode == 'cut_off' else 'stop',
                                     'message': {'content': text}}],
                        'usage': {'prompt_tokens': 31, 'completion_tokens': 17}}
                if mode == 'chat_failure':
                    status = 503
            raw = json.dumps(data).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def config(url, **changes):
    return ServerConfig(url, **{'context_window': 262144, 'max_output_tokens': 4096,
                               'timeout_s': 5, **changes})


def test_server_path_uses_normal_gateway_and_never_authored_fallback(tmp_path, monkeypatch):
    class ForbiddenSources:
        def __getitem__(self, key):
            pytest.fail('Server mode read an authored reply')
    monkeypatch.setattr(demo, 'SOURCES', ForbiddenSources())
    from telohelm.runtime import models
    monkeypatch.setattr(models, 'ScriptedProvider', lambda *a, **kw: pytest.fail('Scripted fallback'))
    with standin() as (url, requests):
        result = demo.run(tmp_path / 'server', server=config(url))
    assert result['passed'], result['checks']
    assert result['inference_verified'] is False and result['real_model_calls'] is None
    assert result['server_chat_requests'] == len(result['learning_charges']) == 1
    assert result['learning_charges'][0]['tokens_in'] == 31
    assert result['learning_charges'][0]['tokens_out'] == 17
    assert not (tmp_path / 'server/authored-calls.json').exists()
    assert [r['path'] for r in requests] == ['/tokenize', '/v1/chat/completions']
    probe, chat = [r['body'] for r in requests]
    assert probe['messages'] == chat['messages']
    assert probe['chat_template_kwargs'] == chat['chat_template_kwargs']
    assert chat['model'] == MODEL and chat['max_tokens'] == 4096
    prompt = json.dumps(chat['messages'])
    assert POLICY not in prompt
    user = json.loads(chat['messages'][-1]['content'])
    assert set(user['library']) == {'control guard (fixed)'}
    assert user['library']['control guard (fixed)'] == demo.GUARD
    assert 'seed' not in user and 'scenario' not in user
    assert [r['seed'] for r in result['episodes']] == [11, 23, 24]
    assert all(r['goal_met'] and r['procedure_steps'] == 128 for r in result['episodes'][1:])


@pytest.mark.parametrize('mode', ['wrong_count', 'bad_count', 'probe_failure'])
def test_failed_count_gate_sends_no_chat_and_cannot_adopt(tmp_path, mode):
    with standin(mode) as (url, requests):
        result = demo.run(tmp_path / mode, server=config(url))
    assert [r['path'] for r in requests] == ['/tokenize']
    assert not result['passed'] and result['server_chat_requests'] == 0
    assert all(r['procedure_steps'] == 0 for r in result['episodes'])
    gates = [json.loads(line) for line in (tmp_path / mode / 'learning/token-count-gates.jsonl').read_text().splitlines()]
    assert len(gates) == 1 and not gates[0]['chat_allowed'] and gates[0]['error']


def test_context_overflow_is_refused_before_any_socket(tmp_path, monkeypatch):
    monkeypatch.setattr(socket, 'create_connection', lambda *a, **kw: pytest.fail('Overflow sent a request'))
    result = demo.run(tmp_path / 'overflow', server=config('http://127.0.0.1:1',
                                                         context_window=1024, max_output_tokens=512))
    assert not result['passed'] and result['server_chat_requests'] == 0
    gate = json.loads((tmp_path / 'overflow/learning/token-count-gates.jsonl').read_text().splitlines()[0])
    assert 'declared context' in gate['error'] and not gate['chat_allowed']


@pytest.mark.parametrize('mode', ['malformed', 'bad_trial', 'bad_model', 'cut_off', 'chat_failure'])
def test_bad_reply_is_retained_without_retry_or_adoption(tmp_path, mode):
    with standin(mode) as (url, requests):
        result = demo.run(tmp_path / mode, server=config(url))
    assert [r['path'] for r in requests] == ['/tokenize', '/v1/chat/completions']
    assert result['server_chat_requests'] == 1 and not result['passed']
    assert all(r['procedure_steps'] == 0 for r in result['episodes'])
    assert list((tmp_path / mode / 'learning').rglob('gateway.log'))
    if mode in ('malformed', 'bad_trial', 'bad_model', 'cut_off'):
        assert result['learning_charges'][0]['tokens_out'] == 17


@pytest.mark.parametrize('changes', [{'model': 'another-model'}, {'context_window': True},
                                    {'context_window': 1024, 'max_output_tokens': 1024},
                                    {'max_output_tokens': 0}, {'input_cap': 16385},
                                    {'timeout_s': float('nan')}, {'timeout_s': 14401}, {'effort': 'high'}])
def test_unsupported_settings_fail_without_network(monkeypatch, changes):
    monkeypatch.setattr(socket, 'create_connection', lambda *a, **kw: pytest.fail('Validation made a request'))
    with pytest.raises(ValueError):
        config('http://127.0.0.1:1', **changes)


@pytest.mark.parametrize('url', ['https://127.0.0.1:1', 'http://localhost:1',
                               'http://192.0.2.1:1', 'http://name:secret@127.0.0.1:1'])
def test_only_an_explicit_numeric_loopback_origin_is_accepted(url):
    with pytest.raises(ValueError):
        config(url)


def test_incomplete_or_ignored_cli_options_do_not_create_output(tmp_path):
    for options in (['--server-url', 'http://127.0.0.1:1'], ['--context-window', '4096'],
                    ['--model', 'other-model']):
        with pytest.raises(SystemExit) as exc:
            demo.main(['--out', str(tmp_path / 'unused'), *options])
        assert exc.value.code == 2
        assert not (tmp_path / 'unused').exists()
