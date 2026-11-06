"""TURN92 independent verified-usage and fresh wire controls; invented CPU faults only."""
import base64
import copy
import hashlib
import json

import pytest

import telohelm.runtime.exchange as X; import telohelm.runtime.models as M
from telohelm.runtime.trusted_log import verify
from .test_exchange_join import Wire; from .test_exchange_join import adapter
from .test_local_adapter import ok_body
from .test_transport import http
from .test_turn65_review_codex import gateway; from .test_turn65_review_codex import submit  # noqa: F401
from .test_turn90_review_codex import first


# Fixed in TURN 93 (Claude): COST92.UNVERIFIED: rejected original receipt still supplies observed ledger usage
@pytest.mark.parametrize('path_used', ['store', 'journal'])
def test_rejected_original_bytes_do_not_supply_verified_token_totals(gateway, tmp_path, path_used):
    wire, path, journal, original = first(gateway, tmp_path)
    original_heads = (original / 'heads').read_bytes()
    body = ok_body(); body['usage'].update(prompt_tokens=99, completion_tokens=99, total_tokens=198)
    raw = http(json.dumps(body).encode()); rec = json.loads(path.read_text())
    rec.update(reply_wire=base64.b64encode(raw).decode(), reply_sha256=hashlib.sha256(raw).hexdigest(), reply_bytes=len(raw))
    path.write_text(json.dumps(rec) + '\n')
    marks = path.with_name(path.name + '.anchored'); mark = json.loads(marks.read_text())
    mark['record_sha256'] = X.record_sha256(rec); marks.write_text(json.dumps(mark) + '\n')
    assert (original / 'heads').read_bytes() == original_heads and verify(original / 'log', original / 'heads')['ok']
    store = X.ExchangeStore(path); a = adapter(wire, store, reportable=True)
    assert M.verify_anchors(store, [(original / 'log', original / 'heads')])
    gw, ledger, _, _ = gateway(provider=a, journal=journal if path_used == 'journal' else tmp_path / 'j2')
    out = submit(gw)
    assert out['text'] is None and out['reportable'] is False and wire.calls == 1
    # Verified original11/4 from independently protected completion, or explicit unknown.
    # Rejected99/99 may be retained separately as claims, never observed costs.
    assert ledger.calls[-1].tokens_in in (None, 11) and ledger.calls[-1].tokens_out in (None, 4)


class ChangedWire(Wire):
    def __init__(self, change):
        super().__init__(); self.change = change

    def exchange(self, url, body, timeout_s):
        altered = copy.deepcopy(body)
        if self.change == 'model': altered['model'] = 'different invented model'
        else: altered['messages'][0]['content'] = 'different invented input'
        return super().exchange(url, altered, timeout_s)


# Fixed in TURN 93 (Claude): WIRE92.FRESH: request-byte join runs on reuse but fresh output escapes
@pytest.mark.parametrize('change', ['model', 'messages'])
def test_fresh_reply_is_not_usable_before_actual_wire_request_join(gateway, tmp_path, change):
    wire = ChangedWire(change); store = X.ExchangeStore(tmp_path / 'x')
    a = adapter(wire, store, reportable=True)
    gw, ledger, _, original = gateway(provider=a, journal=tmp_path / 'j')
    out = submit(gw)
    assert wire.calls == 1 and verify(original / 'log', original / 'heads')['ok']
    assert a._wire_problem(store.records()[0]) is not None  # checker detects saved mismatch, after release
    assert out['text'] is None and out['status'] != 'answered' and out['reportable'] is False


def test_semantic_wire_fields_are_checked_after_body_digest_agrees(gateway, tmp_path):
    wire = Wire(); store = X.ExchangeStore(tmp_path / 'x'); a = adapter(wire, store, reportable=True)
    gw, _, _, _ = gateway(provider=a); assert submit(gw)['status'] == 'answered'
    saved = store.records()[0]; head, _, raw = saved['request_wire'].partition(b'\r\n\r\n')
    for field, value in [('model', 'other'), ('temperature', .7), ('seed', 5), ('max_tokens', 9),
                         ('messages', [{'role': 'user', 'content': 'other question'}])]:
        body = json.loads(raw); body[field] = value
        candidate = {**saved, 'request_wire': head + b'\r\n\r\n' + json.dumps(body).encode(),
                     'body_sha256': M.body_sha256(body)}
        reason = a._wire_problem(candidate)
        assert reason == ('request bytes do not carry the request\'s messages' if field == 'messages'
                          else 'request bytes do not match the configuration')


def test_separate_trusted_registry_recovery_preserves_valid_original_cost(gateway, tmp_path):
    wire, path, journal, original = first(gateway, tmp_path)
    roots = M.ProtectedRoots(); roots.register(original / 'log', original / 'heads')
    a = adapter(wire, X.ExchangeStore(path), reportable=True, roots=roots)
    gw, ledger, _, _ = gateway(provider=a, journal=journal)
    out = submit(gw)
    assert out['text'] == 'invented answer' and out['reportable'] is True and wire.calls == 1
    assert ledger.calls[-1].tokens_in == 11 and ledger.calls[-1].tokens_out == 4 and ledger.calls[-1].wall_s == 2.
