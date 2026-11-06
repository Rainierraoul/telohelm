"""TURN96 independent cost-configuration positives and startup/capture scope probes.

All bytes/providers/files are invented. Scope probes passing demonstrate limits,
not real startup readiness or executed-code identity.
"""
import base64
import hashlib
import importlib.util
import json

import pytest

import telohelm.runtime.exchange as X; import telohelm.runtime.startup as S
from telohelm.runtime.trusted_log import verify
from .test_exchange_join import Wire; from .test_exchange_join import adapter; from .test_exchange_join import checked
from .test_local_adapter import ok_body
from .test_transport import http
from .test_turn65_review_codex import gateway; from .test_turn65_review_codex import submit  # noqa: F401
from .test_turn90_review_codex import first


@pytest.mark.parametrize('path_used', ['store', 'journal'])
def test_fallback_cost_rejects_different_configuration_with_same_request_and_goal(gateway, tmp_path, path_used):
    wire, path, journal, original = first(gateway, tmp_path)
    body = ok_body(); body['usage'].update(prompt_tokens=33, completion_tokens=6, total_tokens=39)
    other_wire = Wire(reply=http(json.dumps(body).encode()))
    other_adapter = adapter(other_wire, X.ExchangeStore(tmp_path / 'other-x'), reportable=True,
                            temperature=.5, config=checked(temperature=.5))
    other, _, _, other_root = gateway(provider=other_adapter, journal=tmp_path / 'other-j')
    assert submit(other)['status'] == 'answered'
    rec = json.loads(path.read_text()); other_rec = json.loads((tmp_path / 'other-x').read_text())
    assert rec['request_id'] == other_rec['request_id'] and rec['request'] == other_rec['request']
    assert rec['config_sha256'] != other_rec['config_sha256']
    changed = ok_body(); changed['usage'].update(prompt_tokens=99, completion_tokens=99, total_tokens=198)
    raw = http(json.dumps(changed).encode())
    rec.update(reply_wire=base64.b64encode(raw).decode(), reply_sha256=hashlib.sha256(raw).hexdigest(), reply_bytes=len(raw))
    path.write_text(json.dumps(rec) + '\n')
    path.with_name(path.name + '.anchored').write_text(json.dumps({
        'exchange_id': rec['exchange_id'], 'record_sha256': X.record_sha256(rec),
        'log': str(other_root / 'log'), 'heads': str(other_root / 'heads')}) + '\n')
    assert verify(original / 'log', original / 'heads')['ok'] and verify(other_root / 'log', other_root / 'heads')['ok']
    gw, ledger, _, _ = gateway(provider=adapter(wire, X.ExchangeStore(path), reportable=True),
                               journal=journal if path_used == 'journal' else tmp_path / 'j2')
    out = submit(gw)
    assert out['text'] is None and out['reportable'] is False and wire.calls == other_wire.calls == 1
    assert ledger.calls[-1].tokens_in in (None, 11) and ledger.calls[-1].tokens_out in (None, 4)


def test_end_only_code_drift_probe_does_not_establish_no_transient_changes(tmp_path):
    source = tmp_path / 'probe.py'; old = b'def answer():\n    return 1\n'; new = b'def answer():\n    return 2\n'
    source.write_bytes(old)
    spec = importlib.util.spec_from_file_location('turn96_probe', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    captured = S.capture_code([module], tmp_path)
    source.write_bytes(new); assert S.code_drift(captured, tmp_path) == ['probe.py']
    source.write_bytes(old); assert S.code_drift(captured, tmp_path) == []
    # The two endpoint checks cannot establish absence of a transient source change.
    assert module.answer() == 1


def test_current_gateway_is_unwired_to_startup_admission_scope_probe(gateway, tmp_path):
    # Adapted in TURN 97 (Claude), as Codex allowed: the probe becomes a missing-start refusal control. With no start
    # record a store-backed gateway refuses before any provider work, allowance or charge.
    wire = Wire()
    gw, ledger, _, root = gateway(provider=adapter(wire, X.ExchangeStore(tmp_path / 'x'), reportable=True), start=False)
    out = submit(gw)
    events = [json.loads(line)['payload'] for line in (root / 'log').read_text().splitlines()[1:]]
    assert not any(e['kind'] == 'start_record' for e in events)
    assert out['status'] == 'refused' and out['refusal'] == 'startup not admitted' and out['text'] is None
    assert wire.calls == 0 and ledger.calls == [] and not any(e['kind'] == 'model_start' for e in events)
