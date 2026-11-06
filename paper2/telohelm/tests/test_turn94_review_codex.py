"""TURN94: invented startup-value, coverage and original-cost controls; CPU only."""
import base64
import hashlib
import importlib.util
import json

import pytest

import telohelm.runtime.startup as S; import telohelm.runtime.exchange as X; import telohelm.runtime.models as M
from telohelm.runtime.trusted_log import TrustedLog; from telohelm.runtime.trusted_log import verify
from .test_exchange_join import Wire; from .test_exchange_join import adapter; from .test_exchange_join import checked
from .test_local_adapter import ok_body
from .test_transport import http
from .test_startup import matching
from .test_turn65_review_codex import gateway; from .test_turn65_review_codex import submit  # noqa: F401
from .test_turn90_review_codex import first


# Fixed in TURN 95 (Claude): START94.TYPES: Python equality admits observed booleans/floats under integer pins
@pytest.mark.parametrize('change', ['concurrency_bool', 'max_tokens_float'])
def test_observed_value_types_must_match_frozen_types(change):
    c = checked(); rec = matching(c)
    if change == 'concurrency_bool': rec['items']['environment']['value']['concurrency'] = True
    else:
        value = rec['items']['effective_defaults']['value']['max_tokens']
        value['small_model'] = float(value['small_model'])
    try:
        problems = S.compare(rec, c)
    except S.StartupError:
        return
    assert problems


# Fixed in TURN 95 (Claude): START94.PROMPTS: prompt/template/parser pins have no required observed counterpart
def test_missing_prompt_template_parser_evidence_cannot_admit_startup(tmp_path):
    c = checked(); expected = S.expected(c)
    legacy = ('installed_code', 'python', 'dependencies', 'weights', 'launch', 'model_list',
              'effective_defaults', 'environment')
    values = {name: expected[name] for name in legacy}
    values['installed_code'] = c.section('runtime')['code_sha256']  # runtime source map only; all three prompt pins absent
    rec = S.record(c.fingerprint, 'invented pre-call capture',
                   {name: S.observed(value, 'invented source', 'invented time', 'invented protection')
                    for name, value in values.items()})
    log = TrustedLog(tmp_path / 'log', tmp_path / 'heads', 'invented')
    S.write(log, rec)
    assert verify(log.path, log.protected)['ok']
    assert S.admit_start(log.path, log.protected, c)


# Fixed in TURN 95 (Claude): COST94.IDENTITY: registered fallback model_end matched by request_id alone
@pytest.mark.parametrize('path_used', ['store', 'journal'])
def test_fallback_usage_cannot_come_from_another_original_goal(gateway, tmp_path, path_used):
    wire, path, journal, original = first(gateway, tmp_path)
    body = ok_body(); body['usage'].update(prompt_tokens=33, completion_tokens=6, total_tokens=39)
    other_wire = Wire(reply=http(json.dumps(body).encode()))
    other, other_ledger, _, other_root = gateway(provider=adapter(other_wire, X.ExchangeStore(tmp_path / 'other-x'),
                                                                reportable=True), journal=tmp_path / 'other-j')
    other_out = submit(other, goal=2)
    assert other_out['status'] == 'answered' and other_ledger.calls[-1].tokens_in == 33
    # Legitimate goal2 and goal1 runs share the sequential request_id. The full immutable original meta differs.
    assert other_ledger.calls[-1].request_id == json.loads(path.read_text())['request_id']
    assert verify(original / 'log', original / 'heads')['ok'] and verify(other_root / 'log', other_root / 'heads')['ok']
    rec = json.loads(path.read_text()); changed = ok_body()
    changed['usage'].update(prompt_tokens=99, completion_tokens=99, total_tokens=198)
    raw = http(json.dumps(changed).encode())
    rec.update(reply_wire=base64.b64encode(raw).decode(), reply_sha256=hashlib.sha256(raw).hexdigest(), reply_bytes=len(raw))
    path.write_text(json.dumps(rec) + '\n')
    # Unprotected pointer now names a registered, valid original of a DIFFERENT goal.
    marks = path.with_name(path.name + '.anchored')
    marks.write_text(json.dumps({'exchange_id': rec['exchange_id'], 'record_sha256': X.record_sha256(rec),
                                'log': str(other_root / 'log'), 'heads': str(other_root / 'heads')}) + '\n')
    store = X.ExchangeStore(path)
    assert M.verify_anchors(store, [(other_root / 'log', other_root / 'heads')])
    gw, ledger, _, _ = gateway(provider=adapter(wire, store, reportable=True),
                               journal=journal if path_used == 'journal' else tmp_path / 'j2')
    out = submit(gw)
    assert out['text'] is None and out['reportable'] is False and wire.calls == 1 and other_wire.calls == 1
    # Unknown or independently verified goal1 usage, never goal2's33/6.
    assert ledger.calls[-1].tokens_in in (None, 11) and ledger.calls[-1].tokens_out in (None, 4)


def test_nominal_start_record_has_exact_required_observed_values_and_order(tmp_path):
    c = checked(); rec = matching(c)
    log = TrustedLog(tmp_path / 'log', tmp_path / 'heads', 'invented')
    S.write(log, rec); log.append({'kind': 'model_start'})
    assert S.admit_start(log.path, log.protected, c) == []


def test_code_capture_reports_disk_source_and_does_not_establish_loaded_behavior(tmp_path):
    """Scope probe: passing demonstrates the limitation, not capture readiness or a failed invariant."""
    source = tmp_path / 'loaded_probe.py'; old = b'def answer():\n    return 1\n'
    new = b'def answer():\n    return 2\n'; source.write_bytes(old)
    spec = importlib.util.spec_from_file_location('turn94_loaded_probe', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.answer() == 1
    source.write_bytes(new)
    captured = S.capture_code([module], tmp_path)
    assert captured['loaded_probe.py'] == hashlib.sha256(new).hexdigest()
    assert captured['loaded_probe.py'] != hashlib.sha256(old).hexdigest() and module.answer() == 1
