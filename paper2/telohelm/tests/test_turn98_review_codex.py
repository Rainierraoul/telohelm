"""TURN98 independent startup/cold-load controls. Invented records and Wire only.

All faults affect temporary files or fixture objects. No actual installed runtime,
model/server, containment attack, episode or scored data is involved.
"""
import hashlib
import importlib.util
import os
import py_compile
import sys

import pytest

import telohelm.codecapture as C
import telohelm.runtime.exchange as X; import telohelm.runtime.manifest as F; import telohelm.runtime.startup as S
from telohelm.runtime.trusted_log import verify
from .test_exchange_join import Wire; from .test_exchange_join import adapter
from .test_manifest import invented
from .test_startup import invented_code; from .test_startup import matching
from .test_startup_gate import events; from .test_startup_gate import unstarted
from .test_turn65_review_codex import gateway; from .test_turn65_review_codex import submit  # noqa: F401
from .test_turn90_review_codex import first


# Fixed in TURN 99 (Claude): SG98.CURRENT: admitted context is cached without an immutable scope binding
def test_changed_current_stream_cannot_borrow_the_old_start(gateway, tmp_path):
    wire = Wire()
    gw, ledger, _, root = gateway(provider=adapter(wire, X.ExchangeStore(tmp_path / 'x'), reportable=True))
    assert verify(root / 'log', root / 'heads')['ok']
    assert events(root)[-1]['binding']['stream'] == 'stream-a'
    try:
        gw.stream = 'another-stream'
    except AttributeError:
        return  # An immutable public context also satisfies the contract.
    out = submit(gw)
    assert wire.calls == 0 and out['text'] is None and out['reportable'] is False
    assert ledger.calls == [] and gw.used == {'small_model': 0, 'reasoning_model': 0}
    assert not any(e['kind'] == 'model_start' for e in events(root))


# Fixed in TURN 99 (Claude): SG98.CURRENT: cached startup admission survives a missing protected start record
def test_cut_current_start_log_refuses_before_fresh_provider_work(gateway, tmp_path):
    wire = Wire()
    gw, ledger, _, root = gateway(provider=adapter(wire, X.ExchangeStore(tmp_path / 'x'), reportable=True))
    assert events(root)[-1]['kind'] == 'start_record'
    # Only the invented log is cut. Its original heads remain intact and reject the cut.
    lines = (root / 'log').read_bytes().splitlines(keepends=True)
    (root / 'log').write_bytes(b''.join(lines[:-1]))
    assert not verify(root / 'log', root / 'heads')['ok']
    try:
        out = submit(gw)
    except (OSError, ValueError, RuntimeError):
        assert wire.calls == 0 and ledger.calls == []
        return
    assert wire.calls == 0 and out['text'] is None and out['reportable'] is False
    assert ledger.calls == [] and gw.used == {'small_model': 0, 'reasoning_model': 0}


# Fixed in TURN 99 (Claude): SG98.CODE.SCHEMA: malformed capture provenance admits startup
@pytest.mark.parametrize('field,value', [('captured_at', None), ('protection', '')])
def test_malformed_code_capture_metadata_refuses_fresh_work(gateway, tmp_path, field, value):
    wire, a, gw, ledger, root = unstarted(gateway, tmp_path)
    code = invented_code(a.config); code[field] = value
    try:
        authority = S.RunAuthority(a.config, code)
    except S.StartupError:
        return
    problems = gw.record_start(matching(a.config), authority)
    out = submit(gw)
    assert problems and wire.calls == 0 and out['text'] is None and out['reportable'] is False
    assert ledger.calls == [] and not any(e['kind'] == 'model_start' for e in events(root))


# Fixed in TURN 99 (Claude): SG98.CODE.COVERAGE: unpinned source in the capture is admitted
def test_extra_captured_source_needs_independent_frozen_authorization(gateway, tmp_path):
    wire, a, gw, ledger, root = unstarted(gateway, tmp_path)
    code = invented_code(a.config)
    code['files']['unreviewed.py'] = '5' * 64
    problems = gw.record_start(matching(a.config), S.RunAuthority(a.config, code))
    out = submit(gw)
    assert problems and wire.calls == 0 and out['text'] is None and out['reportable'] is False
    assert ledger.calls == [] and not any(e['kind'] == 'model_start' for e in events(root))


def temporary_capture(tmp_path, monkeypatch, stale_cache=False):
    """Freeze expected source independently, then capture and cold-load a benign temporary module.

    A timestamp-valid old cache deliberately computes2 while frozen source computes1.
    The source and cache remain unchanged after capture. Read-only installation is
    declared as an invented premise, not OS enforcement established by this test.
    """
    package = tmp_path / 'package'; package.mkdir()
    source = package / 'probe.py'; frozen = b'def answer():\n    return 1\n'
    manifest = invented()
    manifest['runtime']['code_sha256'] = {'probe.py': hashlib.sha256(frozen).hexdigest()}
    expected = tmp_path / 'expected'; expected.write_text(F.fingerprint(manifest))
    config = F.checked(manifest, expected)
    stamp = 1700000000
    if stale_cache:
        source.write_bytes(b'def answer():\n    return 2\n'); os.utime(source, (stamp, stamp))
        py_compile.compile(str(source), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    source.write_bytes(frozen); os.utime(source, (stamp, stamp))
    name = 'turn98_temporary_probe'
    assert name not in sys.modules
    cat = C.catalog(package, name, 'invented pre-import time', 'invented immutable installation premise')
    assert cat['phase'] == 'pre-import' and cat['files'] == config.section('runtime')['code_sha256']
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return config, cat, module


# Fixed in TURN 99 (Claude): SG98.CODE.BYTECODE: captured cache lacks independent frozen identity or source equivalence
def test_wrong_timestamp_valid_bytecode_cannot_admit_frozen_source_identity(gateway, tmp_path, monkeypatch):
    config, cat, module = temporary_capture(tmp_path, monkeypatch, stale_cache=True)
    assert cat['bytecode'] and module.answer() == 2
    assert C.verify_loaded(cat, [module]) == [] and C.drift(cat) == []
    evidence = C.evidence(cat, [module])
    assert evidence['problems'] == []
    wire = Wire(); a = adapter(wire, X.ExchangeStore(tmp_path / 'x'), config=config, reportable=True)
    gw, ledger, _, root = gateway(provider=a, start=False)
    problems = gw.record_start(matching(config), S.RunAuthority(config, evidence))
    out = submit(gw)
    # No frozen cache pins or reviewed source-loading/equivalence mechanism were supplied.
    assert problems and wire.calls == 0 and out['text'] is None and out['reportable'] is False
    assert ledger.calls == [] and not any(e['kind'] == 'model_start' for e in events(root))


def test_source_only_cold_load_matches_independently_frozen_source(gateway, tmp_path, monkeypatch):
    config, cat, module = temporary_capture(tmp_path, monkeypatch)
    assert not cat['bytecode'] and module.answer() == 1
    assert C.verify_loaded(cat, [module]) == [] and C.drift(cat) == []
    wire = Wire(); a = adapter(wire, X.ExchangeStore(tmp_path / 'x'), config=config, reportable=True)
    gw, ledger, _, root = gateway(provider=a, start=False)
    assert gw.record_start(matching(config), S.RunAuthority(config, C.evidence(cat, [module]))) == []
    out = submit(gw)
    assert out['status'] == 'answered' and out['reportable'] and wire.calls == 1
    assert ledger.calls[-1].tokens_in == 11 and ledger.calls[-1].tokens_out == 4
    assert verify(root / 'log', root / 'heads')['ok']


def test_unstarted_current_recovery_retains_verified_original_cost_without_text(gateway, tmp_path):
    wire, path, journal, original = first(gateway, tmp_path)
    assert verify(original / 'log', original / 'heads')['ok']
    gw, ledger, _, root = gateway(provider=adapter(wire, X.ExchangeStore(path), reportable=True),
                                journal=journal, start=False)
    out = submit(gw)
    assert out['recovered'] and out['text'] is None and out['reportable'] is False and wire.calls == 1
    assert ledger.calls[-1].tokens_in == 11 and ledger.calls[-1].tokens_out == 4
    assert ledger.calls[-1].wall_s == 2.0
    assert verify(root / 'log', root / 'heads')['ok']
