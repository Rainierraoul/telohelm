"""TURN100 independent roots, persistent drift disposition and availability attribution.

Temporary modules/files, trusted-writer fixture logs and scripted Wire only.
Fault injection does not demonstrate actual OS containment or log forgery.
"""
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys

import pytest

import telohelm.codecapture as C
import telohelm.runtime.exchange as X; import telohelm.runtime.models as M; import telohelm.runtime.startup as S
from telohelm.runtime.trusted_log import TrustedLog; from telohelm.runtime.trusted_log import verify
from .test_exchange_join import Wire; from .test_exchange_join import adapter
from .test_startup import matching
from .test_startup_gate import events
from .test_turn65_review_codex import gateway; from .test_turn65_review_codex import submit  # noqa: F401
from .test_turn98_review_codex import temporary_capture


def captured_run(gateway, tmp_path, monkeypatch):
    config, cat, module = temporary_capture(tmp_path, monkeypatch)
    wire = Wire(); store = tmp_path / 'x'; journal = tmp_path / 'j1'
    clock = iter([10., 12.])
    gw, ledger, _, root = gateway(provider=adapter(wire, X.ExchangeStore(store), config=config, reportable=True),
                                journal=journal, start=False, clock=lambda: next(clock))
    authority = S.RunAuthority(config, C.evidence(cat, [module]), check_drift=True)
    assert gw.record_start(matching(config), authority) == []
    return config, cat, module, wire, gw, ledger, root, store, journal


# Fixed in TURN 101 (Claude): SG100.VOID.ORIGINAL: reuse/recovery ignores original code_identity_void close
@pytest.mark.parametrize('path_used', ['store', 'journal'])
def test_original_void_code_identity_cannot_supply_usable_recovery(gateway, tmp_path, monkeypatch, path_used):
    config, cat, module, wire, gw, ledger, root, store, journal = captured_run(gateway, tmp_path, monkeypatch)
    assert submit(gw)['status'] == 'answered' and ledger.calls[-1].wall_s == 2.
    source = tmp_path / 'package' / 'probe.py'; original = source.read_bytes()
    source.write_bytes(b'def answer():\n    return 3\n')
    assert gw.close() == ['probe.py'] and events(root)[-1]['code_identity_void'] is True
    source.write_bytes(original)
    assert verify(root / 'log', root / 'heads')['ok']
    resumed, costs, _, _ = gateway(provider=adapter(wire, X.ExchangeStore(store), config=config, reportable=True),
                                  journal=journal if path_used == 'journal' else tmp_path / 'j2')
    out = submit(resumed)
    assert out['text'] is None and out['reportable'] is False and wire.calls == 1
    assert costs.calls[-1].tokens_in == 11 and costs.calls[-1].tokens_out == 4 and costs.calls[-1].wall_s == 2.


@pytest.mark.parametrize('path_used', [
    'store',  # Fixed in TURN 101 (Claude): COST100.REUSE.TIME: verified original duration discarded on normal store reuse
    'journal',
])
def test_cleanly_closed_original_still_supplies_verified_recovery(gateway, tmp_path, monkeypatch, path_used):
    config, cat, module, wire, gw, ledger, root, store, journal = captured_run(gateway, tmp_path, monkeypatch)
    assert submit(gw)['status'] == 'answered' and ledger.calls[-1].wall_s == 2.
    assert gw.close() == [] and events(root)[-1]['code_identity_void'] is False
    provider = adapter(wire, X.ExchangeStore(store), config=config, reportable=True)
    assert provider._original_wall(provider._exchanges.records()[0], (root / 'log', root / 'heads')) == 2.
    resumed, costs, _, _ = gateway(provider=provider,
                                  journal=journal if path_used == 'journal' else tmp_path / 'j2')
    out = submit(resumed)
    assert out['text'] == 'invented answer' and out['reportable'] is True and wire.calls == 1
    assert costs.calls[-1].tokens_in == 11 and costs.calls[-1].tokens_out == 4
    assert costs.calls[-1].wall_s == 2.


# Fixed in TURN 101 (Claude): SG100.VOID.PERSIST: restoration erases a previously detected drift disposition
def test_detected_drift_remains_void_after_file_restoration(gateway, tmp_path, monkeypatch):
    config, cat, module, wire, gw, ledger, root, _, _ = captured_run(gateway, tmp_path, monkeypatch)
    source = tmp_path / 'package' / 'probe.py'; original = source.read_bytes()
    source.write_bytes(b'def answer():\n    return 3\n')
    assert submit(gw)['status'] == 'refused' and wire.calls == 0
    assert events(root)[-1]['problems'] == ['code drift: probe.py']
    source.write_bytes(original)
    # A new independently reviewed bootstrap could start a new run; restoration alone is no such authority.
    out = submit(gw)
    assert out['text'] is None and out['reportable'] is False and wire.calls == 0 and ledger.calls == []
    gw.close()
    assert events(root)[-1]['code_identity_void'] is True


# Fixed in TURN 101 (Claude): SG100.DRIFT.INVENTORY: a newly added unpinned executable cache evades use-time drift
def test_new_unpinned_cache_is_detected_before_use(gateway, tmp_path, monkeypatch):
    config, cat, module, wire, gw, ledger, root, _, _ = captured_run(gateway, tmp_path, monkeypatch)
    source = tmp_path / 'package' / 'probe.py'; original = source.read_bytes(); stamp = 1700000000
    source.write_bytes(b'def answer():\n    return 2\n'); os.utime(source, (stamp, stamp))
    py_compile.compile(str(source), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    source.write_bytes(original); os.utime(source, (stamp, stamp))
    assert cat['bytecode'] == {} and C.drift(cat) == []
    name = 'turn100_late_cache_probe'
    spec = importlib.util.spec_from_file_location(name, source)
    late = importlib.util.module_from_spec(spec); monkeypatch.setitem(sys.modules, name, late)
    spec.loader.exec_module(late)
    assert module.answer() == 1 and late.answer() == 2
    out = submit(gw)
    assert wire.calls == 0 and out['text'] is None and out['reportable'] is False and ledger.calls == []


def logged(gateway, tmp_path):
    wire = Wire(); store = tmp_path / 'x'; journal = tmp_path / 'j1'
    gw, _, _, root = gateway(provider=adapter(wire, X.ExchangeStore(store), reportable=True), journal=journal)
    assert submit(gw)['status'] == 'answered'
    return wire, store, journal, root


# Fixed in TURN 101 (Claude): SG100.ROOT.RECORD: publication appends an unverifiable root to protected registry
def test_registry_record_validates_before_publishing(gateway, tmp_path):
    _, _, _, root = logged(gateway, tmp_path)
    registry = tmp_path / 'registry'
    M.ProtectedRoots.record(registry, root / 'log', root / 'heads'); before = registry.read_bytes()
    bad_log = tmp_path / 'badlog'; bad_heads = tmp_path / 'badheads'
    bad_log.write_bytes((root / 'log').read_bytes().splitlines(keepends=True)[0]); bad_heads.write_bytes((root / 'heads').read_bytes())
    assert not verify(bad_log, bad_heads)['ok']
    with pytest.raises(M.GatewayError):
        M.ProtectedRoots.record(registry, bad_log, bad_heads)
    assert registry.read_bytes() == before
    assert M.ProtectedRoots.restore(registry).trusts(root / 'log', root / 'heads')


def rewritten(root, log, heads, change):
    out = TrustedLog(log, heads, 'invented comparison fixture')
    for event in events(root):
        change(event)
        out.append(event)
    assert verify(log, heads)['ok']
    return log, heads


# Fixed in TURN 101 (Claude): SG100.ATTR.AVAILABLE: comparison omits original availability tags
@pytest.mark.parametrize('field', ['available_second', 'available_step'])
def test_recovery_attribution_rejects_changed_original_availability(gateway, tmp_path, field):
    wire, store, journal, original = logged(gateway, tmp_path)
    gw, _, _, later = gateway(provider=adapter(wire, X.ExchangeStore(store), reportable=True), journal=journal)
    assert submit(gw)['recovered'] and wire.calls == 1
    def change(event):
        if event['kind'] == 'model_start':
            assert field in event
            event[field] += 99
    altered = rewritten(later, tmp_path / 'altered-log', tmp_path / 'altered-heads', change)
    assert M.compare_attribution((original / 'log', original / 'heads'), altered)


def test_registry_restoration_runs_in_a_fresh_interpreter(gateway, tmp_path):
    _, _, _, root = logged(gateway, tmp_path)
    registry = tmp_path / 'registry'; M.ProtectedRoots.record(registry, root / 'log', root / 'heads')
    program = 'import json,sys\nfrom pathlib import Path\nimport telohelm.runtime.models as M\nlog,heads,registry=map(Path,sys.argv[1:])\nassert not M.PROTECTED_ROOTS.trusts(log,heads)\nrestored=M.ProtectedRoots.restore(registry)\nassert restored.trusts(log,heads)\nassert not restored.trusts(log.with_name("unregistered"),heads)\nprint(json.dumps({"restored":True,"preexisting_global_trust":False}))\n'
    done = subprocess.run([sys.executable, '-c', program, str(root / 'log'), str(root / 'heads'), str(registry)],
                          cwd=next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'dcarr-runtime-0.1', check=True, capture_output=True, text=True)
    assert json.loads(done.stdout) == {'restored': True, 'preexisting_global_trust': False}


def test_attribution_rejects_a_changed_original_issue_tag(gateway, tmp_path):
    wire, store, journal, original = logged(gateway, tmp_path)
    gw, _, _, later = gateway(provider=adapter(wire, X.ExchangeStore(store), reportable=True), journal=journal)
    assert submit(gw)['recovered']
    def change(event):
        if event['kind'] == 'model_start':
            event['request']['sim_second'] += 1
    altered = rewritten(later, tmp_path / 'altered-log', tmp_path / 'altered-heads', change)
    assert M.compare_attribution((original / 'log', original / 'heads'), altered)
