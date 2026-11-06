"""TURN102 exact repair review: typed attribution and invalidation publication cuts.

Scripted Wire, temporary modules and trusted-writer fixture logs only. An injected
append error is a CPU API cut, not a demonstration of power-loss/OS durability.
"""
import pytest

import telohelm.runtime.exchange as X; import telohelm.runtime.models as M
from telohelm.runtime.trusted_log import verify
from .test_exchange_join import adapter
from .test_startup_gate import events
from .test_turn65_review_codex import gateway; from .test_turn65_review_codex import submit  # noqa: F401
from .test_turn100_review_codex import captured_run; from .test_turn100_review_codex import logged; from .test_turn100_review_codex import rewritten


# Fixed in TURN 103 (Claude): ATTR102.TYPE: numeric equality admits float availability in place of integer tags
@pytest.mark.parametrize('field', ['available_second', 'available_step'])
def test_original_availability_comparison_requires_integer_type(gateway, tmp_path, field):
    wire, store, journal, original = logged(gateway, tmp_path)
    gw, _, _, later = gateway(provider=adapter(wire, X.ExchangeStore(store), reportable=True), journal=journal)
    assert submit(gw)['recovered'] and wire.calls == 1
    def change(event):
        if event['kind'] == 'model_start':
            assert type(event[field]) is int
            event[field] = float(event[field])
    altered = rewritten(later, tmp_path / 'altered-log', tmp_path / 'altered-heads', change)
    assert M.compare_attribution((original / 'log', original / 'heads'), altered)


# Fixed in TURN 103 (Claude): ATTR102.REQUIRED: mutually absent availability fields are treated as an original match
@pytest.mark.parametrize('field', ['available_second', 'available_step'])
def test_missing_original_and_later_availability_is_not_a_match(gateway, tmp_path, field):
    wire, store, journal, original = logged(gateway, tmp_path)
    gw, _, _, later = gateway(provider=adapter(wire, X.ExchangeStore(store), reportable=True), journal=journal)
    assert submit(gw)['recovered'] and wire.calls == 1
    def change(event):
        if event['kind'] == 'model_start':
            assert field in event
            event.pop(field)
    missing_original = rewritten(original, tmp_path / 'missing-original', tmp_path / 'missing-original-heads', change)
    missing_later = rewritten(later, tmp_path / 'missing-later', tmp_path / 'missing-later-heads', change)
    assert M.compare_attribution(missing_original, missing_later)


# Fixed in TURN 103 (Claude): VOID102.WRITE: failed invalidation publication leaves original admission reusable
@pytest.mark.parametrize('path_used', ['store', 'journal'])
def test_invalidation_write_cut_cannot_reauthorize_the_original(gateway, tmp_path, monkeypatch, path_used):
    config, cat, module, wire, gw, ledger, root, store, journal = captured_run(gateway, tmp_path, monkeypatch)
    assert submit(gw)['status'] == 'answered' and ledger.calls[-1].wall_s == 2.
    source = tmp_path / 'package' / 'probe.py'; original = source.read_bytes()
    source.write_bytes(b'def answer():\n    return 3\n')
    real_append = gw._log.append
    def interrupted(payload):
        if payload.get('kind') == 'code_identity_void':
            raise OSError('invented failure before invalidation publication')
        return real_append(payload)
    monkeypatch.setattr(gw._log, 'append', interrupted)
    with pytest.raises(OSError):
        submit(gw)
    assert gw._void is True and not any(e['kind'] == 'code_identity_void' for e in events(root))
    assert verify(root / 'log', root / 'heads')['ok'] and wire.calls == 1
    source.write_bytes(original)
    assert submit(gw)['status'] == 'refused' and wire.calls == 1
    assert events(root)[-1]['kind'] == 'startup_refused'
    assert events(root)[-1]['problems'] == ['code identity void']
    resumed, costs, _, _ = gateway(provider=adapter(wire, X.ExchangeStore(store), config=config, reportable=True),
                                  journal=journal if path_used == 'journal' else tmp_path / 'j2')
    out = submit(resumed)
    assert out['text'] is None and out['reportable'] is False and wire.calls == 1
    assert costs.calls[-1].tokens_in == 11 and costs.calls[-1].tokens_out == 4 and costs.calls[-1].wall_s == 2.


def test_durable_original_void_quarantines_shared_ledger_recovery_without_double_charge(gateway, tmp_path, monkeypatch):
    config, cat, module, wire, gw, ledger, root, store, journal = captured_run(gateway, tmp_path, monkeypatch)
    assert submit(gw)['status'] == 'answered'
    source = tmp_path / 'package' / 'probe.py'; original = source.read_bytes()
    source.write_bytes(b'def answer():\n    return 3\n')
    assert submit(gw)['status'] == 'refused'
    assert any(e['kind'] == 'code_identity_void' for e in events(root))
    source.write_bytes(original)
    resumed, costs, _, later = gateway(provider=adapter(wire, X.ExchangeStore(store), config=config, reportable=True),
                                      journal=journal, ledger=ledger)
    out = submit(resumed)
    assert out['recovered'] and out['text'] is None and out['reportable'] is False and wire.calls == 1
    assert sum(c.tokens_in or 0 for c in costs.calls) == 11 and sum(c.tokens_out or 0 for c in costs.calls) == 4
    assert costs.calls[0].wall_s == 2.
    assert verify(root / 'log', root / 'heads')['ok'] and verify(later / 'log', later / 'heads')['ok']
