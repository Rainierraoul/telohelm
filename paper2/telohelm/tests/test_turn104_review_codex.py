"""TURN104 independent CPU review. Invented trusted-writer logs and append cuts.

These controls establish no model competence, OS protection or power-loss durability.
"""
import pytest

import telohelm.runtime.exchange as X; import telohelm.runtime.models as M
from telohelm.runtime.trusted_log import verify
from .test_exchange_join import adapter
from .test_startup_gate import events
from .test_turn65_review_codex import gateway; from .test_turn65_review_codex import submit  # noqa: F401
from .test_turn100_review_codex import captured_run; from .test_turn100_review_codex import logged; from .test_turn100_review_codex import rewritten


# Fixed in TURN 105 (Claude): ATTR104.DOMAIN: mutually negative latency is accepted as an original match
def test_mutually_negative_latency_does_not_establish_attribution(gateway, tmp_path):
    wire, store, journal, original = logged(gateway, tmp_path)
    gw, _, _, later = gateway(provider=adapter(wire, X.ExchangeStore(store), reportable=True), journal=journal)
    assert submit(gw)['recovered'] and wire.calls == 1
    def change(event):
        if event['kind'] == 'model_start':
            event['request']['modelled_latency_s'] = -1.0
    bad_original = rewritten(original, tmp_path / 'o-log', tmp_path / 'o-heads', change)
    bad_later = rewritten(later, tmp_path / 'l-log', tmp_path / 'l-heads', change)
    assert M.compare_attribution(bad_original, bad_later)


@pytest.mark.parametrize('detection', ['submit', 'close'])
def test_failed_publication_keeps_refusing_then_seals_once(gateway, tmp_path, monkeypatch, detection):
    config, cat, module, wire, gw, ledger, root, store, journal = captured_run(gateway, tmp_path, monkeypatch)
    assert submit(gw)['status'] == 'answered' and ledger.calls[-1].wall_s == 2.0
    source = tmp_path / 'package' / 'probe.py'; saved = source.read_bytes()
    source.write_bytes(b'def answer():\n    return 3\n')
    real = gw._log.append
    def fail(payload):
        if payload.get('kind') in ('code_identity_void', 'run_close'):
            raise OSError('invented pre-append cut')
        return real(payload)
    monkeypatch.setattr(gw._log, 'append', fail)
    with pytest.raises(OSError):
        submit(gw) if detection == 'submit' else gw.close()
    source.write_bytes(saved)
    for _ in range(3):
        assert submit(gw)['status'] == 'refused'
    assert wire.calls == 1 and len(ledger.calls) == 1
    assert not any(e['kind'] == 'code_identity_void' for e in events(root))
    assert sum(e['kind'] == 'startup_refused' for e in events(root)) == 3
    assert verify(root / 'log', root / 'heads')['ok']
    # The surviving protected refusal independently quarantines a new run.
    resumed, costs, _, later = gateway(provider=adapter(wire, X.ExchangeStore(store), config=config, reportable=True),
                                      journal=journal, ledger=ledger)
    out = submit(resumed)
    assert out['recovered'] and out['text'] is None and not out['reportable']
    assert wire.calls == 1 and sum(c.tokens_in or 0 for c in costs.calls) == 11
    assert sum(c.tokens_out or 0 for c in costs.calls) == 4 and costs.calls[0].wall_s == 2.0
    assert verify(later / 'log', later / 'heads')['ok']
    monkeypatch.setattr(gw._log, 'append', real)
    for _ in range(2):
        assert submit(gw)['status'] == 'refused'
    assert sum(e['kind'] == 'code_identity_void' for e in events(root)) == 1
    assert wire.calls == 1 and verify(root / 'log', root / 'heads')['ok']
