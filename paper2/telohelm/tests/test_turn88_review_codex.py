"""TURN88: independent original-evidence admission controls; invented data only.

These are trusted-caller/configuration and unprotected-file fault controls, not a
same-process security boundary, actual sandbox access or real model evidence.
"""
import base64
import hashlib
import json

import pytest

import telohelm.runtime.exchange as X; import telohelm.runtime.models as M
from telohelm.runtime.trusted_log import verify
from .test_exchange_join import Wire; from .test_exchange_join import adapter
from .test_local_adapter import ok_body
from .test_transport import http
from .test_manifest import invented
from .test_turn65_review_codex import gateway; from .test_turn65_review_codex import submit  # noqa: F401
from .test_turn84_review_codex import req
from .test_turn86_review_codex import ObservedWire; from .test_turn86_review_codex import config_from


def mark(path):
    rec = json.loads(path.read_text())
    path.with_name(path.name + '.anchored').write_text(json.dumps({
        'exchange_id': rec['exchange_id'], 'record_sha256': X.record_sha256(rec)}) + '\n')


def cut_before_anchor(gateway, tmp_path, monkeypatch):
    wire, path = Wire(), tmp_path / 'x'
    a = adapter(wire, X.ExchangeStore(path), reportable=True)
    gw, _, _, root = gateway(provider=a, journal=tmp_path / 'j1')
    def stop(_):
        raise SystemExit('invented cut after store before protected original anchor')
    monkeypatch.setattr(a, '_anchor', stop)
    with pytest.raises(SystemExit):
        submit(gw)
    assert wire.calls == 1 and verify(root / 'log', root / 'heads')['ok']
    return wire, path, root


# Fixed in TURN 89 (Claude): CFG88.SNAPSHOT: replacement checked config retains old cached digest
def test_replacing_checked_snapshot_cannot_send_new_seed_under_old_digest(tmp_path):
    wire = ObservedWire(); a = adapter(wire, X.ExchangeStore(tmp_path / 'x'))
    old = a.config.fingerprint
    manifest = invented(); manifest['sampling']['seed'] = 23
    replacement = config_from(tmp_path, manifest)
    try:
        a.config = replacement
    except (AttributeError, TypeError):
        pass
    response = a.complete(req())
    if wire.calls == 0:
        assert response.status == 'failed'
        return
    rec = a._exchanges.records()[0]
    active = a.config
    assert replacement.fingerprint != old
    assert wire.requests[0][1].get('seed') == active.section('sampling')['seed']
    assert rec['config_sha256'] == active.fingerprint
    assert response.provider_ids['config_sha256'] == active.fingerprint


# Fixed in TURN 89 (Claude): CFG88.SNAPSHOT: mutable digest label bypasses checked config
def test_cached_digest_cannot_be_relabelled_independently_of_checked_snapshot(tmp_path):
    wire = Wire(); a = adapter(wire, X.ExchangeStore(tmp_path / 'x'))
    original = a.config.fingerprint
    try:
        a.config_sha256 = '0' * 64
    except (AttributeError, TypeError):
        pass
    response = a.complete(req())
    if wire.calls == 0:
        assert response.status == 'failed'
        return
    assert a._exchanges.records()[0]['config_sha256'] == original
    assert response.provider_ids['config_sha256'] == original


# Fixed in TURN 89 (Claude): ANCH88.PREUSE: self-consistent marker substitutes for missing original anchor
def test_self_consistent_marker_does_not_authenticate_missing_original_receipt(gateway, tmp_path, monkeypatch):
    wire, path, original = cut_before_anchor(gateway, tmp_path, monkeypatch)
    mark(path)  # controlled write to unprotected sidecar; protected original remains untouched
    store = X.ExchangeStore(path)
    gw, ledger, _, resumed = gateway(provider=adapter(wire, store, reportable=True), journal=tmp_path / 'j2')
    out = submit(gw)
    problems = M.verify_anchors(store, [(original / 'log', original / 'heads'),
                                      (resumed / 'log', resumed / 'heads')])
    # Adapted in TURN 93 (Claude): Codex's COST92.UNVERIFIED makes the counts of bytes that were never admitted
    # unknown unless a protected original recorded them; this receipt was cut before any protected record.
    assert problems and wire.calls == 1 and ledger.calls[-1].tokens_in is None
    assert out['status'] != 'answered' and out['text'] is None and out['reportable'] is False


# Fixed in TURN 89 (Claude): ANCH88.PREUSE: coherent receipt/marker rewrite is usable before original verification
def test_coherent_receipt_and_marker_rewrite_cannot_release_changed_action_text(gateway, tmp_path):
    wire, path = Wire(), tmp_path / 'x'
    first, _, _, original = gateway(provider=adapter(wire, X.ExchangeStore(path), reportable=True), journal=tmp_path / 'j1')
    assert submit(first)['status'] == 'answered'
    rec = json.loads(path.read_text()); body = ok_body()
    body['choices'][0]['message']['content'] = 'altered invented action'
    raw = http(json.dumps(body).encode())
    rec.update(reply_wire=base64.b64encode(raw).decode(), reply_sha256=hashlib.sha256(raw).hexdigest(), reply_bytes=len(raw))
    path.write_text(json.dumps(rec) + '\n'); mark(path)
    assert verify(original / 'log', original / 'heads')['ok']
    store = X.ExchangeStore(path)
    assert M.verify_anchors(store, [(original / 'log', original / 'heads')])
    gw, _, _, _ = gateway(provider=adapter(wire, store, reportable=True), journal=tmp_path / 'j2')
    out = submit(gw)
    assert wire.calls == 1
    assert out['text'] is None and out['reportable'] is False and out['status'] != 'answered'


# Fixed in TURN 89 (Claude): ANCH88.PREUSE: invalid protected-original log does not gate store recovery
def test_original_log_verification_failure_gates_store_recovery(gateway, tmp_path):
    wire, path = Wire(), tmp_path / 'x'
    first, _, _, original = gateway(provider=adapter(wire, X.ExchangeStore(path), reportable=True), journal=tmp_path / 'j1')
    submit(first)
    lines = (original / 'log').read_text().splitlines()
    (original / 'log').write_text('\n'.join(lines[:-1]) + '\n')  # heads untouched
    assert not verify(original / 'log', original / 'heads')['ok']
    store = X.ExchangeStore(path)
    assert M.verify_anchors(store, [(original / 'log', original / 'heads')])
    gw, _, _, _ = gateway(provider=adapter(wire, store, reportable=True), journal=tmp_path / 'j2')
    out = submit(gw)
    assert wire.calls == 1
    assert out['text'] is None and out['reportable'] is False and out['status'] != 'answered'


# Fixed in TURN 89 (Claude): ANCH88.JOURNAL: recovery bypasses protected original/extraction admission
def test_changed_journal_text_is_quarantined_before_recovery_output(gateway, tmp_path):
    wire, path, journal = Wire(), tmp_path / 'x', tmp_path / 'j'
    a = adapter(wire, X.ExchangeStore(path), reportable=True)
    gw, _, _, original = gateway(provider=a, journal=journal); submit(gw)
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    rows[0]['response']['text'] = 'altered invented journal action'
    journal.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    assert verify(original / 'log', original / 'heads')['ok']
    b = adapter(wire, X.ExchangeStore(path), reportable=True)
    assert M.verify_exchange_evidence(b, b._exchanges, journal, [(original / 'log', original / 'heads')])
    resumed, fresh, _, _ = gateway(provider=b, journal=journal)
    out = submit(resumed)
    assert wire.calls == 1 and sum(c.tokens_in or 0 for c in fresh.calls) == 11
    assert out['text'] is None and out['reportable'] is False and out['status'] != 'answered'


def test_nominal_seed_body_and_complete_original_join(gateway, tmp_path):
    manifest = invented(); manifest['sampling']['seed'] = 23
    c = config_from(tmp_path, manifest); wire = ObservedWire(); store = X.ExchangeStore(tmp_path / 'x')
    a = adapter(wire, store, config=c, reportable=True)
    gw, ledger, _, root = gateway(provider=a); out = submit(gw)
    assert out['status'] == 'answered' and out['reportable'] is True
    assert wire.requests[0][1]['seed'] == 23 and store.records()[0]['config_sha256'] == c.fingerprint
    assert M.verify_exchange_evidence(a, store, root / 'journal', [(root / 'log', root / 'heads')]) == []
    assert sum(c.tokens_in or 0 for c in ledger.calls) == 11


def test_unmarked_original_receipt_retains_cost_and_quarantine(gateway, tmp_path, monkeypatch):
    wire, path, original = cut_before_anchor(gateway, tmp_path, monkeypatch)
    b = adapter(wire, X.ExchangeStore(path), reportable=True)
    gw, ledger, _, _ = gateway(provider=b, journal=tmp_path / 'j2'); out = submit(gw)
    assert out['status'] == 'failed' and out['text'] is None and out['reportable'] is False
    # Adapted in TURN 93 (Claude): under COST92.UNVERIFIED the unanchored receipt's own counts (11/4) are claims, kept
    # in its provider identifiers, and its charged usage is unknown, since no protected record holds it.
    assert wire.calls == 1 and ledger.calls[-1].tokens_in is None and ledger.calls[-1].tokens_out is None
    assert ledger.calls[-1].wall_s is None
    ids = json.loads((tmp_path / 'j2').read_text().splitlines()[0])['response']['provider_ids']
    assert (ids['stored_tokens_in'], ids['stored_tokens_out'], ids['usage_source']) == ('11', '4', 'unknown')
    again, _, _, _ = gateway(provider=b, journal=tmp_path / 'j2', ledger=ledger)
    assert submit(again)['recovered'] and wire.calls == 1
    assert sum(c.tokens_in or 0 for c in ledger.calls) == 0


def test_valid_original_store_replay_preserves_unknown_total_and_once_charge(gateway, tmp_path):
    wire, path = Wire(), tmp_path / 'x'
    first, first_ledger, _, original = gateway(provider=adapter(wire, X.ExchangeStore(path), reportable=True),
                                               journal=tmp_path / 'j1')
    assert submit(first)['status'] == 'answered'
    unused = Wire(); store = X.ExchangeStore(path); b = adapter(unused, store, reportable=True)
    replay, ledger, _, resumed = gateway(provider=b, journal=tmp_path / 'j2', replay=True)
    out = submit(replay)
    assert out['status'] == 'answered' and unused.calls == 0 and wire.calls == 1
    # Adapted in TURN 101 (Claude): the original completed, so a store reuse now keeps its verified total time from
    # the protected log (Codex's COST100.REUSE.TIME); time stays unknown only without a valid original completion.
    assert ledger.calls[-1].wall_s == first_ledger.calls[-1].wall_s and sum(c.tokens_in or 0 for c in ledger.calls) == 11
    assert M.verify_exchange_evidence(b, store, tmp_path / 'j2', [(original / 'log', original / 'heads'),
                                                              (resumed / 'log', resumed / 'heads')]) == []
