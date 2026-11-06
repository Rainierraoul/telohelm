"""TURN90 independent admission-root, journal-completeness and cost controls.

Invented file/configuration/byte faults only. Counterfeit heads are ordinary
fixture files, never the original protected heads or a claimed sandbox attack.
"""
import base64
import hashlib
import json

import pytest

import telohelm.runtime.exchange as X; import telohelm.runtime.models as M
from telohelm.runtime.trusted_log import TrustedLog; from telohelm.runtime.trusted_log import verify
from .test_exchange_join import Wire; from .test_exchange_join import adapter; from .test_exchange_join import journal_rows
from .test_local_adapter import ok_body
from .test_transport import http
from .test_turn65_review_codex import gateway; from .test_turn65_review_codex import submit  # noqa: F401
from .test_turn84_review_codex import req


def first(gateway, tmp_path):
    wire = Wire(); path = tmp_path / 'x'; journal = tmp_path / 'j1'
    clock = iter([10., 12.])
    gw, ledger, _, root = gateway(provider=adapter(wire, X.ExchangeStore(path), reportable=True),
                                  journal=journal, clock=lambda: next(clock))
    assert submit(gw)['status'] == 'answered' and ledger.calls[-1].wall_s == 2.
    return wire, path, journal, root


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


# Fixed in TURN 91 (Claude): ADM90.REQUIRED: missing exchange identity skips recovery admission
@pytest.mark.parametrize('erase', ['exchange_id', 'provider_ids'])
def test_missing_original_receipt_identifier_does_not_bypass_admission(gateway, tmp_path, erase):
    wire, path, journal, original = first(gateway, tmp_path)
    rows = journal_rows(journal)
    rows[0]['response']['text'] = 'altered invented action with erased receipt link'
    if erase == 'exchange_id':
        rows[0]['response']['provider_ids'].pop('exchange_id')
    else:
        rows[0]['response']['provider_ids'] = None
    write_rows(journal, rows)
    assert verify(original / 'log', original / 'heads')['ok']
    gw, _, _, _ = gateway(provider=adapter(wire, X.ExchangeStore(path), reportable=True), journal=journal)
    out = submit(gw)
    assert wire.calls == 1
    assert out['text'] is None and out['status'] != 'answered' and out['reportable'] is False


# Fixed in TURN 91 (Claude): ADM90.ROOT: unprotected pointer chooses its own allegedly protected heads
def test_unprotected_marker_cannot_select_counterfeit_authority(gateway, tmp_path):
    wire, path, journal, original = first(gateway, tmp_path)
    protected_original = (original / 'heads').read_bytes()
    rec = json.loads(path.read_text()); body = ok_body()
    body['choices'][0]['message']['content'] = 'altered invented action from counterfeit authority'
    raw = http(json.dumps(body).encode())
    rec.update(reply_wire=base64.b64encode(raw).decode(), reply_sha256=hashlib.sha256(raw).hexdigest(), reply_bytes=len(raw))
    path.write_text(json.dumps(rec) + '\n')
    # Ordinary unprotected fixture files holding a self-consistent chain/heads.
    # The reader's independent trusted authority must exclude these locations.
    fake, fake_heads = tmp_path / 'untrusted-log', tmp_path / 'untrusted-heads'
    writer = TrustedLog(fake, fake_heads, 'counterfeit invented source')
    events = [json.loads(line)['payload'] for line in (original / 'log').read_text().splitlines()[1:]]
    writer.append(next(event for event in events if event['kind'] == 'model_start'))
    anchor = next(event for event in events if event['kind'] == 'exchange_anchor')
    summary = X.summary(rec)
    writer.append({**anchor, 'record': summary, 'record_sha256': X.record_sha256(summary)})
    assert verify(fake, fake_heads)['ok']  # self-consistency, not authenticity
    marks = path.with_name(path.name + '.anchored')
    marks.write_text(json.dumps({'exchange_id': rec['exchange_id'], 'record_sha256': X.record_sha256(rec),
                                'log': str(fake), 'heads': str(fake_heads)}) + '\n')
    assert (original / 'heads').read_bytes() == protected_original and verify(original / 'log', original / 'heads')['ok']
    store = X.ExchangeStore(path)
    assert M.verify_anchors(store, [(original / 'log', original / 'heads')])  # real original rejects changed record
    gw, _, _, _ = gateway(provider=adapter(wire, store, reportable=True), journal=tmp_path / 'j2')
    out = submit(gw)
    assert wire.calls == 1
    assert out['text'] is None and out['status'] != 'answered' and out['reportable'] is False


# Fixed in TURN 91 (Claude): COST90.SOURCE: quarantined journal counts override verified original usage
@pytest.mark.parametrize('invented_usage', [0, 99])
def test_quarantine_keeps_verified_usage_instead_of_changed_journal_counts(gateway, tmp_path, invented_usage):
    wire, path, journal, original = first(gateway, tmp_path)
    rows = journal_rows(journal); rows[0]['response'].update(tokens_in=invented_usage, tokens_out=invented_usage)
    write_rows(journal, rows)
    store = X.ExchangeStore(path)
    assert M.verify_anchors(store, [(original / 'log', original / 'heads')]) == []
    gw, ledger, _, _ = gateway(provider=adapter(wire, store, reportable=True), journal=journal)
    out = submit(gw)
    assert out['text'] is None and out['reportable'] is False and wire.calls == 1
    assert sum(c.tokens_in or 0 for c in ledger.calls) == 11
    assert sum(c.tokens_out or 0 for c in ledger.calls) == 4


# Fixed in TURN 91 (Claude): COST90.TIME: unprotected completion time overrides protected original completion
def test_changed_journal_duration_cannot_replace_original_protected_total(gateway, tmp_path):
    wire, path, journal, original = first(gateway, tmp_path)
    rows = journal_rows(journal)
    for row in rows:
        if row['type'] == 'completion': row['wall_s'] = 99.
    write_rows(journal, rows)
    assert verify(original / 'log', original / 'heads')['ok']
    gw, ledger, _, _ = gateway(provider=adapter(wire, X.ExchangeStore(path), reportable=True), journal=journal)
    submit(gw)
    assert wire.calls == 1
    # Verified original2.0 or explicit unavailable time; never the changed99.
    assert ledger.calls[-1].wall_s in (2., None)


# Fixed in TURN 91 (Claude): ADM90.DIRECT: test-only direct path is not enforced before usable reportable text
def test_reportable_direct_call_without_gateway_is_quarantined(tmp_path):
    wire = Wire(); store = X.ExchangeStore(tmp_path / 'x')
    response = adapter(wire, store, reportable=True).complete(req())
    assert wire.calls <= 1  # refusal before contact or retained unanchored receipt are both acceptable
    assert response.text is None and response.status != 'answered'
    assert response.origin is None or response.origin.get('reportable') is False


def test_original_protected_receipt_and_exact_journal_recover_known_cost(gateway, tmp_path):
    wire, path, journal, original = first(gateway, tmp_path)
    a = adapter(wire, X.ExchangeStore(path), reportable=True)
    gw, ledger, _, resumed = gateway(provider=a, journal=journal)
    out = submit(gw)
    assert out['status'] == 'answered' and out['reportable'] is True and out['text'] == 'invented answer'
    assert wire.calls == 1 and sum(c.tokens_in or 0 for c in ledger.calls) == 11
    assert sum(c.tokens_out or 0 for c in ledger.calls) == 4 and ledger.calls[-1].wall_s == 2.
    assert M.verify_exchange_evidence(a, a._exchanges, journal, [(original / 'log', original / 'heads'),
                                                                (resumed / 'log', resumed / 'heads')]) == []


def test_original_paths_reuse_without_extra_wire_and_keep_unknown_total(gateway, tmp_path):
    wire, path, _, original = first(gateway, tmp_path)
    store = X.ExchangeStore(path); a = adapter(wire, store, reportable=True)
    gw, ledger, _, resumed = gateway(provider=a, journal=tmp_path / 'j2')
    out = submit(gw)
    assert out['text'] == 'invented answer' and out['reportable'] is True and wire.calls == 1
    # Adapted in TURN 101 (Claude): a completed original's verified 2.0 is kept on store reuse (COST100.REUSE.TIME).
    assert sum(c.tokens_in or 0 for c in ledger.calls) == 11 and ledger.calls[-1].wall_s == 2.
    assert M.verify_anchors(store, [(original / 'log', original / 'heads'), (resumed / 'log', resumed / 'heads')]) == []
