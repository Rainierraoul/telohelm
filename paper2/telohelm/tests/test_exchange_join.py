"""The draft exchange store and its join to the adapter and gateway (TURN 83). Invented transports and, in one test, an
invented server on 127.0.0.1. No model server is contacted."""
from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import threading
from pathlib import Path

import pytest

import telohelm.runtime.manifest as F
import telohelm.runtime.models as M
from telohelm.runtime.exchange import ExchangeError; from telohelm.runtime.exchange import ExchangeStore; from telohelm.runtime.exchange import record_sha256
from telohelm.runtime.trusted_log import verify
from .test_local_adapter import ROLES; from .test_local_adapter import TIMEOUTS; from .test_local_adapter import ok_body
from .test_manifest import invented
from .test_transport import Invented; from .test_transport import http
from .test_turn65_review_codex import gateway; from .test_turn65_review_codex import submit  # noqa: F401  (fixture)

CONFIG = "c" * 64                     # a fingerprint for store-level tests, which take a digest


def checked(**kw):
    """A configuration made by manifest.checked() from a complete invented manifest (TURN 85, Codex's MF84.GATE)."""
    m = invented(**kw)
    path = Path(tempfile.mkdtemp()) / "expected"
    path.write_text(F.fingerprint(m))
    return F.checked(m, path)


class Wire:
    """An invented transport that returns exchange evidence for scripted reply bytes and counts its exchanges."""

    origin, max_response_bytes = "http://127.0.0.1:9", 1 << 20       # bound to the configuration (MF84.GATE)

    def __init__(self, reply=None, outcome="received", complete=True):
        self.reply = http(json.dumps(ok_body()).encode()) if reply is None else reply
        self.outcome, self.complete, self.calls = outcome, complete, 0

    def exchange(self, url, body, timeout_s):
        self.calls += 1
        request = b"POST /v1/chat/completions HTTP/1.0\r\n\r\n" + json.dumps(body, sort_keys=True).encode()
        return {"url": url, "outcome": self.outcome, "request_wire": request, "reply_wire": self.reply,
                "reply_complete": self.complete, "elapsed_s": 0.01}

    def __call__(self, *a):
        raise AssertionError("with a store the adapter asks for evidence, not a parsed reply")


def adapter(wire, store, **kw):
    """A store-backed adapter. config may be a CheckedConfig; anything else (a bare digest, say) is passed through so
    that tests can show it is refused."""
    config = kw.pop("config", None)
    return M.LocalChatProvider("http://127.0.0.1:9", ROLES, wire, TIMEOUTS, exchanges=store,
                               config=checked() if config is None else config, **kw)


def journal_rows(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def test_the_exchange_is_saved_before_parsing_and_linked_through_the_gateway(gateway, tmp_path):
    wire, store = Wire(), ExchangeStore(tmp_path / "exchanges")
    a = adapter(wire, store, reportable=True)
    gw, ledger, _, root = gateway(provider=a, journal=tmp_path / "journal")
    out = submit(gw)
    assert out["status"] == "answered" and out["text"] == "invented answer" and out["reportable"] is True
    [rec] = store.records()
    assert rec["outcome"] == "received" and rec["reply_complete"] and rec["reply_wire"] == wire.reply
    ids = journal_rows(tmp_path / "journal")[0]["response"]["provider_ids"]
    assert ids["exchange_id"] == rec["exchange_id"] and ids["wire_sha256"] == rec["reply_sha256"]
    assert ids["config_sha256"] == a.config_sha256 and ids["wire_complete"] == "true" and ids["wire_outcome"] == "received"
    events = [json.loads(l)["payload"] for l in (root / "log").read_text().splitlines()[1:]]
    end = events[-1]
    assert end["kind"] == "model_end" and end["response"]["provider_ids"]["wire_sha256"] == rec["reply_sha256"]
    anchor = next(e for e in events if e["kind"] == "exchange_anchor")
    assert anchor["use"] == "saved" and anchor["record"]["reply_sha256"] == rec["reply_sha256"]
    assert anchor["record"]["request"] == rec["request"] and rec["anchored"] is True     # TURN 87: the whole record
    assert events.index(anchor) < events.index(end)
    assert verify(root / "log", root / "heads")["ok"]
    assert M.verify_extraction(a, store, tmp_path / "journal") == []
    assert M.verify_anchors(store, [(root / "log", root / "heads")]) == []


def test_a_complete_wire_reply_can_still_be_unusable(tmp_path):
    wire = Wire(reply=b"HTTP/1.0 200 OK\r\nContent-Length: 99\r\n\r\n{}")
    store = ExchangeStore(tmp_path / "x")
    r = adapter(wire, store).complete(M.ModelRequest('["s"]', "attempt1", "e", "c", "s", "small_model",
                                                     (("user", "q"),), 64, 1, 0, 0, 2.4))
    assert r.status == "failed" and r.text is None
    assert r.provider_ids["wire_complete"] == "true" and store.records()[0]["reply_complete"] is True


@pytest.mark.parametrize("outcome,status", [("deadline", "timed_out"), ("transport error", "failed"),
                                            ("refused", "failed")])
def test_a_partial_reply_is_kept_and_marked_incomplete(tmp_path, outcome, status):
    wire = Wire(reply=b"HTTP/1.0 200 OK\r\nCont", outcome=outcome, complete=False)
    store = ExchangeStore(tmp_path / "x")
    r = adapter(wire, store).complete(M.ModelRequest('["s"]', "attempt1", "e", "c", "s", "small_model",
                                                     (("user", "q"),), 64, 1, 0, 0, 2.4))
    assert r.status == status and r.provider_ids["wire_complete"] == "false"
    rec = store.records()[0]
    assert rec["reply_wire"] == b"HTTP/1.0 200 OK\r\nCont" and rec["outcome"] == outcome and not rec["reply_complete"]


def test_a_failed_save_fails_the_call_and_is_never_asked_again(gateway, tmp_path, monkeypatch):
    wire, store = Wire(), ExchangeStore(tmp_path / "exchanges")
    def broken(*a, **kw):
        raise OSError("invented full disk")
    monkeypatch.setattr(store, "save", broken)
    a = adapter(wire, store)
    gw, ledger, _, _ = gateway(provider=a, journal=tmp_path / "journal")
    out = submit(gw)
    assert out["status"] == "failed" and out["text"] is None and wire.calls == 1
    resumed, _, _, _ = gateway(provider=a, ledger=ledger, journal=tmp_path / "journal")
    assert submit(resumed)["recovered"] and wire.calls == 1


def test_a_stop_after_the_exchange_is_saved_recovers_from_the_store(gateway, tmp_path, monkeypatch):
    """The cut between the exchange store and the gateway's journal: the next run has a fresh ledger, the same journal
    and the same store, and the server is not asked again."""
    wire = Wire()
    a = adapter(wire, ExchangeStore(tmp_path / "exchanges"))
    gw, ledger, _, _ = gateway(provider=a, journal=tmp_path / "journal")
    real = gw._write
    def stop_before_journal(rec):
        if rec["type"] == "response":
            raise OSError("invented stop before the journal")
        real(rec)
    monkeypatch.setattr(gw, "_write", stop_before_journal)
    with pytest.raises(OSError):
        submit(gw)
    assert [c.status for c in ledger.calls] == ["interrupted"] and wire.calls == 1
    b = adapter(wire, ExchangeStore(tmp_path / "exchanges"))          # reloaded from disk
    resumed, ledger2, _, _ = gateway(provider=b, journal=tmp_path / "journal")
    out = submit(resumed)
    assert out["status"] == "answered" and wire.calls == 1 and not out["recovered"]
    assert sum(c.tokens_in or 0 for c in ledger2.calls) == 11


def test_a_stop_before_the_charge_is_charged_once_on_recovery(gateway, tmp_path, monkeypatch):
    """The cut between the journal and the ledger: the response is recovered from the journal, the store is not used
    again and the tokens are charged once."""
    wire = Wire()
    a = adapter(wire, ExchangeStore(tmp_path / "exchanges"))
    gw, ledger, _, _ = gateway(provider=a, journal=tmp_path / "journal")
    def stop(call):
        raise OSError("invented stop before the charge")
    monkeypatch.setattr(ledger, "add", stop)
    with pytest.raises(OSError):
        submit(gw)
    resumed, ledger2, _, _ = gateway(provider=a, journal=tmp_path / "journal")
    out = submit(resumed)
    assert out["recovered"] and out["status"] == "answered" and wire.calls == 1
    assert sum(c.tokens_in or 0 for c in ledger2.calls) == 11


def test_a_saved_exchange_replays_with_no_fresh_call(gateway, tmp_path):
    wire = Wire()
    first = adapter(wire, ExchangeStore(tmp_path / "exchanges"))
    gw, _, _, _ = gateway(provider=first, journal=tmp_path / "j1")
    live = submit(gw)
    replayer = adapter(Wire(reply=b"never used"), ExchangeStore(tmp_path / "exchanges"))
    gw2, _, _, _ = gateway(provider=replayer, journal=tmp_path / "j2", replay=True)
    again = submit(gw2)
    assert replayer._transport.calls == 0 and wire.calls == 1
    assert {k: again[k] for k in ("status", "text", "tokens_in", "tokens_out")} == \
           {k: live[k] for k in ("status", "text", "tokens_in", "tokens_out")}


def test_a_stored_exchange_is_never_reused_for_another_body_or_configuration(gateway, tmp_path):
    wire = Wire()
    gw, _, _, _ = gateway(provider=adapter(wire, ExchangeStore(tmp_path / "exchanges")), journal=tmp_path / "j1")
    submit(gw)
    other = Wire()
    m = invented(); m["environment"]["hardware"] = "other invented hardware"
    path = tmp_path / "other-expected"; path.write_text(F.fingerprint(m))
    gw2, _, _, _ = gateway(provider=adapter(other, ExchangeStore(tmp_path / "exchanges"), config=F.checked(m, path)),
                           journal=tmp_path / "j2")
    out = submit(gw2)
    assert out["status"] == "failed" and other.calls == 0


def _store_with_one(tmp_path):
    store = ExchangeStore(tmp_path / "exchanges")
    store.save('["r"]', "attempt1", CONFIG, "small_model", "b" * 64, Wire().exchange("u", {"a": 1}, 1.0))
    return tmp_path / "exchanges"


@pytest.mark.parametrize("damage", ["reply bytes", "digest", "torn line", "repeat", "not json", "complete flag",
                                    "length", "identifier"])
def test_a_damaged_store_fails_closed_on_loading(tmp_path, damage):
    path = _store_with_one(tmp_path)
    line = path.read_bytes().splitlines()[0]
    rec = json.loads(line)
    if damage == "reply bytes":
        raw = bytearray(base64.b64decode(rec["reply_wire"])); raw[0] ^= 1
        rec["reply_wire"] = base64.b64encode(bytes(raw)).decode()
    elif damage == "digest":
        rec["request_sha256"] = "0" * 64
    elif damage == "complete flag":
        rec["outcome"] = "deadline"
    elif damage == "length":
        rec["reply_bytes"] += 1
    elif damage == "identifier":
        rec["exchange_id"] = '["other","attempt1"]'
    if damage == "torn line":
        path.write_bytes(line + b"\n" + line[: len(line) // 2])
    elif damage == "repeat":
        path.write_bytes(line + b"\n" + line + b"\n")
    elif damage == "not json":
        path.write_bytes(b"\xff\xfe\n")
    else:
        path.write_text(json.dumps(rec) + "\n")
    with pytest.raises(ExchangeError):
        ExchangeStore(path)


def test_the_extraction_check_finds_a_journal_that_disagrees_with_its_bytes(gateway, tmp_path):
    store = ExchangeStore(tmp_path / "exchanges")
    a = adapter(Wire(), store)
    gw, _, _, _ = gateway(provider=a, journal=tmp_path / "journal")
    submit(gw)
    rows = journal_rows(tmp_path / "journal")
    good = [dict(r) for r in rows]
    for change, expect in [(lambda r: r["response"].update(text="an altered answer"), "text differs"),
                           (lambda r: r["response"].update(tokens_in=99), "tokens_in differs"),
                           (lambda r: r["response"]["provider_ids"].update(exchange_id='["x","y"]'), "no stored exchange"),
                           (lambda r: r.update(request_id="another"), "belongs to request")]:
        rows = json.loads(json.dumps(good))
        change(rows[0])
        (tmp_path / "journal").write_text("".join(json.dumps(r) + "\n" for r in rows))
        problems = M.verify_extraction(a, store, tmp_path / "journal")
        assert problems and expect in problems[0]


def test_concurrent_saves_are_all_kept_whole(tmp_path):
    store = ExchangeStore(tmp_path / "exchanges")
    def save(i):
        store.save(f'["r{i}"]', "attempt1", CONFIG, "small_model", "b" * 64, Wire().exchange("u", {"i": i}, 1.0))
    threads = [threading.Thread(target=save, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    again = ExchangeStore(tmp_path / "exchanges")
    assert len(again.records()) == 16
    with pytest.raises(ExchangeError):                                  # a repeat is refused, not appended
        save(3)
    assert len(ExchangeStore(tmp_path / "exchanges").records()) == 16


@pytest.mark.parametrize("kw", [{"config": None}, {"config": "c" * 64}, {"config": {"fingerprint": "c" * 64}}])
def test_a_store_needs_a_checked_configuration(tmp_path, kw):
    with pytest.raises(M.GatewayError):
        M.LocalChatProvider("http://127.0.0.1:9", ROLES, Wire(), TIMEOUTS, exchanges=ExchangeStore(tmp_path / "x"), **kw)


def test_construction_fails_closed_without_the_pieces(tmp_path):
    with pytest.raises(M.GatewayError):                                 # reportable needs a store
        M.LocalChatProvider("http://127.0.0.1:9", ROLES, Wire(), TIMEOUTS, reportable=True)
    with pytest.raises(M.GatewayError):                                 # a fingerprint means nothing without one
        M.LocalChatProvider("http://127.0.0.1:9", ROLES, Wire(), TIMEOUTS, config=checked())
    with pytest.raises(M.GatewayError):                                 # the transport must give evidence
        M.LocalChatProvider("http://127.0.0.1:9", ROLES, lambda *a: (200, {}), TIMEOUTS,
                            exchanges=ExchangeStore(tmp_path / "x"), config=checked())


@pytest.mark.parametrize("model", ["invented-other-model", None])
def test_a_reportable_adapter_still_refuses_the_wrong_model(tmp_path, model):
    body = ok_body(); body["model"] = model
    r = adapter(Wire(reply=http(json.dumps(body).encode())), ExchangeStore(tmp_path / "x"), reportable=True).complete(
        M.ModelRequest('["s"]', "attempt1", "e", "c", "s", "small_model", (("user", "q"),), 64, 1, 0, 0, 2.4))
    assert r.status != "answered" and r.text is None


def test_through_a_real_loopback_socket(gateway, tmp_path):
    from telohelm.runtime.transport import LoopbackTransport
    srv = Invented(lambda req: [(http(json.dumps(ok_body()).encode()), 0)])
    try:
        store = ExchangeStore(tmp_path / "exchanges")
        a = M.LocalChatProvider(f"http://127.0.0.1:{srv.port}", ROLES,
                                LoopbackTransport(f"http://127.0.0.1:{srv.port}", 1 << 20), TIMEOUTS,
                                exchanges=store, config=checked(origin=f"http://127.0.0.1:{srv.port}"))
        gw, _, _, _ = gateway(provider=a, journal=tmp_path / "journal")
        assert submit(gw)["text"] == "invented answer"
        rec = store.records()[0]
        assert rec["request_wire"] == srv.requests[0] and rec["reply_complete"]
        assert M.verify_extraction(a, store, tmp_path / "journal") == []
    finally:
        srv.close()


# TURN 85 (Codex's TURN 84 review).

@pytest.mark.parametrize("change", [dict(base="http://127.0.0.1:10"), dict(timeouts={"small_model": 31.0,
                                    "reasoning_model": 600.0}), dict(temperature=0.5), dict(served={"small_model":
                                    ("invented-small", "alias"), "reasoning_model": ("invented-reasoning",)}),
                                    dict(origin="http://127.0.0.1:10"), dict(limit=1 << 10)])
def test_the_adapter_must_match_its_configuration(tmp_path, change):
    wire = Wire()
    if "origin" in change:
        wire.origin = change["origin"]
    if "limit" in change:
        wire.max_response_bytes = change["limit"]
    kw = {k: v for k, v in change.items() if k in ("temperature",)}
    if "served" in change:
        kw["served_names_by_role"] = change["served"]
    with pytest.raises(M.GatewayError):
        M.LocalChatProvider(change.get("base", "http://127.0.0.1:9"), ROLES, wire, change.get("timeouts", TIMEOUTS),
                            exchanges=ExchangeStore(tmp_path / "x"), config=checked(), **kw)


def test_a_request_above_the_configured_output_limit_fails_before_contact(tmp_path):
    wire = Wire()
    r = adapter(wire, ExchangeStore(tmp_path / "x")).complete(
        M.ModelRequest('["s"]', "attempt1", "e", "c", "s", "small_model", (("user", "q"),), 1025, 1, 0, 0, 2.4))
    assert r.status == "failed" and r.provider_ids["refused"] == "outside the configuration" and wire.calls == 0


def test_a_tainted_store_refuses_reads_and_writes_until_reopened(tmp_path, monkeypatch):
    import telohelm.runtime.exchange as X
    store = ExchangeStore(tmp_path / "x")
    def fail(fd):
        raise OSError("invented sync failure")
    monkeypatch.setattr(X.os, "fsync", fail)
    with pytest.raises(OSError):
        store.save('["r"]', "attempt1", CONFIG, "small_model", "b" * 64, Wire().exchange("u", {}, 1.0))
    monkeypatch.undo()
    for call in (lambda: store.get('["r"]', "attempt1", CONFIG, "b" * 64), store.records,
                 lambda: store.save('["q"]', "attempt1", CONFIG, "small_model", "b" * 64, Wire().exchange("u", {}, 1.))):
        with pytest.raises(ExchangeError):
            call()
    assert len(ExchangeStore(tmp_path / "x").records()) == 1           # reopened: the flushed line is committed


def test_an_unanchored_or_altered_exchange_is_found(gateway, tmp_path):
    store = ExchangeStore(tmp_path / "x")
    gw, _, _, root = gateway(provider=adapter(Wire(), store), journal=tmp_path / "j")
    submit(gw)
    store.save('["unanchored"]', "attempt1", CONFIG, "small_model", "b" * 64, Wire().exchange("u", {}, 1.0))
    problems = M.verify_anchors(store, [(root / "log", root / "heads")])
    assert len(problems) == 1 and "unanchored" in problems[0]
    lines = (root / "log").read_text().splitlines()
    (root / "log").write_text("\n".join(lines[:-1]) + "\n")
    assert any("protected" in p or "count" in p for p in M.verify_anchors(store, [(root / "log", root / "heads")]))


def test_a_reused_exchange_is_anchored_as_a_reuse(gateway, tmp_path):
    wire = Wire()
    gw, first_ledger, _, _ = gateway(provider=adapter(wire, ExchangeStore(tmp_path / "x")), journal=tmp_path / "j1")
    submit(gw)
    gw2, ledger, _, root = gateway(provider=adapter(wire, ExchangeStore(tmp_path / "x")), journal=tmp_path / "j2")
    out = submit(gw2)
    events = [json.loads(l)["payload"] for l in (root / "log").read_text().splitlines()[1:]]
    assert [e["use"] for e in events if e["kind"] == "exchange_anchor"] == ["reused"] and wire.calls == 1
    # TURN 101: the reuse keeps the original's verified time (COST100.REUSE.TIME), never the store-reading time.
    assert ledger.calls[-1].wall_s == first_ledger.calls[-1].wall_s and out["status"] == "answered"


# TURN 87 (Codex's TURN 86 review).

def test_a_changed_record_loses_its_anchor_mark_and_is_quarantined_on_reuse(gateway, tmp_path):
    wire, path = Wire(), tmp_path / "x"
    gw, _, _, root = gateway(provider=adapter(wire, ExchangeStore(path), reportable=True), journal=tmp_path / "j1")
    assert submit(gw)["status"] == "answered"
    ex = json.loads(path.read_text()); ex["elapsed_s"] = 9.0
    path.write_text(json.dumps(ex) + "\n")
    store = ExchangeStore(path)
    assert store.records()[0]["anchored"] is False
    gw2, _, _, _ = gateway(provider=adapter(wire, store, reportable=True), journal=tmp_path / "j2")
    out = submit(gw2)
    assert out["status"] == "failed" and out["text"] is None and out["reportable"] is False and wire.calls == 1
    assert "differs from its protected anchor" in M.verify_anchors(store, [(root / "log", root / "heads")])[0]


@pytest.mark.parametrize("mark", [b'{"exchange_id":"unknown","record_sha256":"0"}\n', b"not json\n",
                                  b'{"exchange_id":"x"}\n', b"partial"])
def test_a_malformed_anchor_mark_fails_closed(gateway, tmp_path, mark):
    gw, _, _, _ = gateway(provider=adapter(Wire(), ExchangeStore(tmp_path / "x")), journal=tmp_path / "j")
    submit(gw)
    marks = tmp_path / "x.anchored"
    marks.write_bytes(marks.read_bytes() + mark)
    with pytest.raises(ExchangeError):
        ExchangeStore(tmp_path / "x")


def test_a_failed_mark_quarantines_the_fresh_reply(gateway, tmp_path, monkeypatch):
    store = ExchangeStore(tmp_path / "x")
    def fail(xid, *refs):                                               # TURN 89: the mark also names the log
        raise OSError("invented failure writing the mark")
    monkeypatch.setattr(store, "mark_anchored", fail)
    gw, _, _, _ = gateway(provider=adapter(Wire(), store, reportable=True), journal=tmp_path / "j")
    out = submit(gw)
    assert out["status"] == "failed" and out["text"] is None and out["reportable"] is False and out["tokens_in"] == 11


def test_an_anchor_without_its_original_request_is_not_joined(gateway, tmp_path):
    from telohelm.runtime.trusted_log import TrustedLog
    store = ExchangeStore(tmp_path / "x")
    gw, _, _, root = gateway(provider=adapter(Wire(), store), journal=tmp_path / "j")
    submit(gw)
    events = [json.loads(l)["payload"] for l in (root / "log").read_text().splitlines()[1:]]
    forged = TrustedLog(tmp_path / "f", tmp_path / "fh", "invented")
    for e in events:
        if e["kind"] != "model_start":                                  # the anchor alone, with no original request
            forged.append(e)
    assert "not joined" in M.verify_anchors(store, [(tmp_path / "f", tmp_path / "fh")])[0]
    assert M.verify_exchange_evidence(adapter(Wire(), store), store, tmp_path / "j", [(root / "log", root / "heads")]) == []


def test_the_checked_configuration_is_frozen():
    c = checked()
    for attempt in (lambda: setattr(c, "_json", "{}"), lambda: setattr(c, "fingerprint", "0" * 64)):
        with pytest.raises(AttributeError):
            attempt()


# TURN 89 (Codex's TURN 88 review).

def _first_run(gateway, tmp_path):
    wire = Wire()
    gw, _, _, root = gateway(provider=adapter(wire, ExchangeStore(tmp_path / "x"), reportable=True),
                             journal=tmp_path / "j1")
    assert submit(gw)["status"] == "answered"
    return wire, root


def test_a_coherent_rewrite_that_keeps_the_log_pointer_is_still_refused(gateway, tmp_path):
    """The record and its mark rewritten together, the mark still naming the original log: only the protected anchor
    can tell, and it does."""
    wire, root = _first_run(gateway, tmp_path)
    path, marks = tmp_path / "x", tmp_path / "x.anchored"
    body = ok_body(); body["choices"][0]["message"]["content"] = "altered invented action"
    raw = http(json.dumps(body).encode())
    rec = json.loads(path.read_text())
    rec.update(reply_wire=base64.b64encode(raw).decode(), reply_sha256=hashlib.sha256(raw).hexdigest(),
               reply_bytes=len(raw))
    path.write_text(json.dumps(rec) + "\n")
    m = json.loads(marks.read_text()); m["record_sha256"] = record_sha256(rec)
    marks.write_text(json.dumps(m) + "\n")
    store = ExchangeStore(path)
    assert store.records()[0]["anchored"] is True                       # the cache is fooled
    gw2, _, _, _ = gateway(provider=adapter(wire, store, reportable=True), journal=tmp_path / "j2")
    out = submit(gw2)
    assert out["text"] is None and out["reportable"] is False and wire.calls == 1
    assert journal_rows(tmp_path / "j2")[0]["response"]["provider_ids"]["refused"] == "original not admitted"


def test_a_mark_with_no_log_pointer_gives_no_admission(gateway, tmp_path):
    wire, root = _first_run(gateway, tmp_path)
    marks = tmp_path / "x.anchored"
    m = json.loads(marks.read_text()); m.pop("log"); m.pop("heads")
    marks.write_text(json.dumps(m) + "\n")
    gw2, _, _, _ = gateway(provider=adapter(wire, ExchangeStore(tmp_path / "x"), reportable=True),
                           journal=tmp_path / "j2")
    assert submit(gw2)["text"] is None and wire.calls == 1


def test_a_journal_with_altered_tokens_is_quarantined_but_charged_once(gateway, tmp_path):
    wire, root = _first_run(gateway, tmp_path)
    rows = journal_rows(tmp_path / "j1"); rows[0]["response"]["tokens_out"] = 999
    (tmp_path / "j1").write_text("".join(json.dumps(r) + "\n" for r in rows))
    gw2, ledger, _, _ = gateway(provider=adapter(wire, ExchangeStore(tmp_path / "x"), reportable=True),
                                journal=tmp_path / "j1")
    out = submit(gw2)
    assert out["recovered"] and out["text"] is None and out["reportable"] is False and wire.calls == 1
    assert len([c for c in ledger.calls if c.tokens_in]) == 1


def test_an_intact_journal_recovery_is_admitted(gateway, tmp_path):
    wire, root = _first_run(gateway, tmp_path)
    gw2, _, _, _ = gateway(provider=adapter(wire, ExchangeStore(tmp_path / "x"), reportable=True),
                           journal=tmp_path / "j1")
    out = submit(gw2)
    assert out["recovered"] and out["status"] == "answered" and out["text"] == "invented answer" and out["reportable"]


def test_the_configuration_snapshot_cannot_be_replaced(tmp_path):
    a = adapter(Wire(), ExchangeStore(tmp_path / "x"))
    for name in ("config", "config_sha256"):
        with pytest.raises(AttributeError):
            setattr(a, name, None)


# TURN 91 (Codex's TURN 90 review).

def test_only_a_registered_root_admits_and_a_trusted_caller_can_register_one(gateway, tmp_path):
    wire, root = _first_run(gateway, tmp_path)
    fresh = M.ProtectedRoots()                                          # a new process: nothing registered yet
    a = M.LocalChatProvider("http://127.0.0.1:9", ROLES, wire, TIMEOUTS, exchanges=ExchangeStore(tmp_path / "x"),
                            config=checked(), roots=fresh)
    out = a.complete(M.ModelRequest('["stream-a","episode-a","condition-a","small_model",1]', "attempt1",
                                    "episode-a", "condition-a", "stream-a", "small_model",
                                    (("user", "invented request"),), 256, 1, 10, 10, 2.4))
    assert out.text is None and out.provider_ids["refused"] == "original not admitted"
    fresh.register(root / "log", root / "heads")                       # from the caller's own protected records
    b = M.LocalChatProvider("http://127.0.0.1:9", ROLES, wire, TIMEOUTS, exchanges=ExchangeStore(tmp_path / "x"),
                            config=checked(), roots=fresh)
    gw, _, _, _ = gateway(provider=b, journal=tmp_path / "j2")
    assert submit(gw)["text"] == "invented answer" and wire.calls == 1


@pytest.mark.parametrize("change", [("model", "other-model"), ("temperature", 0.7), ("seed", 5), ("max_tokens", 9),
                                    ("messages", [{"role": "user", "content": "another question"}])])
def test_request_bytes_that_do_not_match_the_configuration_are_refused(gateway, tmp_path, change):
    wire, root = _first_run(gateway, tmp_path)
    path, marks = tmp_path / "x", tmp_path / "x.anchored"
    rec = json.loads(path.read_text())
    head, _, payload = base64.b64decode(rec["request_wire"]).partition(b"\r\n\r\n")
    body = json.loads(payload); body[change[0]] = change[1]
    raw = head + b"\r\n\r\n" + json.dumps(body, sort_keys=True).encode()
    rec.update(request_wire=base64.b64encode(raw).decode(), request_sha256=hashlib.sha256(raw).hexdigest())
    path.write_text(json.dumps(rec) + "\n")
    store = ExchangeStore(path)
    gw2, _, _, _ = gateway(provider=adapter(wire, store, reportable=True), journal=tmp_path / "j2")
    out = submit(gw2)
    assert out["text"] is None and wire.calls == 1                      # changed record: refused before any wire check
    assert M.verify_anchors(store, [(root / "log", root / "heads")])


def test_the_wire_check_itself_finds_bytes_that_disagree_with_the_configuration(gateway, tmp_path):
    a = adapter(Wire(), ExchangeStore(tmp_path / "x"))
    gw, _, _, _ = gateway(provider=a, journal=tmp_path / "j")
    submit(gw)
    saved = a._exchanges.records()[0]
    assert a._wire_problem(saved) is None
    head, _, payload = saved["request_wire"].partition(b"\r\n\r\n")
    for k, v in [("model", "other"), ("temperature", 0.7), ("seed", 5), ("max_tokens", 9)]:
        body = json.loads(payload); body[k] = v
        assert a._wire_problem({**saved, "request_wire": head + b"\r\n\r\n" + json.dumps(body).encode()})


def test_an_unreportable_direct_call_is_labelled_and_never_admitted_later(gateway, tmp_path):
    wire = Wire()
    r = adapter(wire, ExchangeStore(tmp_path / "x")).complete(
        M.ModelRequest('["stream-a","episode-a","condition-a","small_model",1]', "attempt1", "episode-a",
                       "condition-a", "stream-a", "small_model", (("user", "invented request"),), 256, 1, 10, 10, 2.4))
    assert r.status == "answered" and r.provider_ids["scope"] == "direct"
    gw, _, _, _ = gateway(provider=adapter(wire, ExchangeStore(tmp_path / "x"), reportable=True), journal=tmp_path / "j")
    out = submit(gw)
    assert out["text"] is None and wire.calls == 1


def test_a_journal_response_with_no_stored_exchange_has_unknown_usage(gateway, tmp_path, monkeypatch):
    store = ExchangeStore(tmp_path / "x")
    def broken(*a, **kw):
        raise OSError("invented full disk")
    monkeypatch.setattr(store, "save", broken)
    a = adapter(Wire(), store)
    gw, _, _, _ = gateway(provider=a, journal=tmp_path / "j")
    submit(gw)
    rows = journal_rows(tmp_path / "j"); rows[0]["response"].update(tokens_in=50, tokens_out=50)
    (tmp_path / "j").write_text("".join(json.dumps(r) + "\n" for r in rows))
    gw2, ledger, _, _ = gateway(provider=adapter(Wire(), ExchangeStore(tmp_path / "x2")), journal=tmp_path / "j")
    out = submit(gw2)
    assert out["recovered"] and out["text"] is None and ledger.calls[-1].tokens_in is None


# TURN 93 (Codex's TURN 92 review).

def test_a_rejected_receipt_with_no_registered_original_has_unknown_usage(gateway, tmp_path):
    wire, root = _first_run(gateway, tmp_path)
    a = M.LocalChatProvider("http://127.0.0.1:9", ROLES, wire, TIMEOUTS, exchanges=ExchangeStore(tmp_path / "x"),
                            config=checked(), roots=M.ProtectedRoots(), reportable=True)
    gw, ledger, _, _ = gateway(provider=a, journal=tmp_path / "j2")
    out = submit(gw)
    assert out["text"] is None and ledger.calls[-1].tokens_in is None and ledger.calls[-1].tokens_out is None
    ids = journal_rows(tmp_path / "j2")[0]["response"]["provider_ids"]
    assert (ids["stored_tokens_in"], ids["usage_source"], ids["refused"]) == ("11", "unknown", "original not admitted")


def test_a_fresh_reply_whose_sent_bytes_match_is_released(gateway, tmp_path):
    a = adapter(Wire(), ExchangeStore(tmp_path / "x"), reportable=True)
    gw, _, _, _ = gateway(provider=a, journal=tmp_path / "j")
    assert submit(gw)["text"] == "invented answer" and a._wire_problem(a._exchanges.records()[0]) is None
