"""TURN84 independent manifest/exchange controls. Invented data and faults only; no model calls."""
import json
import threading
from dataclasses import replace

import pytest

import telohelm.runtime.manifest as F; import telohelm.runtime.models as M
import telohelm.runtime.exchange as X
from telohelm.runtime.trusted_log import verify
from .test_exchange_join import Wire; from .test_exchange_join import adapter; from .test_exchange_join import CONFIG; from .test_exchange_join import journal_rows
from .test_manifest import invented
from .test_turn65_review_codex import gateway; from .test_turn65_review_codex import submit


def req():
    return M.ModelRequest('["invented-request"]', "attempt1", "e", "c", "s", "small_model",
                          (("user", "question"),), 64, 1, 10, 10, 2.4)


# Fixed in TURN 85 (Claude): MF84.REQUIRED: version/schema validation admits float version and entirely unknown required evidence
@pytest.mark.parametrize("kind", ["float_version", "unknown_fields"])
def test_manifest_required_evidence_has_typed_version_and_real_required_fields(kind):
    m = invented()
    if kind == "float_version":
        m["version"] = 1.0
    else:
        m = {"version": 1, **{name: {"unknown": None} for name in F.SECTIONS}}
    with pytest.raises(F.ManifestError):
        F.validate(m)


# Fixed in TURN 85 (Claude): MF84.GATE: an arbitrary hex string enables reportability without manifest/expected-file/start checks
def test_arbitrary_fingerprint_cannot_construct_a_reportable_configuration(tmp_path):
    with pytest.raises(M.GatewayError):
        adapter(Wire(), X.ExchangeStore(tmp_path / "x"), reportable=True, config="0" * 64)
    # A checked configuration/factory API with marked fixture changes is an acceptable equivalent gate.


# Fixed in TURN 85 (Claude): EX84.IDENTITY: exchange reuse matches body/config only and silently replaces original issue metadata
@pytest.mark.parametrize("field,value", [("goal_version", 2), ("sim_second", 11), ("action_step", 11),
                                        ("modelled_latency_s", 3.4)])
def test_exchange_reuse_cannot_retag_an_original_reply(gateway, tmp_path, field, value):
    wire = Wire(); store_path = tmp_path / "x"
    first, _, _, _ = gateway(provider=adapter(wire, X.ExchangeStore(store_path)), journal=tmp_path / "j1")
    first.submit("small_model", [("user", "same input")], 64, 1, 10, 10, 2.4)
    resumed, _, _, _ = gateway(provider=adapter(wire, X.ExchangeStore(store_path)), journal=tmp_path / "j2")
    kwargs = dict(role="small_model", messages=[("user", "same input")], max_output_tokens=64,
                  goal_version=1, sim_second=10, action_step=10, modelled_latency_s=2.4)
    kwargs[field] = value
    try:
        out = resumed.submit(**kwargs)
    except M.GatewayError:
        return
    assert out["status"] != "answered" and out["text"] is None and wire.calls == 1


# Fixed in TURN 85 (Claude): EX84.IDENTITY: extraction verification ignores journal attempt and original goal metadata
@pytest.mark.parametrize("field,value", [("goal_version", 2), ("attempt", "attempt2")])
def test_extraction_verifier_checks_original_request_meta(gateway, tmp_path, field, value):
    store = X.ExchangeStore(tmp_path / "x"); a = adapter(Wire(), store)
    gw, _, _, _ = gateway(provider=a, journal=tmp_path / "j")
    submit(gw)
    rows = journal_rows(tmp_path / "j")
    rows[0]["meta"][field] = value
    (tmp_path / "j").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert M.verify_extraction(a, store, tmp_path / "j")


# Fixed in TURN 85 (Claude): EX84.DURABLE: get() exposes an admitted record while its fsync is still pending
def test_a_record_is_invisible_until_its_write_is_durable(tmp_path, monkeypatch):
    store = X.ExchangeStore(tmp_path / "x")
    writing = threading.Event(); release = threading.Event(); read_done = threading.Event()
    reads, errors = [], []
    real_sync = X.os.fsync
    def delayed_sync(fd):
        writing.set()
        assert release.wait(2.0)
        real_sync(fd)
    monkeypatch.setattr(X.os, "fsync", delayed_sync)
    def save():
        try:
            store.save("r", "a", CONFIG, "small_model", "b" * 64, Wire().exchange("u", {}, 1.))
        except BaseException as e:
            errors.append(e)
    def read():
        reads.append(store.get("r", "a", CONFIG, "b" * 64)); read_done.set()
    writer = threading.Thread(target=save); reader = threading.Thread(target=read)
    writer.start()
    try:
        assert writing.wait(1.0)
        reader.start()
        done = read_done.wait(0.1)
        exposed = done and reads[0] is not None
    finally:
        release.set(); writer.join(2.0)
        if reader.ident is not None:
            reader.join(2.0)
    assert not errors and not exposed


# Fixed in TURN 85 (Claude): EX84.DURABLE: fsync failure rolls back only memory and a later save duplicates the complete on-disk line
def test_failed_sync_cannot_leave_a_store_that_accepts_a_duplicate_append(tmp_path, monkeypatch):
    store = X.ExchangeStore(tmp_path / "x")
    real = X.os.fsync; calls = [0]
    def stop_once(fd):
        calls[0] += 1
        if calls[0] == 1:
            raise OSError("invented fsync failure after complete flushed line")
        real(fd)
    monkeypatch.setattr(X.os, "fsync", stop_once)
    evidence = Wire().exchange("u", {}, 1.)
    with pytest.raises(OSError):
        store.save("r", "a", CONFIG, "small_model", "b" * 64, evidence)
    try:
        store.save("r", "a", CONFIG, "small_model", "b" * 64, evidence)
    except (OSError, X.ExchangeError):
        pass                       # refusing writes to a tainted store is acceptable
    assert len(X.ExchangeStore(tmp_path / "x").records()) <= 1


# Fixed in TURN 85 (Claude): EX84.TIME: store recovery measures cache processing as original call wall time
def test_store_recovery_keeps_original_total_wall_unknown(gateway, tmp_path, monkeypatch):
    wire = Wire(); a = adapter(wire, X.ExchangeStore(tmp_path / "x"))
    first, _, _, _ = gateway(provider=a, journal=tmp_path / "j")
    original_write = first._write
    def cut(rec):
        if rec["type"] == "response":
            raise OSError("invented cut after exchange before response journal")
        original_write(rec)
    monkeypatch.setattr(first, "_write", cut)
    with pytest.raises(OSError):
        submit(first)
    ticks = iter([100., 101.])
    recovered, ledger, _, _ = gateway(provider=adapter(wire, X.ExchangeStore(tmp_path / "x")),
                                      journal=tmp_path / "j", clock=lambda: next(ticks))
    assert submit(recovered)["status"] == "answered" and wire.calls == 1
    assert ledger.calls[-1].wall_s is None
    # Known wire elapsed can be retained as a separate component; it is not the unknown original total provider call.


# Fixed in TURN 85 (Claude): EX84.ORIGIN: unreportable stored exchange is promoted by a later adapter flag
def test_store_recovery_does_not_promote_original_reportability(gateway, tmp_path):
    wire = Wire()
    first, _, _, _ = gateway(provider=adapter(wire, X.ExchangeStore(tmp_path / "x"), reportable=False),
                              journal=tmp_path / "j1")
    assert submit(first)["reportable"] is False
    second, _, _, _ = gateway(provider=adapter(wire, X.ExchangeStore(tmp_path / "x"), reportable=True),
                               journal=tmp_path / "j2", replay=True)
    assert submit(second)["reportable"] is False and wire.calls == 1


# Fixed in TURN 85 (Claude): EX84.SCHEMA: unsupported role/nonfinite elapsed/incomplete received evidence is admitted
@pytest.mark.parametrize("change", ["role", "elapsed", "received_incomplete"])
def test_semantically_malformed_exchange_evidence_is_refused(tmp_path, change):
    evidence = Wire().exchange("u", {}, 1.); role = "small_model"
    if change == "role":
        role = "unsupported role"
    elif change == "elapsed":
        evidence["elapsed_s"] = float("nan")
    else:
        evidence["reply_complete"] = False
    with pytest.raises(X.ExchangeError):
        X.ExchangeStore(tmp_path / "x").save("r", "a", CONFIG, role, "b" * 64, evidence)


def test_exchange_is_durable_before_interpretation(tmp_path, monkeypatch):
    store = X.ExchangeStore(tmp_path / "x"); a = adapter(Wire(), store)
    real = M.interpret; seen = []
    def inspect_saved(evidence):
        on_disk = X.ExchangeStore(tmp_path / "x").records()
        assert len(on_disk) == 1 and on_disk[0]["reply_wire"] == evidence["reply_wire"]
        seen.append(True)
        return real(evidence)
    monkeypatch.setattr(M, "interpret", inspect_saved)
    assert a.complete(req()).status == "answered" and seen == [True]


def test_distinct_streams_have_distinct_exchange_identity_and_charges(gateway, tmp_path):
    wire = Wire(); store = X.ExchangeStore(tmp_path / "x")
    roots = []
    for stream in ("alpha", "alpha|beta"):
        gw, ledger, _, root = gateway(provider=adapter(wire, store), stream=stream)
        assert submit(gw)["status"] == "answered"
        assert sum(c.tokens_in or 0 for c in ledger.calls) == 11
        roots.append(root)
    assert wire.calls == 2 and len({r["exchange_id"] for r in store.records()}) == 2
    assert all(verify(r / "log", r / "heads")["ok"] for r in roots)


def test_exact_original_exchange_replay_has_no_fresh_call(gateway, tmp_path):
    wire = Wire()
    gw, _, _, _ = gateway(provider=adapter(wire, X.ExchangeStore(tmp_path / "x")), journal=tmp_path / "j1")
    live = submit(gw)
    after, _, _, _ = gateway(provider=adapter(wire, X.ExchangeStore(tmp_path / "x")), journal=tmp_path / "j2", replay=True)
    replay = submit(after)
    assert {k: replay[k] for k in ("status", "text", "tokens_in", "tokens_out", "goal_version")} == \
           {k: live[k] for k in ("status", "text", "tokens_in", "tokens_out", "goal_version")}
    assert wire.calls == 1
