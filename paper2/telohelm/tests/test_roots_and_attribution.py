"""Roots restored in a new process, drift at use and close, replay attribution, and pinned bytecode (TURN 99, after
Codex's TURN 98 review). Invented values, temporary files and modules only."""
from __future__ import annotations

import hashlib
import json
import os
import py_compile

import pytest

import telohelm.codecapture as C
import telohelm.runtime.exchange as X; import telohelm.runtime.manifest as F; import telohelm.runtime.models as M; import telohelm.runtime.startup as S
from telohelm.runtime.trusted_log import TrustedLog
from .test_exchange_join import Wire; from .test_exchange_join import adapter
from .test_startup import matching
from .test_startup_gate import events
from .test_turn65_review_codex import gateway; from .test_turn65_review_codex import submit  # noqa: F401  (fixture)
from .test_turn98_review_codex import temporary_capture


# Roots restored from a protected registry in a new process.

def _logged_run(gateway, tmp_path):
    wire = Wire()
    gw, _, _, root = gateway(provider=adapter(wire, X.ExchangeStore(tmp_path / "x"), reportable=True),
                             journal=tmp_path / "j1")
    assert submit(gw)["status"] == "answered"
    return wire, root


def test_a_restored_registry_admits_the_original_and_nothing_else(gateway, tmp_path):
    wire, root = _logged_run(gateway, tmp_path)
    registry = tmp_path / "roots"
    M.ProtectedRoots.record(registry, root / "log", root / "heads")
    restored = M.ProtectedRoots.restore(registry)                       # as a new process would
    assert restored.trusts(root / "log", root / "heads")
    assert not restored.trusts(tmp_path / "other-log", tmp_path / "other-heads")
    from .test_exchange_join import checked
    from .test_local_adapter import ROLES; from .test_local_adapter import TIMEOUTS
    a = M.LocalChatProvider("http://127.0.0.1:9", ROLES, wire, TIMEOUTS, exchanges=X.ExchangeStore(tmp_path / "x"),
                            config=checked(), roots=restored, reportable=True)
    gw, _, _, _ = gateway(provider=a, journal=tmp_path / "j2")
    assert submit(gw)["text"] == "invented answer" and wire.calls == 1


@pytest.mark.parametrize("damage", ["anchor", "heads", "log", "torn", "missing"])
def test_a_damaged_registry_or_root_fails_closed(gateway, tmp_path, damage):
    wire, root = _logged_run(gateway, tmp_path)
    registry = tmp_path / "roots"
    M.ProtectedRoots.record(registry, root / "log", root / "heads")
    if damage == "anchor":
        e = json.loads(registry.read_text()); e["anchor"] = "0" * 64; registry.write_text(json.dumps(e) + "\n")
    elif damage == "heads":                                             # substituted heads for another log
        other = TrustedLog(tmp_path / "olog", tmp_path / "oheads", "invented")
        e = json.loads(registry.read_text()); e["heads"] = str(tmp_path / "oheads"); registry.write_text(json.dumps(e) + "\n")
    elif damage == "log":
        lines = (root / "log").read_text().splitlines(); (root / "log").write_text("\n".join(lines[:-1]) + "\n")
    elif damage == "torn":
        registry.write_bytes(registry.read_bytes().rstrip(b"\n"))
    else:
        registry.unlink()
    with pytest.raises(M.GatewayError):
        M.ProtectedRoots.restore(registry)


# Drift at use and at close, with a real catalog of a temporary module.

def _started_with_catalog(gateway, tmp_path, monkeypatch):
    config, cat, module = temporary_capture(tmp_path, monkeypatch)
    wire = Wire()
    a = adapter(wire, X.ExchangeStore(tmp_path / "x"), config=config, reportable=True)
    gw, ledger, _, root = gateway(provider=a, start=False)
    authority = S.RunAuthority(config, C.evidence(cat, [module]), check_drift=True)
    assert gw.record_start(matching(config), authority) == []
    return wire, gw, ledger, root, tmp_path / "package" / "probe.py"


def test_drift_before_a_call_refuses_it(gateway, tmp_path, monkeypatch):
    wire, gw, ledger, root, source = _started_with_catalog(gateway, tmp_path, monkeypatch)
    source.write_bytes(b"def answer():\n    return 3\n")
    out = submit(gw)
    assert out["refusal"] == "startup not admitted" and wire.calls == 0 and ledger.calls == []
    refused = [e for e in events(root) if e["kind"] == "startup_refused"][-1]
    assert refused["problems"] == ["code drift: probe.py"]


def test_drift_at_close_voids_the_code_identity_and_stops_further_calls(gateway, tmp_path, monkeypatch):
    wire, gw, ledger, root, source = _started_with_catalog(gateway, tmp_path, monkeypatch)
    assert submit(gw)["status"] == "answered" and wire.calls == 1
    source.write_bytes(b"def answer():\n    return 3\n")
    assert gw.close() == ["probe.py"]
    close = [e for e in events(root) if e["kind"] == "run_close"][-1]
    assert close["code_identity_void"] is True and ledger.calls[-1].tokens_in == 11      # the call is not undone
    assert submit(gw)["status"] == "refused" and wire.calls == 1


def test_a_clean_close_records_no_drift(gateway, tmp_path, monkeypatch):
    wire, gw, ledger, root, source = _started_with_catalog(gateway, tmp_path, monkeypatch)
    submit(gw)
    assert gw.close() == [] and [e for e in events(root) if e["kind"] == "run_close"][-1]["code_identity_void"] is False


# Bytecode a reviewed build pinned is admitted; other loaders are not.

def test_bytecode_pinned_by_the_configuration_is_admitted(gateway, tmp_path, monkeypatch):
    package = tmp_path / "package"; package.mkdir()
    source = package / "probe.py"; source.write_bytes(b"def answer():\n    return 1\n")
    os.utime(source, (1700000000, 1700000000))
    py_compile.compile(str(source), doraise=True)
    from .test_manifest import invented
    import importlib.util, sys
    name = "turn99_pinned_probe"
    cat = C.catalog(package, name, "invented pre-import time", "invented reviewed build")
    m = invented(); m["runtime"]["code_sha256"] = dict(cat["files"]); m["runtime"]["bytecode_sha256"] = dict(cat["bytecode"])
    (tmp_path / "expected").write_text(F.fingerprint(m))
    config = F.checked(m, tmp_path / "expected")
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec); monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    gw, _, _, _ = gateway(provider=adapter(Wire(), X.ExchangeStore(tmp_path / "x"), config=config, reportable=True),
                          start=False)
    assert gw.record_start(matching(config), S.RunAuthority(config, C.evidence(cat, [module]))) == []


def test_a_module_loaded_by_another_loader_is_refused():
    from .test_exchange_join import checked
    from .test_startup import invented_code
    c = checked()
    code = invented_code(c); code["loaded"][0]["loader"] = "SourcelessFileLoader"
    event = {"binding": S.binding("a" * 64, "e", "c", "s", c.fingerprint), "record": matching(c),
             "record_sha256": hashlib.sha256(S._canonical(matching(c)).encode()).hexdigest(), "code": code}
    assert any("not loaded from pinned source" in p for p in S.decide(event, "a" * 64, "e", "c", "s", c, True))


# Replay attribution against the original run.

def test_a_recovery_is_attributed_as_the_original_decided(gateway, tmp_path):
    wire, root = _logged_run(gateway, tmp_path)
    gw, _, _, later = gateway(provider=adapter(wire, X.ExchangeStore(tmp_path / "x"), reportable=True),
                              journal=tmp_path / "j1")
    assert submit(gw)["recovered"]
    assert M.compare_attribution((root / "log", root / "heads"), (later / "log", later / "heads")) == []


def test_a_reuse_with_text_the_original_did_not_give_is_found(gateway, tmp_path):
    wire, root = _logged_run(gateway, tmp_path)
    gw, _, _, later = gateway(provider=adapter(wire, X.ExchangeStore(tmp_path / "x"), reportable=True),
                              journal=tmp_path / "j2")
    assert submit(gw)["text"] == "invented answer"
    # The original log rewritten as a fresh, self-consistent log whose original answer differs.
    fake = TrustedLog(tmp_path / "flog", tmp_path / "fheads", "invented")
    for e in events(root):
        if e["kind"] == "model_end":
            e = {**e, "response": {**e["response"], "text": "a different original answer"}}
        fake.append(e)
    problems = M.compare_attribution((tmp_path / "flog", tmp_path / "fheads"), (later / "log", later / "heads"))
    assert problems and "text the original did not give" in problems[0]


def test_a_replay_with_no_matching_original_call_is_found(gateway, tmp_path):
    wire, root = _logged_run(gateway, tmp_path)
    gw, _, _, later = gateway(provider=adapter(wire, X.ExchangeStore(tmp_path / "x"), reportable=True),
                              journal=tmp_path / "j1")
    submit(gw)
    empty = TrustedLog(tmp_path / "elog", tmp_path / "eheads", "invented")
    assert "no original call" in M.compare_attribution((tmp_path / "elog", tmp_path / "eheads"),
                                                       (later / "log", later / "heads"))[0]


# TURN 101 (Codex's TURN 100 review).

def test_drift_found_once_is_logged_and_persists_after_the_file_is_restored(gateway, tmp_path, monkeypatch):
    wire, gw, ledger, root, source = _started_with_catalog(gateway, tmp_path, monkeypatch)
    original = source.read_bytes()
    source.write_bytes(b"def answer():\n    return 3\n")
    submit(gw)
    source.write_bytes(original)
    submit(gw)
    refusals = [e["problems"] for e in events(root) if e["kind"] == "startup_refused"]
    assert refusals == [["code drift: probe.py"], ["code identity void"]] and wire.calls == 0
    assert [e["code_drift"] for e in events(root) if e["kind"] == "code_identity_void"] == [["probe.py"]]


def test_a_new_source_file_under_the_captured_base_is_drift(gateway, tmp_path, monkeypatch):
    wire, gw, ledger, root, source = _started_with_catalog(gateway, tmp_path, monkeypatch)
    (source.parent / "added.py").write_text("x = 1\n")
    assert submit(gw)["status"] == "refused" and wire.calls == 0
    assert [e for e in events(root) if e["kind"] == "startup_refused"][-1]["problems"] == ["code drift: added.py (not captured)"]


def test_an_unverifiable_root_is_not_recorded(tmp_path):
    with pytest.raises(M.GatewayError):
        M.ProtectedRoots.record(tmp_path / "registry", tmp_path / "no-log", tmp_path / "no-heads")
    assert not (tmp_path / "registry").exists()


# TURN 103 (Codex's TURN 102 review).

def test_a_void_that_could_not_be_written_is_published_on_the_next_proof(gateway, tmp_path, monkeypatch):
    wire, gw, ledger, root, source = _started_with_catalog(gateway, tmp_path, monkeypatch)
    source.write_bytes(b"def answer():\n    return 3\n")
    real = gw._log.append
    def once(payload):
        if payload.get("kind") == "code_identity_void" and not getattr(once, "failed", False):
            once.failed = True
            raise OSError("invented failure writing the void event")
        return real(payload)
    monkeypatch.setattr(gw._log, "append", once)
    with pytest.raises(OSError):
        submit(gw)
    assert not any(e["kind"] == "code_identity_void" for e in events(root))
    assert submit(gw)["status"] == "refused"                            # the retry publishes it
    assert [e["kind"] for e in events(root)].count("code_identity_void") == 1
    assert submit(gw)["status"] == "refused"
    assert [e["kind"] for e in events(root)].count("code_identity_void") == 1   # once only


def test_an_original_typed_differently_is_not_the_same_call(gateway, tmp_path):
    wire, root = _logged_run(gateway, tmp_path)
    gw, _, _, later = gateway(provider=adapter(wire, X.ExchangeStore(tmp_path / "x"), reportable=True),
                              journal=tmp_path / "j1")
    submit(gw)
    fake = TrustedLog(tmp_path / "flog", tmp_path / "fheads", "invented")
    for e in events(root):
        if e["kind"] == "model_start":
            e = {**e, "request": {**e["request"], "goal_version": float(e["request"]["goal_version"])}}
        fake.append(e)
    assert "no original call" in M.compare_attribution((tmp_path / "flog", tmp_path / "fheads"),
                                                       (later / "log", later / "heads"))[0]
