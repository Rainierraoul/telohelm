"""The gateway's start boundary (TURN 97, Codex's SG96 contract). Invented configuration, evidence and transports; no
server, cluster or real capture."""
from __future__ import annotations

import json

import pytest

import telohelm.runtime.exchange as X; import telohelm.runtime.manifest as F; import telohelm.runtime.models as M; import telohelm.runtime.startup as S
from telohelm.runtime.trusted_log import TrustedLog
from .test_exchange_join import Wire; from .test_exchange_join import adapter; from .test_exchange_join import checked
from .test_manifest import invented
from .test_startup import invented_code; from .test_startup import matching; from .test_startup import start_gateway
from .test_turn65_review_codex import gateway; from .test_turn65_review_codex import submit  # noqa: F401  (fixture)


def events(root):
    return [json.loads(l)["payload"] for l in (root / "log").read_text().splitlines()[1:]]


def unstarted(gateway, tmp_path, name="x"):
    wire = Wire()
    a = adapter(wire, X.ExchangeStore(tmp_path / name), reportable=True)
    gw, ledger, _, root = gateway(provider=a, journal=tmp_path / f"j{name}", start=False)
    return wire, a, gw, ledger, root


# SG96.1: the authority comes from the trusted bootstrap and cannot be substituted.

def test_the_authority_needs_a_checked_configuration_and_code_evidence():
    c = checked()
    with pytest.raises(S.StartupError):
        S.RunAuthority({"fingerprint": c.fingerprint}, invented_code(c))
    with pytest.raises(S.StartupError):
        S.RunAuthority(c, {"phase": "pre-import"})
    a = S.RunAuthority(c, invented_code(c))
    with pytest.raises(AttributeError):
        a._config = checked(temperature=0.5)
    code = a.code; code["phase"] = "post-import"
    assert a.code["phase"] == "pre-import"                              # handed out as a copy


def test_a_substituted_authority_does_not_admit(gateway, tmp_path):
    wire, a, gw, ledger, root = unstarted(gateway, tmp_path)
    other = checked(temperature=0.5)
    problems = gw.record_start(matching(other), S.RunAuthority(other, invented_code(other)))
    assert "the provider's configuration is not the authority's" in problems
    assert submit(gw)["refusal"] == "startup not admitted" and wire.calls == 0


def test_a_missing_or_altered_frozen_fingerprint_gives_no_authority(tmp_path):
    m = invented()
    with pytest.raises(F.ManifestError):
        F.checked(m, tmp_path / "missing")
    (tmp_path / "expected").write_text("0" * 64)
    with pytest.raises(F.ManifestError):
        F.checked(m, tmp_path / "expected")


# SG96.2 and SG96.3: a complete, bound, timely record, or no provider work at all.

def test_no_start_means_no_provider_work_and_no_allowance(gateway, tmp_path):
    wire, a, gw, ledger, root = unstarted(gateway, tmp_path)
    for _ in range(3):
        out = submit(gw)
        assert out["status"] == "refused" and out["text"] is None and out["reportable"] is False
    assert wire.calls == 0 and ledger.calls == [] and gw.used == {"small_model": 0, "reasoning_model": 0}
    assert [e["kind"] for e in events(root)].count("startup_refused") == 3
    assert not any(e["kind"] == "model_start" for e in events(root))
    assert start_gateway(gw, a) == []                                  # then a valid start permits the first call
    out = submit(gw)
    assert out["status"] == "answered" and out["request_id"].endswith(",1]") and wire.calls == 1


@pytest.mark.parametrize("change,expect", [
    (lambda r: r["items"].pop("weights"), "weights: missing"),
    (lambda r: r["items"].update(launch=S.unknown("invented", "t", "p")), "launch: unknown"),
    (lambda r: r["items"]["python"].update(value="3.11"), "python: differs from the configuration"),
    (lambda r: r.update(config_sha256="0" * 64), "the start record was made for another configuration"),
    (lambda r: r["items"]["python"].update(value=float("nan")), "malformed start record"),
])
def test_an_incomplete_unknown_or_mismatching_record_refuses(gateway, tmp_path, change, expect):
    wire, a, gw, ledger, root = unstarted(gateway, tmp_path)
    rec = matching(a.config); change(rec)
    problems = gw.record_start(rec, S.RunAuthority(a.config, invented_code(a.config)))
    assert any(p.startswith(expect) for p in problems)
    assert submit(gw)["status"] == "refused" and wire.calls == 0 and gw.used["small_model"] == 0


@pytest.mark.parametrize("code,expect", [
    (dict(phase="post-import"), "the code evidence was not captured before import"),
    (dict(problems=["x.py: differs from the catalog"]), "the code evidence has problems"),
    (dict(files={"other.py": "6" * 64}), "the captured source files are not exactly the pinned ones"),   # TURN 99
])
def test_weak_or_failing_code_evidence_refuses(gateway, tmp_path, code, expect):
    wire, a, gw, ledger, root = unstarted(gateway, tmp_path)
    assert expect in start_gateway(gw, a, **code)
    assert submit(gw)["status"] == "refused" and wire.calls == 0


def test_only_the_first_start_counts(gateway, tmp_path):
    wire, a, gw, ledger, root = unstarted(gateway, tmp_path)
    bad = matching(a.config); bad["items"].pop("python")
    assert gw.record_start(bad, S.RunAuthority(a.config, invented_code(a.config)))
    assert start_gateway(gw, a) == ["a start record was already written"]   # a later good record admits nothing
    assert submit(gw)["status"] == "refused" and wire.calls == 0
    assert [e["kind"] for e in events(root)].count("start_record_ignored") == 1


def test_a_start_after_a_model_call_is_late(gateway, tmp_path):
    gw, _, _, _ = gateway()                                            # a scripted provider, not store-backed
    submit(gw)
    c = checked()
    problems = gw.record_start(matching(c), S.RunAuthority(c, invented_code(c)))
    assert "written after a model call" in problems and not gw.started


def test_the_start_record_is_bound_to_this_log_and_scope(gateway, tmp_path):
    gw, _, provider, root = gateway(provider=adapter(Wire(), X.ExchangeStore(tmp_path / "y"), reportable=True),
                                    journal=tmp_path / "jy")
    start = next(e for e in events(root) if e["kind"] == "start_record")
    assert start["admitted"] and start["binding"]["log_anchor"] == S.log_anchor(root / "heads")
    assert start["binding"]["stream"] == "stream-a" and start["binding"]["config_sha256"] == provider.config.fingerprint
    # The same event judged as another log's or another stream's start is refused.
    assert "bound to another run, scope or configuration" in S.decide(
        start, "0" * 64, "episode-a", "condition-a", "stream-a", provider.config, True)
    assert "bound to another run, scope or configuration" in S.decide(
        start, start["binding"]["log_anchor"], "episode-a", "condition-a", "stream-b", provider.config, True)


# SG96.4: recovery needs both this run's start and the original run's.

def _original(gateway, tmp_path):
    wire = Wire()
    gw, _, _, root = gateway(provider=adapter(wire, X.ExchangeStore(tmp_path / "x"), reportable=True),
                             journal=tmp_path / "j1")
    assert submit(gw)["status"] == "answered"
    return wire, root


def test_journal_recovery_without_a_current_start_keeps_the_verified_cost_only(gateway, tmp_path):
    wire, root = _original(gateway, tmp_path)
    a = adapter(wire, X.ExchangeStore(tmp_path / "x"), reportable=True)
    gw, ledger, _, _ = gateway(provider=a, journal=tmp_path / "j1", start=False)
    out = submit(gw)
    assert out["recovered"] and out["text"] is None and out["reportable"] is False and wire.calls == 1
    assert (ledger.calls[-1].tokens_in, ledger.calls[-1].tokens_out) == (11, 4)


def test_store_reuse_without_a_current_start_is_refused_before_anything(gateway, tmp_path):
    wire, root = _original(gateway, tmp_path)
    a = adapter(wire, X.ExchangeStore(tmp_path / "x"), reportable=True)
    gw, ledger, _, _ = gateway(provider=a, journal=tmp_path / "j2", start=False)
    assert submit(gw)["refusal"] == "startup not admitted" and ledger.calls == [] and wire.calls == 1


def test_an_original_log_without_its_start_admits_no_reuse(gateway, tmp_path):
    wire, root = _original(gateway, tmp_path)
    # A copy of the original log without its start record, registered as if trusted: the anchor joins, but the
    # original run's start is missing, so its exchange gives no text.
    copy = TrustedLog(tmp_path / "nostart", tmp_path / "nostart-heads", "invented")
    for e in events(root):
        if e["kind"] != "start_record":
            copy.append(e)
    M.PROTECTED_ROOTS.register(tmp_path / "nostart", tmp_path / "nostart-heads")
    marks = tmp_path / "x.anchored"
    m = json.loads(marks.read_text()); m.update(log=str(tmp_path / "nostart"), heads=str(tmp_path / "nostart-heads"))
    marks.write_text(json.dumps(m) + "\n")
    gw, _, _, _ = gateway(provider=adapter(wire, X.ExchangeStore(tmp_path / "x"), reportable=True),
                          journal=tmp_path / "j2")
    out = submit(gw)
    assert out["text"] is None and wire.calls == 1
    ids = json.loads((tmp_path / "j2").read_text().splitlines()[0])["response"]["provider_ids"]
    assert ids["refused"] == "original startup not admitted"


def test_a_started_original_and_current_run_recover_in_full(gateway, tmp_path):
    wire, root = _original(gateway, tmp_path)
    gw, ledger, _, _ = gateway(provider=adapter(wire, X.ExchangeStore(tmp_path / "x"), reportable=True),
                               journal=tmp_path / "j1")
    out = submit(gw)
    assert out["recovered"] and out["text"] == "invented answer" and out["reportable"] is True and wire.calls == 1


# SG96.7: cuts around the start write.

def test_a_failed_start_write_admits_nothing_and_a_retry_can_start(gateway, tmp_path, monkeypatch):
    wire, a, gw, ledger, root = unstarted(gateway, tmp_path)
    real = gw._log.append
    def fail_once(payload):
        if payload.get("kind") == "start_record" and not getattr(fail_once, "done", False):
            fail_once.done = True
            raise OSError("invented failure writing the start record")
        real(payload)
    monkeypatch.setattr(gw._log, "append", fail_once)
    with pytest.raises(OSError):
        start_gateway(gw, a)
    assert not gw.started and submit(gw)["status"] == "refused" and wire.calls == 0
    assert start_gateway(gw, a) == [] and submit(gw)["status"] == "answered" and wire.calls == 1


def test_a_new_run_after_a_stop_must_start_itself(gateway, tmp_path):
    wire, a, gw, ledger, root = unstarted(gateway, tmp_path)
    assert start_gateway(gw, a) == []                                  # started, then the run stops before any call
    gw2, ledger2, _, _ = gateway(provider=adapter(wire, X.ExchangeStore(tmp_path / "x"), reportable=True),
                                 journal=tmp_path / "jx", start=False)
    assert submit(gw2)["status"] == "refused" and wire.calls == 0      # the earlier run's start does not carry over
