"""TURN86 independent configuration, anchor and store controls. Invented data/faults only."""
import copy
import json
from dataclasses import replace

import pytest

import telohelm.runtime.manifest as F; import telohelm.runtime.models as M; import telohelm.runtime.exchange as X
from telohelm.runtime.trusted_log import verify
from .test_exchange_join import Wire; from .test_exchange_join import adapter; from .test_exchange_join import checked; from .test_exchange_join import journal_rows; from .test_exchange_join import CONFIG
from .test_manifest import invented
from .test_turn65_review_codex import gateway; from .test_turn65_review_codex import submit  # noqa: F401 (fixture)
from .test_turn84_review_codex import req


class ObservedWire(Wire):
    def __init__(self):
        super().__init__()
        self.requests = []

    def exchange(self, url, body, timeout_s):
        self.requests.append((url, copy.deepcopy(body), timeout_s))
        return super().exchange(url, body, timeout_s)


def config_from(tmp_path, manifest):
    path = tmp_path / "expected"
    path.write_text(F.fingerprint(manifest))
    return F.checked(manifest, path)


# Fixed in TURN 87 (Claude): CFG86.MUTABLE: checked manifest and fingerprint remain publicly mutable
def test_checked_configuration_cannot_change_under_its_original_fingerprint(tmp_path):
    c = checked()
    original = c.fingerprint
    try:
        c.manifest["sampling"]["temperature"] = 0.8
    except (AttributeError, TypeError):
        pass
    assert c.fingerprint == original and c.section("sampling")["temperature"] == 0.0


# Fixed in TURN 87 (Claude): CFG86.MUTABLE: post-construction settings evade the frozen binding
def test_effective_adapter_setting_change_is_refused_before_contact(tmp_path):
    w = ObservedWire()
    a = adapter(w, X.ExchangeStore(tmp_path / "x"), reportable=True)
    try:
        a.temperature = 0.8
    except (AttributeError, TypeError):
        pass
    r = a.complete(req())
    assert (w.calls == 0 and r.status == "failed") or w.requests[0][1]["temperature"] == 0.0


# Fixed in TURN 87 (Claude): CFG86.EFFECTIVE: pinned client seed/reasoning settings are silently ignored
@pytest.mark.parametrize("setting", ["seed", "reasoning"])
def test_a_pinned_nondefault_setting_is_effective_or_refused(tmp_path, setting):
    m = invented()
    field, value = {"seed": ("seed", 23),
                    "reasoning": ("reasoning_effort", "medium")}[setting]
    if setting == "seed":
        m["sampling"]["seed"] = value  # client pin23, distinct from declared server defaultNone
    else:
        m["roles"]["reasoning"]["small_model"] = {field: value}
    w = ObservedWire()
    try:
        a = adapter(w, X.ExchangeStore(tmp_path / "x"), reportable=True, config=config_from(tmp_path, m))
    except (F.ManifestError, M.GatewayError):
        assert w.calls == 0
        return
    r = a.complete(req())
    assert (w.calls == 0 and r.status == "failed") or w.requests[0][1].get(field) == value


def test_explicit_client_limit_can_differ_from_declared_server_default(tmp_path):
    """A declared server default is not a client ceiling; this does not verify any real server default."""
    m = invented(); m["generation"]["max_tokens"]["small_model"] = 8
    w = ObservedWire()
    # Adapted in TURN 91 (Claude): a reportable adapter called with no gateway is now quarantined (Codex's
    # ADM90.DIRECT), so this limit check runs on an unreportable direct adapter.
    a = adapter(w, X.ExchangeStore(tmp_path / "x"), config=config_from(tmp_path, m))
    r = a.complete(req())
    assert r.status == "answered" and w.requests[0][1]["max_tokens"] == 64


# Fixed in TURN 87 (Claude): CFG86.EFFECTIVE: modelled latency pin is not bound to requests
def test_request_latency_outside_the_pin_is_refused_before_contact(gateway, tmp_path):
    w = Wire()
    gw, _, _, _ = gateway(provider=adapter(w, X.ExchangeStore(tmp_path / "x"), reportable=True))
    out = gw.submit("small_model", [("user", "invented request")], 256, 1, 10, 10, 9.0)
    assert out["status"] == "failed" and out["text"] is None and w.calls == 0


# Fixed in TURN 87 (Claude): ANCH86.GATE: an unanchored saved exchange returns usable reportable output
def test_cut_before_original_anchor_cannot_make_reportable_action_text(gateway, tmp_path, monkeypatch):
    w = Wire(); path = tmp_path / "x"
    a = adapter(w, X.ExchangeStore(path), reportable=True)
    gw, _, _, original = gateway(provider=a, journal=tmp_path / "j1")
    def stop(_):
        raise SystemExit("invented cut after store, before protected anchor")
    monkeypatch.setattr(a, "_anchor", stop)
    with pytest.raises(SystemExit):
        submit(gw)
    assert verify(original / "log", original / "heads")["ok"] and w.calls == 1
    b = adapter(w, X.ExchangeStore(path), reportable=True)
    after, _, _, resumed = gateway(provider=b, journal=tmp_path / "j2")
    out = submit(after)
    problems = M.verify_anchors(b._exchanges, [(original / "log", original / "heads"),
                                              (resumed / "log", resumed / "heads")])
    assert problems and w.calls == 1  # current verifier correctly finds absence, after usable output escaped
    assert out["status"] != "answered" and out["text"] is None and out["reportable"] is False


# Fixed in TURN 87 (Claude): ANCH86.BIND: original metadata/body/role/wire time are outside saved anchors
@pytest.mark.parametrize("change", ["goal_meta", "endpoint", "body_sha256", "elapsed_s", "role"])
def test_protected_anchor_detects_changed_original_evidence(gateway, tmp_path, change):
    path, journal = tmp_path / "x", tmp_path / "j"
    a = adapter(Wire(), X.ExchangeStore(path), reportable=True)
    gw, _, _, root = gateway(provider=a, journal=journal)
    submit(gw)
    ex = json.loads(path.read_text()); rows = journal_rows(journal)
    if change == "goal_meta":
        ex["request"]["meta"]["goal_version"] = 2
        for r in rows:
            r["meta"]["goal_version"] = 2
    elif change == "endpoint":
        ex["request"]["endpoint"] = "http://127.0.0.1:10/v1/chat/completions"
    elif change == "body_sha256":
        ex["body_sha256"] = "d" * 64
    elif change == "elapsed_s":
        ex["elapsed_s"] = 7.0
    else:
        ex["role"] = "reasoning_model"
    path.write_text(json.dumps(ex) + "\n"); journal.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert verify(root / "log", root / "heads")["ok"]  # original protected evidence is untouched
    assert M.verify_anchors(X.ExchangeStore(path), [(root / "log", root / "heads")])


# Fixed in TURN 87 (Claude): JOIN86.IDENTITY: extraction accepts mutually changed store/journal meta
def test_extraction_requires_original_protected_request_not_just_pair_agreement(gateway, tmp_path):
    path, journal = tmp_path / "x", tmp_path / "j"
    a = adapter(Wire(), X.ExchangeStore(path), reportable=True)
    gw, _, _, root = gateway(provider=a, journal=journal); submit(gw)
    ex = json.loads(path.read_text()); rows = journal_rows(journal)
    ex["request"]["meta"]["goal_version"] = 2
    for r in rows:
        r["meta"]["goal_version"] = 2
    path.write_text(json.dumps(ex) + "\n"); journal.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert verify(root / "log", root / "heads")["ok"]
    reloaded = X.ExchangeStore(path)
    assert M.verify_anchors(reloaded, [(root / "log", root / "heads")]) or M.verify_extraction(a, reloaded, journal)


# Fixed in TURN 87 (Claude): EX86.COPY: returned nested identity aliases the committed in-memory row
@pytest.mark.parametrize("read", ["get", "records"])
def test_returned_record_cannot_mutate_the_original_identity(tmp_path, read):
    path = tmp_path / "x"; store = X.ExchangeStore(path); a = adapter(Wire(), store)
    assert a.complete(req()).status == "answered"
    original = X.ExchangeStore(path).records()[0]
    row = store.records()[0] if read == "records" else store.get(
        req().request_id, req().attempt_id, a.config_sha256, original["body_sha256"], original["request"])
    try:
        row["request"]["meta"]["goal_version"] = 2
    except (AttributeError, TypeError):
        pass
    assert store.records()[0]["request"] == X.ExchangeStore(path).records()[0]["request"]


# Fixed in TURN 87 (Claude): EX86.TAIL: complete JSON without final newline is admitted then concatenated
def test_reopen_of_unterminated_line_cannot_corrupt_the_next_append(tmp_path):
    path = tmp_path / "x"; store = X.ExchangeStore(path)
    store.save("r", "a", CONFIG, "small_model", "b" * 64, Wire().exchange("u", {}, 1.0))
    path.write_bytes(path.read_bytes().rstrip(b"\n"))  # invented partial write lost the line terminator only
    try:
        reopened = X.ExchangeStore(path)
    except X.ExchangeError:
        return  # quarantining the ambiguous tail is acceptable
    try:
        reopened.save("r2", "a", CONFIG, "small_model", "b" * 64, Wire().exchange("u", {}, 1.0))
    except X.ExchangeError:
        return
    assert len(X.ExchangeStore(path).records()) == 2  # or explicitly reconcile the terminator before appending


def test_fully_pinned_nominal_body_and_original_anchors_agree(gateway, tmp_path):
    w = ObservedWire(); store = X.ExchangeStore(tmp_path / "x")
    a = adapter(w, store, reportable=True)
    gw, ledger, _, root = gateway(provider=a); out = submit(gw)
    assert out["status"] == "answered" and w.calls == 1 and ledger.calls[-1].wall_s is not None
    assert w.requests[0][1] == {"model": "invented-small", "messages": [{"role": "user", "content": "invented request"}],
                              "max_tokens": 256, "temperature": 0.0}
    assert M.verify_anchors(store, [(root / "log", root / "heads")]) == []
    assert M.verify_extraction(a, store, root / "journal") == []


def test_changed_frozen_config_is_refused_even_when_new_arguments_would_match(tmp_path):
    c = checked()
    m = c.section("sampling"); m["temperature"] = 0.1  # section() returns a copy
    assert c.section("sampling")["temperature"] == 0.0
    altered = copy.deepcopy(invented()); altered["sampling"]["temperature"] = 0.1
    expected = tmp_path / "expected"; expected.write_text(c.fingerprint)
    with pytest.raises(F.ManifestError, match="does not match"):
        F.checked(altered, expected)


def test_completed_journal_recovery_keeps_original_cost_and_no_new_wire_call(gateway, tmp_path):
    w = Wire(); a = adapter(w, X.ExchangeStore(tmp_path / "x"), reportable=True)
    clock = iter([10.0, 12.0]); journal = tmp_path / "j"
    gw, ledger, _, original = gateway(provider=a, journal=journal, clock=lambda: next(clock))
    live = submit(gw); assert ledger.calls[-1].wall_s == 2.0
    after, fresh, _, resumed = gateway(provider=adapter(w, X.ExchangeStore(tmp_path / "x")), journal=journal)
    out = submit(after)
    assert out["status"] == live["status"] and out["recovered"] and w.calls == 1
    assert fresh.calls[-1].wall_s == 2.0 and sum(c.tokens_in or 0 for c in fresh.calls) == 11
    assert M.verify_anchors(X.ExchangeStore(tmp_path / "x"), [(original / "log", original / "heads"),
                                                            (resumed / "log", resumed / "heads")]) == []
