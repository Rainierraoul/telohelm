"""The model gateway with stub providers (TURN 60). Invented requests; no model, provider or server is contacted."""
from __future__ import annotations

import json

import pytest

import telohelm.runtime.models as M
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.trusted_log import TrustedLog; from telohelm.runtime.trusted_log import verify

MSG = [("system", "invented system text"), ("user", "invented question")]


def answer(req):
    return M.ModelResponse("answered", text=f"reply to {req.request_id}", tokens_in=12, tokens_out=3,
                           provider_ids={"id": "invented"})


@pytest.fixture
def make(tmp_path):
    n = [0]

    def factory(provider=None, replay=False, limits=None, clock=None, journal=None):
        n[0] += 1
        root = tmp_path / str(n[0])
        root.mkdir()
        ledger = Ledger()
        prov = provider or M.ScriptedProvider(answer)
        kw = {} if clock is None else {"clock": clock}
        gw = M.ModelGateway("invented-episode", "telohelm", "invented-stream", prov,
                            TrustedLog(root / "log", root / "heads", "invented-run"), ledger,
                            journal or root / "journal.jsonl", replay=replay, limits=limits, **kw)
        return gw, ledger, root, prov
    return factory


def events(root):
    return [json.loads(l)["payload"] for l in (root / "log").read_text().splitlines()[1:]]


def submit(gw, role="small_model", second=10, step=10, latency=2.4, **kw):
    return gw.submit(role, MSG, 256, 1, second, step, latency, **kw)


def test_answered_call_is_logged_before_and_after_and_charged_with_links(make):
    gw, ledger, root, prov = make()
    out = submit(gw)
    assert out["status"] == "answered" and out["available_second"] == 13 and out["available_step"] == 13
    assert out["reportable"] is False                                   # scripted output is never a result
    (c,) = ledger.calls
    assert (c.kind, c.action_step, c.answered_step, c.sim_second, c.condition, c.stream, c.tokens_in) == \
           ("small_model", 10, 13, 10, "telohelm", "invented-stream", 12)
    ev = events(root)
    assert [e["kind"] for e in ev] == ["gateway_open", "model_start", "model_end"]
    assert ev[1]["request"]["messages"] == [list(m) for m in MSG] and ev[1]["fingerprint"]
    assert all(e["reportable"] is False for e in ev)
    assert verify(root / "log", root / "heads")["ok"]


def test_the_request_is_durable_before_the_provider_runs(make):
    seen = []

    def checking(req):
        seen.append(events(root)[-1])
        return answer(req)
    gw, ledger, root, _ = make(provider=M.ScriptedProvider(checking))
    submit(gw)
    assert seen[0]["kind"] == "model_start" and seen[0]["request"]["request_id"] == ledger.calls[0].request_id


def test_limits_refuse_without_calling_or_charging(make):
    gw, ledger, root, prov = make(limits={"small_model": 2, "reasoning_model": 1})
    assert [submit(gw)["status"] for _ in range(3)] == ["answered", "answered", "refused"]
    assert submit(gw, role="reasoning_model")["status"] == "answered"
    assert submit(gw, role="reasoning_model")["refusal"] == "over_allowance"
    assert prov.calls == 3 and len(ledger.calls) == 3
    assert M.CALL_LIMITS == {"reasoning_model": 4, "small_model": 30}


@pytest.mark.parametrize("bad", [None, "text", M.ModelResponse("answered", text=None),
                                 M.ModelResponse("answered", text="x", tokens_in=-1),
                                 M.ModelResponse("answered", text="x", tokens_in=True),
                                 M.ModelResponse("interrupted"), M.ModelResponse("weird")])
def test_bad_provider_responses_become_failed_calls(make, bad):
    gw, ledger, _, _ = make(provider=M.ScriptedProvider(lambda r: bad))
    out = submit(gw)
    assert out["status"] == "failed" and ledger.calls[0].status == "failed"
    assert ledger.calls[0].answered_step == 13                            # a failure also arrives after L


def test_a_provider_exception_is_a_failed_call_and_reveals_nothing(make):
    def broken(req):
        raise RuntimeError("private provider detail")
    gw, ledger, root, _ = make(provider=M.ScriptedProvider(broken))
    out = submit(gw)
    assert out["status"] == "failed" and out["text"] is None
    assert "private" not in json.dumps(events(root))


@pytest.mark.parametrize("where", ["start clock", "provider", "end clock"])
def test_interruption_anywhere_charges_one_row_and_reraises(make, where):
    ticks = []

    def clock():
        ticks.append(1)
        if (where == "start clock" and len(ticks) == 1) or (where == "end clock" and len(ticks) == 2):
            raise KeyboardInterrupt(where)
        return float(len(ticks))

    def provider(req):
        if where == "provider":
            raise KeyboardInterrupt(where)
        return answer(req)
    gw, ledger, root, _ = make(provider=M.ScriptedProvider(provider), clock=clock)
    with pytest.raises(KeyboardInterrupt, match=where):
        submit(gw)
    (c,) = ledger.calls
    assert c.status == "interrupted" and c.answered_step is None
    assert (c.wall_s is None) == (where != "provider")
    (stop,) = [e for e in events(root) if e["kind"] == "model_interrupted"]
    assert stop["phase"] == where and not any(e["kind"] == "model_end" for e in events(root))


def test_a_saved_answer_is_recovered_without_calling_the_model_again(make, tmp_path):
    journal = tmp_path / "shared-journal.jsonl"
    gw, _, _, prov = make(journal=journal)
    first = submit(gw)
    gw2, ledger2, root2, prov2 = make(journal=journal)
    second = submit(gw2)
    assert prov.calls == 1 and prov2.calls == 0
    assert second["text"] == first["text"] and ledger2.calls[0].status == "answered"
    assert [e for e in events(root2) if e["kind"] == "model_end"][0]["recovered"] is True


def test_a_changed_request_is_not_recovered_from_the_journal(make, tmp_path):
    journal = tmp_path / "shared-journal.jsonl"
    gw, _, _, _ = make(journal=journal)
    submit(gw)
    gw2, _, _, prov2 = make(journal=journal)
    gw2.submit("small_model", [("user", "a different question")], 256, 1, 10, 10, 2.4)
    assert prov2.calls == 1


def test_replay_provider_matches_identity_and_fingerprint(make):
    gw, ledger, _, _ = make()
    live = submit(gw)
    req_id = ledger.calls[0].request_id
    probe = M.ModelRequest(req_id, "attempt1", "invented-episode", "telohelm", "invented-stream", "small_model",
                           tuple(MSG), 256, 1, 10, 10, 2.4)
    stored = {(req_id, probe.fingerprint): {"response": M.ModelResponse("answered", text=live["text"], tokens_in=12,
                                                                     tokens_out=3), "meta": M.request_meta(probe)}}
    rgw, rledger, _, rprov = make(provider=M.ReplayProvider(stored), replay=True)
    assert submit(rgw)["text"] == live["text"] and rledger.calls[0].scope == "replay"
    rgw2, rledger2, _, _ = make(provider=M.ReplayProvider(stored), replay=True)
    rgw2.submit("small_model", [("user", "changed")], 256, 1, 10, 10, 2.4)
    assert rledger2.calls[0].status == "failed"                          # not in the store: never invented


@pytest.mark.parametrize("args", [
    ("oracle", MSG, 256, 1, 10, 10, 1.0), ("small_model", [], 256, 1, 10, 10, 1.0),
    ("small_model", [("user", 3)], 256, 1, 10, 10, 1.0), ("small_model", MSG, -1, 1, 10, 10, 1.0),
    ("small_model", MSG, 256, 1, 10, 10, float("nan")), ("small_model", MSG, 256, True, 10, 10, 1.0),
    ("small_model", [("user", "x" * (M.MAX_MESSAGES_BYTES + 1))], 256, 1, 10, 10, 1.0),
])
def test_impossible_requests_are_trusted_side_errors(make, args):
    gw, ledger, _, prov = make()
    with pytest.raises(M.GatewayError):
        gw.submit(*args)
    assert prov.calls == 0 and ledger.calls == []


def test_recovery_charges_tokens_once(make, tmp_path):
    """MG61.RECEIVED: after an end-clock interruption the interrupted row has no tokens; the row that recovers the
    saved answer carries them, under a distinct identifier, so nothing is charged twice."""
    journal = tmp_path / "shared-journal.jsonl"
    ticks = []

    def clock():
        ticks.append(1)
        if len(ticks) == 2:
            raise KeyboardInterrupt("end clock")
        return 1.0
    shared = Ledger()
    gw, _, _, prov = make(journal=journal, clock=clock)
    gw._ledger = shared
    with pytest.raises(KeyboardInterrupt):
        submit(gw)
    gw2, _, _, prov2 = make(journal=journal)
    gw2._ledger = shared
    out = submit(gw2)
    first, second = shared.calls
    assert prov.calls == 1 and prov2.calls == 0 and out["recovered"] is True
    assert first.status == "interrupted" and first.tokens_in is None and first.wall_s is None
    assert second.request_id == first.request_id + "#recovered1" and second.tokens_in == 12
    assert second.recovers == first.request_id


def test_a_journal_record_without_origin_is_never_reportable(make, tmp_path):
    journal = tmp_path / "shared-journal.jsonl"
    gw, ledger, _, _ = make(journal=journal)
    submit(gw)
    rid = ledger.calls[0].request_id
    rec = json.loads(journal.read_text().splitlines()[0])
    rec.pop("reportable")
    rec.pop("provider")
    journal.write_text(json.dumps(rec) + "\n")
    gw2, _, _, _ = make(provider=M.ReplayProvider({}), journal=journal, replay=True)
    out = submit(gw2)
    assert out["recovered"] is True and out["reportable"] is False and out["request_id"] == rid



def test_repeated_recovery_never_charges_again_and_stays_out_of_time_shares(make, tmp_path):
    """MG65.COST: the original completed and charged; two later recoveries add linked rows with no tokens, and their
    copied wall time is left out of GPU time shares."""
    journal = tmp_path / "shared-journal.jsonl"
    ticks = iter([4.0, 6.5])
    shared = Ledger()
    gw, _, _, _ = make(journal=journal, clock=lambda: next(ticks))
    gw._ledger = shared
    submit(gw)
    for _ in range(2):
        g, _, _, prov = make(journal=journal)
        g._ledger = shared
        assert submit(g)["recovered"] and prov.calls == 0
    original, r1, r2 = shared.calls
    assert original.wall_s == 2.5 and r1.wall_s == r2.wall_s == 2.5
    assert (r1.request_id, r2.request_id) == (original.request_id + "#recovered1", original.request_id + "#recovered2")
    assert r1.recovers == r2.recovers == original.request_id
    assert sum(c.tokens_in or 0 for c in shared.calls) == 12
    from dataclasses import replace
    placed = Ledger()
    for c in shared.calls:
        placed.add(replace(c, started_s=0.0 if c.started_s is None else c.started_s, allocation="job", gpus=(0,)))
    shares = placed.gpu_shares(1.0, [("job", 0.0, 3600.0, 1)])["shares_h"]
    assert set(shares) == {original.request_id}


def test_a_replayed_answer_with_a_complete_origin_keeps_it(make):
    gw, ledger, _, _ = make()
    live = submit(gw)
    rid = ledger.calls[0].request_id
    probe = M.ModelRequest(rid, "attempt1", "invented-episode", "telohelm", "invented-stream", "small_model",
                           tuple(MSG), 256, 1, 10, 10, 2.4)
    good = M.ModelResponse("answered", text=live["text"], origin={"provider": "local-model", "reportable": True})
    partial = M.ModelResponse("answered", text=live["text"], origin={"reportable": True})
    for stored, expected in ((good, True), (partial, False)):
        entry = {"response": stored, "meta": M.request_meta(probe)}
        rgw, _, _, _ = make(provider=M.ReplayProvider({(rid, probe.fingerprint): entry}), replay=True)
        assert submit(rgw)["reportable"] is expected


@pytest.mark.parametrize("change", [{"step": 11, "second": 11}, {"latency": 3.0}, {"attempt": 2}])
def test_a_cached_answer_with_different_request_metadata_is_not_recovered(make, tmp_path, change):
    journal = tmp_path / "shared-journal.jsonl"
    gw, _, _, _ = make(journal=journal)
    submit(gw)
    gw2, _, _, prov2 = make(journal=journal)
    out = gw2.submit("small_model", MSG, 256, 1, change.get("second", 10), change.get("step", 10),
                     change.get("latency", 2.4), attempt=change.get("attempt", 1))
    assert prov2.calls == 1 and out["recovered"] is False



def test_a_fresh_ledger_recovering_twice_charges_once(make, tmp_path):
    """MG67.COST.FRESH: a fresh ledger has no record of the original charge, so its first recovery carries the charge
    and its second links to that row."""
    journal = tmp_path / "shared-journal.jsonl"
    gw, _, _, _ = make(journal=journal)
    submit(gw)
    fresh = Ledger()
    for _ in range(2):
        g, _, _, prov = make(journal=journal)
        g._ledger = fresh
        assert submit(g)["recovered"] and prov.calls == 0
    a, b = fresh.calls
    assert a.tokens_in == 12 and a.recovers is None and b.tokens_in is None and b.recovers == a.request_id
    assert len(fresh.ending_calls("invented-episode", 99)["counted"]) == 1


def test_identity_already_in_the_ledger_is_refused_before_the_provider(make, tmp_path):
    shared = Ledger()
    gw, _, _, _ = make()
    gw._ledger = shared
    submit(gw)
    g2, _, _, prov2 = make()
    g2._ledger = shared
    with pytest.raises(M.GatewayError):
        submit(g2)
    assert prov2.calls == 0 and len(shared.calls) == 1


def test_a_relaying_provider_needs_a_complete_origin_and_a_producer_cannot_promote(make):
    claim = {"provider": "local-model", "reportable": True}
    gw, _, root, _ = make(provider=M.ScriptedProvider(lambda r: M.ModelResponse("answered", "x", origin=claim)))
    assert submit(gw)["reportable"] is False
    assert all(e["reportable"] is False for e in events(root))
