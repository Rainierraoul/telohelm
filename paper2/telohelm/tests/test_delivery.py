"""Delayed delivery of model replies in the episode loop (TURN 69). The real world with invented scenarios, a stand-in
chooser that forces the model rung, and the scripted provider. No model, provider or server is contacted."""
from __future__ import annotations

import hashlib
import json
import math

import pytest

import telohelm.runtime.broker as B
import telohelm.runtime.episode_loop as L
import telohelm.runtime.models as M
from telohelm.runtime.delivery import DelayedModels
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.library import Library
from telohelm.runtime.trusted_log import TrustedLog
from .test_broker_channel import tool
from .test_episode_loop import scenario
import telohelm.taskpacks.tanker.world.episode as E
import telohelm.taskpacks.tanker.world.physics as P


def joining_scenario():
    """A stationary receiver on a stationary tanker: the join is found over [0, 1], so the episode ends at second 1."""
    orbit = P.Orbit(5.0, 5.0, 1.0, 0.0, 0.0)
    x, y = orbit.position(0.0)
    rcv = P.ReceiverState(0.0, x, y, 0.0, 0.0, 0.0, 0.0)
    tankers = (E.Tanker("SHELL 1", orbit), E.Tanker("SHELL 2", P.Orbit(40.0, 40.0, 1.0, 0.0, 0.0)))
    return E.Scenario(episode_id="loop-test", receiver=rcv, wind=(0.0, 0.0), tankers=tankers,
                      goals=((0, E.Goal(1, "SHELL 1", 20.0)),))


def run(tmp_path, latency=2.4, steps=6, sc=None, decisions=("heading",), answer=None, limits=None, journal=None,
        broker=False, name="r", ledger=None, monkeypatch=None):
    root = tmp_path / name
    root.mkdir(parents=True)
    sc = sc or scenario()
    ledger = ledger if ledger is not None else Ledger()
    seen_ctx = []
    answer = answer or (lambda req: M.ModelResponse("answered", json.dumps({"heading_deg": 123.0}), 5, 1))
    provider = M.ScriptedProvider(answer)
    gw = M.ModelGateway(sc.episode_id, "telohelm", "invented-stream", provider,
                        TrustedLog(root / "glog", root / "gheads", "invented"), ledger,
                        journal or root / "journal", limits=limits)

    def messages(decision, ctx, reason):
        seen_ctx.append(json.loads(json.dumps(ctx)))
        return [("user", json.dumps({"decision": decision, "t": ctx["t"]}))]
    models = DelayedModels(gw, role_for=lambda d, r: "small_model", messages_for=messages,
                           latency_s={"small_model": latency, "reasoning_model": 60.0},
                           parse=lambda d, text: json.loads(text) if d == "heading" else {"report": "invented"})

    def choose(decision, ctx, library, **kw):
        if decision in decisions:
            return {"climb": "missing_tool", "tried": []}
        return {"use": "invented", "output": {}, "tried": [], "checks": []}
    monkeypatch.setattr(L, "choose", choose)
    br = None
    if broker:
        br = B.Broker(sc.episode_id, TrustedLog(root / "blog", root / "bheads", "invented"), ledger, tool,
                      condition="telohelm", stream="invented-stream")
    log = TrustedLog(root / "log", root / "heads", "invented")
    result = L.run_episode(sc, Library(root / "lib"), models, log, ledger, max_steps=steps, broker=br)
    entries = [json.loads(l)["payload"] for l in (root / "log").read_text().splitlines()[1:]]
    return result, ledger, provider, entries, seen_ctx, br


def replies(entries):
    return [e for e in entries if e["kind"] == "model_reply"]


def test_a_reply_arrives_at_its_second_and_no_call_is_reissued_while_pending(tmp_path, monkeypatch):
    r, ledger, provider, entries, _, _ = run(tmp_path, latency=2.4, steps=6, monkeypatch=monkeypatch)
    (first, *_) = replies(entries)
    assert first["available_second"] == 3 and first["t"] == 3 and first["subphase"] == "reply phase"
    assert [a.get("heading_deg") for a in r["actions"][:4]] == [None, None, None, 123.0]
    assert provider.calls == 2                         # one at second 0, the next at second 4 after delivery
    c = ledger.calls[0]
    assert (c.action_step, c.answered_step, c.sim_second) == (0, 3, 0)


@pytest.mark.parametrize("latency, second", [(0.0, 0), (0.4, 1), (1.0, 1), (2.6, 3)])
def test_latency_rounds_up_and_zero_uses_the_post_issuance_subphase(tmp_path, monkeypatch, latency, second):
    r, _, _, entries, _, _ = run(tmp_path, latency=latency, steps=4, monkeypatch=monkeypatch)
    first = replies(entries)[0]
    assert first["available_second"] == second == math.ceil(latency)
    assert first["subphase"] == ("post-issuance" if latency == 0 else "reply phase")
    assert r["actions"][second].get("heading_deg") == 123.0


def test_a_goal_change_in_the_reply_second_makes_the_reply_out_of_date(tmp_path, monkeypatch):
    base = scenario()
    sc = E.Scenario(episode_id=base.episode_id, receiver=base.receiver, wind=base.wind, tankers=base.tankers,
                    goals=((0, E.Goal(1, "SHELL 1", 20.0)), (3, E.Goal(2, "SHELL 1", 15.0))))
    r, _, _, entries, _, _ = run(tmp_path, latency=2.4, steps=5, sc=sc, monkeypatch=monkeypatch)
    first = replies(entries)[0]
    assert first["available_second"] == 3 and first["goal_version"] == 1 and first["stale"] is True
    assert first["part"] is None and "heading_deg" not in r["actions"][3]


def test_a_reply_due_after_the_ending_is_never_delivered_and_straddles(tmp_path, monkeypatch):
    r, ledger, _, entries, _, _ = run(tmp_path, latency=2.4, steps=10, sc=joining_scenario(), monkeypatch=monkeypatch)
    assert r["outcome"]["end_reason"] == "join" and r["outcome"]["ended_at"] == 1
    assert replies(entries) == []
    (u,) = [e for e in entries if e["kind"] == "model_undelivered"]
    assert u["available_second"] == 3
    cut = ledger.ending_calls("loop-test", r["outcome"]["action_steps"])
    assert cut["counted"] == [ledger.calls[0].request_id] == cut["straddling"]


def test_a_failure_is_delivered_after_the_same_latency(tmp_path, monkeypatch):
    r, ledger, _, entries, _, _ = run(tmp_path, latency=2.4, steps=4,
                                      answer=lambda req: M.ModelResponse("failed"), monkeypatch=monkeypatch)
    first = replies(entries)[0]
    assert first["available_second"] == 3 and first["status"] == "failed" and first["part"] is None
    assert ledger.calls[0].status == "failed" and ledger.calls[0].answered_step == 3


def test_replies_due_together_come_in_the_order_their_calls_were_made(tmp_path, monkeypatch):
    r, _, _, entries, _, _ = run(tmp_path, latency=2.4, steps=4, decisions=("heading", "report"),
                                 monkeypatch=monkeypatch)
    at3 = [e for e in replies(entries) if e["t"] == 3]
    assert [e["decision"] for e in at3] == ["heading", "report"]


def test_the_reply_keeps_the_context_and_tags_it_was_issued_under(tmp_path, monkeypatch):
    r, _, _, entries, seen, _ = run(tmp_path, latency=2.4, steps=4, monkeypatch=monkeypatch)
    first = replies(entries)[0]
    expected = hashlib.sha256(json.dumps(seen[0], sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                         allow_nan=False).encode()).hexdigest()
    assert first["context_sha256"] == expected and first["issued_second"] == 0 and first["issued_step"] == 0


def test_the_allowance_carries_across_seconds_and_refused_calls_are_not_charged(tmp_path, monkeypatch):
    r, ledger, provider, entries, _, _ = run(tmp_path, latency=2.4, steps=8,
                                            limits={"small_model": 1, "reasoning_model": 4}, monkeypatch=monkeypatch)
    assert provider.calls == 1 and len(ledger.calls) == 1 and len(replies(entries)) == 1


def test_deliveries_are_broker_checkpoints_in_the_logical_prefix(tmp_path, monkeypatch):
    a = run(tmp_path, latency=2.4, steps=4, broker=True, name="a", monkeypatch=monkeypatch)
    b = run(tmp_path, latency=2.4, steps=4, broker=True, name="b", monkeypatch=monkeypatch,
            answer=lambda req: M.ModelResponse("answered", json.dumps({"heading_deg": 200.0}), 5, 1))
    events = [json.loads(l)["payload"] for l in (tmp_path / "a" / "blog").read_text().splitlines()[1:]]
    assert any(e["kind"] == "model_reply" and e["request_id"] for e in events)
    assert a[5].logical_prefix != b[5].logical_prefix


def test_a_resumed_run_recovers_the_reply_without_a_second_call(tmp_path, monkeypatch):
    journal = tmp_path / "shared-journal"
    first = run(tmp_path, latency=2.4, steps=4, journal=journal, name="first", monkeypatch=monkeypatch)
    second = run(tmp_path, latency=2.4, steps=4, journal=journal, name="second", monkeypatch=monkeypatch)
    assert first[2].calls == 1 and second[2].calls == 0
    assert replies(second[3])[0]["part"] == {"heading_deg": 123.0}
    assert sum(c.tokens_in or 0 for c in second[1].calls) == 5           # charged once in the fresh ledger


def test_a_gateway_with_another_ledger_is_refused(tmp_path, monkeypatch):
    sc = scenario()
    gw = M.ModelGateway(sc.episode_id, "telohelm", "s", M.ScriptedProvider(lambda r: M.ModelResponse("failed")),
                        TrustedLog(tmp_path / "g", tmp_path / "gh", "r"), Ledger(), tmp_path / "j")
    models = DelayedModels(gw, lambda d, r: "small_model", lambda d, c, r: [("user", "x")],
                           {"small_model": 1.0, "reasoning_model": 60.0}, lambda d, t: None)
    with pytest.raises(ValueError):
        L.run_episode(sc, Library(tmp_path / "lib"), models, TrustedLog(tmp_path / "l", tmp_path / "h", "r"),
                      Ledger(), max_steps=1)


def test_a_part_of_the_wrong_shape_is_refused_publicly(tmp_path, monkeypatch):
    r, _, _, entries, _, _ = run(tmp_path, latency=1.0, steps=3, monkeypatch=monkeypatch,
                                 answer=lambda req: M.ModelResponse("answered", json.dumps({"truth": 1}), 5, 1))
    first = replies(entries)[0]
    assert first["refusal"] == "invalid_part" and first["part"] is None and first["reply_sha256"]
    assert "heading_deg" not in r["actions"][1]


def test_an_unrelated_parser_fault_is_not_masked(tmp_path, monkeypatch):
    import telohelm.runtime.delivery as D
    original = D.DelayedModels.part_for

    def faulty(self, p, g, **kwargs):
        self.parse = lambda d, t: (_ for _ in ()).throw(RuntimeError("invented unrelated fault"))
        return original(self, p, g, **kwargs)
    monkeypatch.setattr(D.DelayedModels, "part_for", faulty)
    with pytest.raises(RuntimeError, match="invented unrelated fault"):
        run(tmp_path, latency=1.0, steps=3, monkeypatch=monkeypatch)
