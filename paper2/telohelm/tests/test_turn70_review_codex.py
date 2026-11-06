"""TURN70 exact snapshot review: invented CPU controls, no actual model/provider requests."""
from __future__ import annotations

import hashlib
import json

import pytest

import telohelm.runtime.broker as B; import telohelm.runtime.episode_loop as L; import telohelm.runtime.models as M
from telohelm.runtime.delivery import DelayedModels
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.library import Library
from telohelm.runtime.trusted_log import TrustedLog; from telohelm.runtime.trusted_log import verify
from .test_broker_channel import tool
from .test_delivery import joining_scenario
from .test_episode_loop import scenario
from .test_turn65_review_codex import MSG; from .test_turn65_review_codex import gateway; from .test_turn65_review_codex import submit
import telohelm.taskpacks.tanker.world.episode as E


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def flight(tmp_path, monkeypatch, *, sc=None, latency=1., steps=3, text=None, parser=None, binding=None):
    """Actual world/event loop, with an invented heading-only model request and stand-in chooser."""
    sc = sc or scenario()
    ledger = Ledger()
    identity = {"episode": sc.episode_id, "condition": "condition", "stream": "stream"}
    identity.update(binding or {})
    provider = M.ScriptedProvider(lambda r: M.ModelResponse(
        "answered", text if text is not None else canonical({"heading_deg": 123.}), 7, 2))
    gateway = M.ModelGateway(identity["episode"], identity["condition"], identity["stream"], provider,
                             TrustedLog(tmp_path / "glog", tmp_path / "gheads", "invented"), ledger,
                             tmp_path / "journal")
    models = DelayedModels(gateway, lambda d, r: "small_model",
                           lambda d, c, r: [("user", canonical(c))], {"small_model": latency},
                           parser or (lambda d, t: json.loads(t)))

    def choose(decision, ctx, library, **kw):
        return {"climb": "missing_tool", "tried": []} if decision == "heading" else {
            "use": "invented", "output": {}, "tried": [], "checks": []}
    monkeypatch.setattr(L, "choose", choose)
    br = B.Broker(sc.episode_id, TrustedLog(tmp_path / "blog", tmp_path / "bheads", "invented"),
                  ledger, tool, condition="condition", stream="stream")
    log = TrustedLog(tmp_path / "log", tmp_path / "heads", "invented")
    return sc, ledger, provider, models, br, log, steps


def execute(parts, tmp_path):
    sc, ledger, _, models, br, log, steps = parts
    return L.run_episode(sc, Library(tmp_path / "library"), models, log, ledger,
                         max_steps=steps, broker=br)


def records(path):
    return [json.loads(line)["payload"] for line in path.read_text().splitlines()[1:]]


# Fixed in TURN 71 (Claude): MG70.CUT.CHARGE: first recovery's tokens omitted from counted ending charges
@pytest.mark.parametrize("replay", [False, True])
def test_ending_cut_retains_the_charge_carried_by_first_recovery(gateway, tmp_path, replay):
    journal = tmp_path / "shared"
    ledger = Ledger()
    ticks = [0]
    stop = KeyboardInterrupt("invented receipt stop")

    def clock():
        ticks[0] += 1
        if ticks[0] == 2:
            raise stop
        return 4.
    first, _, _, _ = gateway(journal=journal, ledger=ledger, clock=clock, replay=replay)
    with pytest.raises(KeyboardInterrupt) as caught:
        submit(first)
    assert caught.value is stop
    for _ in range(2):
        resumed, _, provider, _ = gateway(journal=journal, ledger=ledger, replay=replay)
        assert submit(resumed)["recovered"] and provider.calls == 0
    assert sum(c.tokens_in or 0 for c in ledger.calls) == 7
    cut = ledger.ending_calls("episode-a", 12)
    assert len(cut["counted"]) == 1
    # An explicit charge-id view is also acceptable: bookkeeping cannot disappear from the ending's costs.
    charged = set(cut.get("charge_ids", cut["counted"]))
    assert sum(c.tokens_in or 0 for c in ledger.calls if c.request_id in charged) == 7
    assert sum(c.tokens_out or 0 for c in ledger.calls if c.request_id in charged) == 2
    assert ledger.calls[0].status == "interrupted" and ledger.calls[0].wall_s is None


# Fixed in TURN 71 (Claude): DL70.BIND: shared ledger does not establish gateway episode/condition/stream
@pytest.mark.parametrize("binding", [{"episode": "other-episode"}, {"condition": "other-condition"},
                                     {"stream": "other-stream"}])
def test_wrong_gateway_identity_is_refused_before_provider_work(tmp_path, monkeypatch, binding):
    parts = flight(tmp_path, monkeypatch, binding=binding)
    with pytest.raises(ValueError):
        execute(parts, tmp_path)
    assert parts[2].calls == 0 and parts[1].calls == []


# Fixed in TURN 71 (Claude): DL70.PARSE: ordinary parser error aborts loop after pending reply removal
@pytest.mark.parametrize("mode", ["invalid_json", "parser_error"])
def test_bad_reply_parse_is_logged_and_refused_without_aborting(tmp_path, monkeypatch, mode):
    def broken(decision, text):
        raise ValueError("invented private parser detail")
    parts = flight(tmp_path, monkeypatch, text="not JSON" if mode == "invalid_json" else "invented",
                    parser=None if mode == "invalid_json" else broken)
    result = execute(parts, tmp_path)
    assert len(result["actions"]) == 3 and "heading_deg" not in result["actions"][1]
    replies = [e for e in records(tmp_path / "log") if e["kind"] == "model_reply"]
    assert replies[0]["t"] == 1 and replies[0]["part"] is None
    assert verify(tmp_path / "log", tmp_path / "heads")["ok"]
    assert "invented private parser detail" not in canonical(records(tmp_path / "blog"))


@pytest.mark.parametrize("change", [{"goal_version": 2}, {"sim_second": 11}, {"action_step": 11},
                                    {"modelled_latency_s": 3.}, {"attempt": "attempt2"}])
def test_complete_replay_envelope_with_other_original_metadata_fails(gateway, change):
    rid = canonical(["stream-a", "episode-a", "condition-a", "small_model", 1])
    fingerprint = hashlib.sha256(canonical({"role": "small_model", "messages": [list(m) for m in MSG],
                                           "max_output_tokens": 256}).encode()).hexdigest()
    meta = {"goal_version": 1, "sim_second": 10, "action_step": 10, "modelled_latency_s": 2.4,
            "attempt": "attempt1", "max_output_tokens": 256}
    meta.update(change)
    response = M.ModelResponse("answered", "answer for a different original request", 7, 2,
                               origin={"provider": "invented-original", "reportable": True})
    gw, _, _, _ = gateway(provider=M.ReplayProvider({(rid, fingerprint): {"response": response, "meta": meta}}),
                           replay=True)
    out = submit(gw)
    assert out["status"] == "failed" and out["text"] is None and out["reportable"] is False


def test_complete_matched_replay_envelope_retains_origin_and_identity(gateway):
    rid = canonical(["stream-a", "episode-a", "condition-a", "small_model", 1])
    fingerprint = hashlib.sha256(canonical({"role": "small_model", "messages": [list(m) for m in MSG],
                                           "max_output_tokens": 256}).encode()).hexdigest()
    meta = {"goal_version": 1, "sim_second": 10, "action_step": 10, "modelled_latency_s": 2.4,
            "attempt": "attempt1", "max_output_tokens": 256}
    response = M.ModelResponse("answered", "invented original", 7, 2,
                               origin={"provider": "invented-original", "reportable": True})
    gw, ledger, _, _ = gateway(provider=M.ReplayProvider({(rid, fingerprint): {"response": response, "meta": meta}}),
                                replay=True)
    out = submit(gw)
    assert out["request_id"] == rid and out["goal_version"] == 1 and out["reportable"] is True
    assert out["text"] == "invented original" and ledger.calls[0].scope == "replay"


@pytest.mark.parametrize("latency, expected", [(0., 0), (.4, 1), (2.6, 3)])
def test_actual_reply_phase_and_single_charge_before_and_at_availability(tmp_path, monkeypatch, latency, expected):
    parts = flight(tmp_path, monkeypatch, latency=latency, steps=expected + 1)
    result = execute(parts, tmp_path)
    assert all("heading_deg" not in a for a in result["actions"][:expected])
    assert result["actions"][expected]["heading_deg"] == 123.
    assert parts[2].calls == 1 and len(parts[1].calls) == 1
    ev = records(tmp_path / "log")
    reply = next(e for e in ev if e["kind"] == "model_reply")
    assert reply["t"] == expected == reply["available_second"] and reply["issued_second"] == 0
    decision_index = next(i for i, e in enumerate(ev) if e["kind"] == "decision" and e["t"] == expected)
    assert ev.index(reply) < decision_index
    assert reply["subphase"] == ("post-issuance" if latency == 0 else "reply phase")
    assert verify(tmp_path / "blog", tmp_path / "bheads")["ok"]


def test_join_exactly_at_reply_second_prevents_delivery_and_keeps_charge(tmp_path, monkeypatch):
    parts = flight(tmp_path, monkeypatch, sc=joining_scenario(), latency=1., steps=4)
    result = execute(parts, tmp_path)
    assert result["outcome"]["ended_at"] == 1 and result["outcome"]["end_reason"] == "join"
    assert not any(e["kind"] == "model_reply" for e in records(tmp_path / "log"))
    undelivered = [e for e in records(tmp_path / "log") if e["kind"] == "model_undelivered"]
    assert len(undelivered) == 1 and undelivered[0]["available_second"] == 1
    cut = next(c for c in result["ending_cuts"] if c["ending"] == "nominal")
    assert len(cut["counted"]) == 1 and cut["straddling"] == cut["counted"]
    assert sum(c.tokens_in or 0 for c in parts[1].calls) == 7


def test_same_second_new_goal_refuses_old_reply_before_action(tmp_path, monkeypatch):
    base = scenario()
    sc = E.Scenario(episode_id=base.episode_id, receiver=base.receiver, wind=base.wind, tankers=base.tankers,
                    goals=((0, E.Goal(1, "SHELL 1", 20.)), (1, E.Goal(2, "SHELL 1", 15.))))
    parts = flight(tmp_path, monkeypatch, sc=sc, steps=2)
    result = execute(parts, tmp_path)
    reply = next(e for e in records(tmp_path / "log") if e["kind"] == "model_reply")
    assert reply["t"] == 1 and reply["goal_version"] == 1 and reply["stale"] and reply["part"] is None
    assert "heading_deg" not in result["actions"][1]


def test_fresh_recovered_cost_stays_unknown_in_placement_and_counts_once(gateway, tmp_path):
    journal = tmp_path / "shared"
    first, _, _, _ = gateway(journal=journal)
    submit(first)
    shared = Ledger()
    for _ in range(2):
        gw, _, provider, _ = gateway(journal=journal, ledger=shared)
        assert submit(gw)["recovered"] and provider.calls == 0
    assert sum(c.tokens_in or 0 for c in shared.calls) == 7
    assert len(shared.ending_calls("episode-a", 12)["counted"]) == 1
    shares = shared.gpu_shares(1., [("invented", 0., 3600., 1)])
    assert shares["busy_h"] is None and shares["shares_h"] is None
    assert shares["calls_unknown_time_or_placement"]
