"""TURN61 independent controls: invented requests/geometry, stub providers only; no collected research data."""
from __future__ import annotations

import hashlib
import json

import pytest

import telohelm.runtime.broker as B; import telohelm.runtime.models as M
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.trusted_log import TrustedLog; from telohelm.runtime.trusted_log import verify
from .test_broker_channel import tool
from .test_turn59_review_codex import flight

MSG = [("user", "invented review request")]


def answer(request):
    return M.ModelResponse("answered", "invented retained answer", 7, 2)


@pytest.fixture
def gateway(tmp_path):
    count = [0]

    def make(provider=None, clock=None, journal=None, stream="stream-a", replay=False, ledger=None):
        count[0] += 1
        root = tmp_path / f"gateway{count[0]}"
        root.mkdir()
        provider = provider if provider is not None else M.ScriptedProvider(answer)
        ledger = ledger if ledger is not None else Ledger()
        kw = {} if clock is None else {"clock": clock}
        gw = M.ModelGateway("review-episode", "invented-condition", stream, provider,
                            TrustedLog(root / "log", root / "heads", "turn61"), ledger,
                            journal if journal is not None else root / "journal", replay=replay, **kw)
        return gw, ledger, provider, root
    return make


def submit(gateway):
    return gateway.submit("small_model", MSG, 256, 1, 10, 10, 2.4)


def events(root):
    return [json.loads(line)["payload"] for line in (root / "log").read_text().splitlines()[1:]]


@pytest.mark.parametrize("where", ["start", "end"])
@pytest.mark.parametrize("replay", [False, True])
def test_ordinary_clock_unavailability_preserves_completed_answer(gateway, where, replay):
    ticks = [0]

    def clock():
        ticks[0] += 1
        if ticks[0] == (1 if where == "start" else 2):
            raise ValueError("invented unavailable clock")
        return 4.0
    gw, ledger, provider, root = gateway(clock=clock, replay=replay)
    out = submit(gw)
    assert out["status"] == "answered" and provider.calls == 1
    (row,) = ledger.calls
    assert row.wall_s is None and row.cpu_s is None
    assert row.scope == ("replay" if replay else "running")
    assert row.answered_step == 13 and row.tokens_in == 7 and row.tokens_out == 2
    assert len([e for e in events(root) if e["kind"] == "model_end"]) == 1
    assert verify(root / "log", root / "heads")["ok"]


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("replay", [False, True])
def test_provider_interruption_preserves_original_exception_and_one_charge(gateway, exc_type, replay):
    original = exc_type("invented stop")

    def stop(request):
        raise original
    gw, ledger, provider, root = gateway(provider=M.ScriptedProvider(stop), clock=lambda: 4., replay=replay)
    with pytest.raises(exc_type) as caught:
        submit(gw)
    assert caught.value is original and provider.calls == 1
    (row,) = ledger.calls
    assert row.status == "interrupted" and row.wall_s == 0. and row.answered_step is None
    assert row.scope == ("replay" if replay else "running")
    assert len([e for e in events(root) if e["kind"] == "model_interrupted"]) == 1
    assert verify(root / "log", root / "heads")["ok"]


# Fixed in TURN 62 (Claude): MG61.RECEIVED: completed response is not saved before end-clock interruption
@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
def test_completed_response_survives_later_clock_interruption_without_new_call(gateway, tmp_path, exc_type):
    journal = tmp_path / "shared-journal"
    ticks = [0]
    original = exc_type("after a completed response")

    def clock():
        ticks[0] += 1
        if ticks[0] == 2:
            raise original
        return 4.
    gw, ledger, provider, root = gateway(clock=clock, journal=journal)
    with pytest.raises(exc_type) as caught:
        submit(gw)
    assert caught.value is original and provider.calls == 1 and len(ledger.calls) == 1
    resumed, _, replacement, _ = gateway(journal=journal)
    out = submit(resumed)
    assert out["text"] == "invented retained answer"
    assert replacement.calls == 0, "already received response must be recovered, not requested again"


# Fixed in TURN 62 (Claude): MG61.IDENTITY: request/journal matching omits stream identity
def test_journal_does_not_recover_another_streams_response(gateway, tmp_path):
    journal = tmp_path / "shared-journal"
    first, first_ledger, _, _ = gateway(journal=journal, stream="stream-a")
    submit(first)
    other, other_ledger, provider, _ = gateway(journal=journal, stream="stream-b")
    submit(other)
    assert provider.calls == 1, "another stream's response is not this request's saved response"
    assert first_ledger.calls[0].request_id != other_ledger.calls[0].request_id


# Fixed in TURN 62 (Claude): MG61.IDENTITY: stream collision reaches duplicate-ledger error after start
def test_two_streams_have_distinct_identified_calls_in_a_shared_ledger(gateway):
    ledger = Ledger()
    first, _, _, _ = gateway(ledger=ledger, stream="stream-a")
    second, _, provider, _ = gateway(ledger=ledger, stream="stream-b")
    submit(first)
    submit(second)
    assert provider.calls == 1 and len(ledger.calls) == 2
    assert len({c.request_id for c in ledger.calls}) == 2


# Fixed in TURN 62 (Claude): MG61.ORIGIN: journal recovery promotes scripted answers to reportable
def test_scripted_answer_keeps_unreportable_origin_on_recovery(gateway, tmp_path):
    journal = tmp_path / "shared-journal"
    gw, _, _, _ = gateway(journal=journal)
    assert submit(gw)["reportable"] is False
    resumed, _, provider, root = gateway(provider=M.ReplayProvider({}), journal=journal, replay=True)
    out = submit(resumed)
    assert provider.calls == 0 and out["text"] == "invented retained answer"
    assert out["reportable"] is False
    end = [e for e in events(root) if e["kind"] == "model_end"]
    assert len(end) == 1 and end[0]["reportable"] is False


@pytest.fixture
def broker(tmp_path):
    count = [0]

    def make(replay=False):
        count[0] += 1
        root = tmp_path / f"broker{count[0]}"
        root.mkdir()
        ledger = Ledger()
        br = B.Broker("invented-boundary", TrustedLog(root / "blog", root / "heads", "turn61"),
                      ledger, tool, replay=replay, condition="invented", stream="invented")
        return br, ledger, root
    return make


def broker_events(root):
    return [json.loads(line)["payload"] for line in (root / "blog").read_text().splitlines()[1:]]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def test_saved_public_inputs_and_independent_logical_cut_recomputation(broker, monkeypatch):
    result, br, ledger, root, received = flight(broker, monkeypatch)
    stored = B.load_inputs(root / "blog", root / "heads", "invented-boundary")  # TURN 62: authenticated signature
    assert [item["bundle"] for item in stored["bundles"]] == received
    assert [item["action_step"] for item in stored["bundles"]] == [0, 1, 2]
    ev = broker_events(root)
    assert stored["briefing"] == next(e["briefing"] for e in ev if e["kind"] == "briefing")
    prefix = "0" * 64
    cuts = {}
    for e in ev:
        if e["kind"] in B.LOGICAL_KINDS:  # TURN 62: the logical kinds now include outcomes (RP61.OUTCOME)
            body = {k: v for k, v in e.items() if k not in ("seq", "event_prefix_sha256")}
            prefix = hashlib.sha256((prefix + canonical(body)).encode()).hexdigest()
        if e["kind"] == "world_step":
            cuts[e["action_steps"]] = prefix
    assert prefix == br.logical_prefix
    assert all(c["logical_prefix_sha256"] == cuts[c["action_steps"]] for c in result["ending_cuts"])
    assert verify(root / "blog", root / "heads")["ok"] and len(ledger.calls) == 3


# Fixed in TURN 62 (Claude): RP61.AUTH: loader does not authenticate briefing/full sequence against protected heads
@pytest.mark.parametrize("change", ["briefing", "bundle and digest", "missing bundle"])
def test_reconstruction_rejects_unauthenticated_saved_inputs(broker, monkeypatch, change):
    _, _, _, root, _ = flight(broker, monkeypatch)
    records = [json.loads(line) for line in (root / "blog").read_text().splitlines()]
    if change == "briefing":
        item = next(r["payload"] for r in records if r["payload"]["kind"] == "briefing")
        item["briefing"]["review_injected"] = "invented altered briefing"
    elif change == "bundle and digest":
        item = next(r["payload"] for r in records if r["payload"]["kind"] == "bundle")
        item["bundle"]["t"] = 999
        item["bundle_sha256"] = hashlib.sha256(canonical(item["bundle"]).encode()).hexdigest()
    else:
        records.remove(next(r for r in records if r["payload"]["kind"] == "bundle"))
    (root / "blog").write_text("\n".join(canonical(r) for r in records) + "\n")
    assert not verify(root / "blog", root / "heads")["ok"]
    # A reviewed verified loader/wrapper may replace this call; protected heads must be a required trust input.
    with pytest.raises(B.BrokerError):
        B.load_inputs(root / "blog", root / "heads", "invented-boundary")  # TURN 62: authenticated signature


# Fixed in TURN 62 (Claude): RP61.OUTCOME: logical prefix omits monitor outcomes/trial/replay/undo events
@pytest.mark.parametrize("kind,first,second", [
    ("episode_end", {"goal_met": True, "failed_part": None}, {"goal_met": False, "failed_part": "join"}),
    ("replay_test", {"entry": "a" * 64, "passed": True}, {"entry": "a" * 64, "passed": False}),
    ("trial", {"entry": "a" * 64, "passed": True}, {"entry": "a" * 64, "passed": False}),
    ("undo", {"entry": "a" * 64, "reason": B.UNDO_REASONS[0]},
             {"entry": "a" * 64, "reason": B.UNDO_REASONS[1]}),
])
def test_logical_comparison_distinguishes_different_public_monitor_outcomes(broker, kind, first, second):
    a, _, _ = broker()
    b, _, _ = broker()
    for br in (a, b):
        br.checkpoint("action", action={"heading_deg": 90.})
        br.end_episode()
    a.feedback(kind, **first)
    b.feedback(kind, **second)
    assert a.event_prefix != b.event_prefix
    assert a.logical_prefix != b.logical_prefix


# Fixed in TURN 62 (Claude): RP61.OUTCOME: logical prefix omits procedure close/refusal outcome
def test_logical_comparison_distinguishes_procedure_format_failure(broker):
    a, _, _ = broker()
    b, _, _ = broker()
    a.close_call(a.open_call("heading:invented"), "completed")
    b.close_call(b.open_call("heading:invented"), "bad_reply")
    assert a.event_prefix != b.event_prefix
    assert a.logical_prefix != b.logical_prefix
