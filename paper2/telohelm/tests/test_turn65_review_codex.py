"""TURN65 exact repair review. Invented CPU controls and stub providers only; no research collection."""
from __future__ import annotations

import hashlib
import json

import pytest

import telohelm.runtime.broker as B; import telohelm.runtime.models as M
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.trusted_log import TrustedLog; from telohelm.runtime.trusted_log import verify
from .test_broker_channel import tool
from .test_turn59_review_codex import flight

MSG = [("user", "invented request")]
REQUIRED_LOGICAL_KINDS = {"briefing", "bundle", "decision", "action", "world_step", "prediction_start",
                          "prediction_end", "prediction_interrupted", "reply", "call_close", "feedback", "episode_end"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def answer(request):
    return M.ModelResponse("answered", f"invented answer under goal {request.goal_version}", 7, 2)


@pytest.fixture
def gateway(tmp_path):
    counter = [0]

    def make(provider=None, clock=None, journal=None, ledger=None, stream="stream-a", episode="episode-a", replay=False,
             start=True):
        counter[0] += 1
        root = tmp_path / str(counter[0]); root.mkdir()
        provider = provider if provider is not None else M.ScriptedProvider(answer)
        ledger = ledger if ledger is not None else Ledger()
        kw = {} if clock is None else {"clock": clock}
        gw = M.ModelGateway(episode, "condition-a", stream, provider,
                            TrustedLog(root / "log", root / "heads", "turn65"), ledger,
                            journal if journal is not None else root / "journal", replay=replay, **kw)
        # Adapted in TURN 97 (Claude): a store-backed gateway now refuses fresh calls until its start is admitted
        # (Codex's SG96.3), so the fixture performs an invented, matching start step unless a test passes start=False.
        if start and getattr(provider, "_exchanges", None) is not None:
            from .test_startup import start_gateway
            assert start_gateway(gw, provider) == []
        return gw, ledger, provider, root
    return make


def submit(gw, goal=1):
    return gw.submit("small_model", MSG, 256, goal, 10, 10, 2.4)


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("replay", [False, True])
def test_received_response_survives_end_clock_stop_in_shared_ledger(gateway, tmp_path, exc_type, replay):
    journal = tmp_path / "shared"
    ledger = Ledger()
    ticks = [0]
    original = exc_type("invented late clock stop")

    def clock():
        ticks[0] += 1
        if ticks[0] == 2:
            raise original
        return 4.
    first, _, provider, root = gateway(clock=clock, journal=journal, ledger=ledger, replay=replay)
    with pytest.raises(exc_type) as caught:
        submit(first)
    assert caught.value is original and provider.calls == 1
    saved = json.loads(journal.read_text().splitlines()[0])
    assert saved["response"]["text"] == "invented answer under goal 1"
    assert saved["response"]["tokens_in"] == 7 and saved["reportable"] is False
    assert saved["wall_s"] is None and verify(root / "log", root / "heads")["ok"]
    recovered, _, replacement, rroot = gateway(journal=journal, ledger=ledger, replay=replay)
    out = submit(recovered)
    assert replacement.calls == 0 and out["recovered"] and out["reportable"] is False
    assert sum(c.tokens_in or 0 for c in ledger.calls) == 7
    assert all(c.wall_s is None for c in ledger.calls)
    assert all(c.scope == ("replay" if replay else "running") for c in ledger.calls)
    assert verify(rroot / "log", rroot / "heads")["ok"]


# Fixed in TURN 66 (Claude): MG65.COST: already-accounted response tokens charged again on recovery
@pytest.mark.parametrize("status", ["answered", "failed"])
def test_already_accounted_answer_or_failure_is_not_charged_twice(gateway, tmp_path, status):
    journal = tmp_path / "shared"
    ledger = Ledger()
    response = M.ModelResponse(status, "invented" if status == "answered" else None, 7, 2)
    first, _, _, _ = gateway(journal=journal, ledger=ledger, provider=M.ScriptedProvider(lambda r: response))
    submit(first)
    resumed, _, provider, _ = gateway(journal=journal, ledger=ledger)
    assert submit(resumed)["recovered"] and provider.calls == 0
    assert sum(c.tokens_in or 0 for c in ledger.calls) == 7
    assert sum(c.tokens_out or 0 for c in ledger.calls) == 2


# Fixed in TURN 66 (Claude): MG65.COST: journal always stores null timing even after measured completion
def test_successfully_measured_duration_is_retained_for_recovery(gateway, tmp_path):
    journal = tmp_path / "shared"
    ticks = iter([4., 6.5])
    first, ledger, _, _ = gateway(journal=journal, clock=lambda: next(ticks))
    submit(first)
    assert ledger.calls[0].wall_s == 2.5
    resumed, recovered_ledger, provider, _ = gateway(journal=journal)
    submit(resumed)
    assert provider.calls == 0
    assert recovered_ledger.calls[0].wall_s == 2.5


# Fixed in TURN 66 (Claude): MG65.ORIGIN: provider replay of answers without provenance defaults reportable
def test_provider_replay_does_not_promote_scripted_or_unknown_origin(gateway):
    first, ledger, _, _ = gateway()
    out = submit(first)
    assert out["reportable"] is False
    rid = ledger.calls[0].request_id
    request = M.ModelRequest(rid, "attempt1", "episode-a", "condition-a", "stream-a", "small_model",
                             tuple(MSG), 256, 1, 10, 10, 2.4)
    # TURN 69 (Claude): replay entries now carry the original request's details (MG67.ID.REPLAY); same content.
    stored = {(rid, request.fingerprint): {"response": M.ModelResponse("answered", out["text"], 7, 2),
                                           "meta": M.request_meta(request)}}
    replay, _, _, _ = gateway(provider=M.ReplayProvider(stored), replay=True)
    result = submit(replay)
    assert result["text"] == out["text"]
    assert result["reportable"] is False


# Fixed in TURN 66 (Claude): MG65.ORIGIN: reportable=true accepted from journal lacking provider origin
def test_incomplete_journal_origin_is_unreportable(gateway, tmp_path):
    journal = tmp_path / "shared"
    first, _, _, _ = gateway(journal=journal)
    submit(first)
    rec = json.loads(journal.read_text().splitlines()[0])
    rec.pop("provider"); rec["reportable"] = True
    journal.write_text(canonical(rec) + "\n")
    resumed, _, provider, _ = gateway(journal=journal)
    out = submit(resumed)
    assert provider.calls == 0 and out["recovered"]
    assert out["reportable"] is False


# Fixed in TURN 66 (Claude): MG65.ID: slash-joined identity tuples are not collision-free
def test_distinct_identifier_tuples_cannot_share_journal_keys(gateway, tmp_path):
    journal = tmp_path / "shared"
    first, left, _, _ = gateway(journal=journal, stream="a/b", episode="c")
    submit(first)
    second, right, provider, _ = gateway(journal=journal, stream="a", episode="b/c")
    submit(second)
    assert provider.calls == 1
    assert left.calls[0].request_id != right.calls[0].request_id


# Fixed in TURN 66 (Claude): MG65.ID: old-goal saved response is relabelled with current goal version
def test_old_goal_journal_response_cannot_be_relabelled_as_current(gateway, tmp_path):
    journal = tmp_path / "shared"
    first, _, _, _ = gateway(journal=journal)
    assert submit(first, goal=1)["text"] == "invented answer under goal 1"
    changed, _, _, _ = gateway(journal=journal)
    result = submit(changed, goal=2)
    assert not (result["recovered"] and result["text"] == "invented answer under goal 1"
                and result["goal_version"] == 2)


@pytest.fixture
def broker(tmp_path):
    counter = [0]

    def make(replay=False):
        counter[0] += 1
        root = tmp_path / f"broker{counter[0]}"; root.mkdir()
        ledger = Ledger()
        br = B.Broker("invented-boundary", TrustedLog(root / "blog", root / "heads", "turn65"),
                      ledger, tool, condition="invented", stream="invented", replay=replay)
        return br, ledger, root
    return make


def test_expanded_logical_kinds_and_actual_cut_hashes_are_independently_recomputed(broker, monkeypatch):
    assert REQUIRED_LOGICAL_KINDS <= set(B.LOGICAL_KINDS)
    result, br, _, root, _ = flight(broker, monkeypatch)
    prefix = "0" * 64; cuts = {}
    for line in (root / "blog").read_text().splitlines()[1:]:
        event = json.loads(line)["payload"]
        # Adapted in TURN 79 (Claude): the ending records are a new logical kind (AT78 cut evidence). They come after
        # the last world step, so the cut hashes checked below are unchanged; only the whole-log prefix includes them.
        if event["kind"] in REQUIRED_LOGICAL_KINDS | {"ending"}:
            body = {k: v for k, v in event.items() if k not in ("seq", "event_prefix_sha256")}
            prefix = hashlib.sha256((prefix + canonical(body)).encode()).hexdigest()
        if event["kind"] == "world_step":
            cuts[event["action_steps"]] = prefix
    assert prefix == br.logical_prefix
    assert all(c["logical_prefix_sha256"] == cuts[c["action_steps"]] for c in result["ending_cuts"])
    assert verify(root / "blog", root / "heads")["ok"]


@pytest.mark.parametrize("malformed", ["duplicate briefing", "broker sequence gap", "missing first bundle", "reversed bundles"])
def test_authenticated_but_semantically_bad_input_sequence_is_refused(tmp_path, malformed):
    briefing = {"kind": "briefing", "episode": "invented", "seq": 1, "briefing": {"units": "NM"}}
    events = [briefing]
    for step in range(2):
        bundle = {"t": step, "invented": True}
        events.append({"kind": "bundle", "episode": "invented", "seq": step + 2,
                       "action_step": step, "t": step, "bundle": bundle,
                       "bundle_sha256": hashlib.sha256(canonical(bundle).encode()).hexdigest()})
    if malformed == "duplicate briefing": events.append(dict(briefing, seq=4))
    elif malformed == "broker sequence gap": events[-1]["seq"] = 4
    elif malformed == "missing first bundle": events.pop(1); events[-1]["seq"] = 2
    else:
        events[1], events[2] = events[2], events[1]
        events[1]["seq"], events[2]["seq"] = 2, 3
    log = TrustedLog(tmp_path / "log", tmp_path / "heads", "turn65")
    for event in events: log.append(event)
    assert verify(tmp_path / "log", tmp_path / "heads")["ok"]
    with pytest.raises(B.BrokerError):
        B.load_inputs(tmp_path / "log", tmp_path / "heads", "invented")


def test_authenticated_unicode_inputs_roundtrip(tmp_path):
    log = TrustedLog(tmp_path / "log", tmp_path / "heads", "turn65")
    br = B.Broker("invented", log, Ledger(), tool)
    br.checkpoint("briefing", briefing={"text": "invented α"})
    bundle = {"t": 0, "text": "invented λ"}
    br.checkpoint("bundle", bundle=bundle, bundle_sha256=hashlib.sha256(canonical(bundle).encode()).hexdigest())
    br.end_episode()
    saved = B.load_inputs(tmp_path / "log", tmp_path / "heads", "invented")
    assert saved["briefing"] == {"text": "invented α"} and saved["bundles"][0]["bundle"] == bundle
