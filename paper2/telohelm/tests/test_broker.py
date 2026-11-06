"""The trusted broker's contract (TURN 50, paper2/notes/BROKER-CONTRACT.md). CPU-only, stubbed prediction tool,
invented requests. No episode, model, provider or cluster is used."""
from __future__ import annotations

import json

import pytest

import telohelm.runtime.broker as B
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.trusted_log import TrustedLog; from telohelm.runtime.trusted_log import verify


def predict_stub(args):
    """An invented, deterministic stand-in for the prediction tool."""
    n = len(args["headings_deg"])
    return {"positions_nm": [[args["start"]["x_nm"] + k, args["start"]["y_nm"]] for k in range(1, n + 1)],
            "closest_approach_nm": 1.5, "closest_at_s": float(n)}


def request(goal_version=0, **override):
    args = {"goal_version": goal_version,
            "start": {"x_nm": 0.0, "y_nm": 0.0, "heading_deg": 90.0, "tas_nm_s": 0.125},
            "headings_deg": [90.0, 95.0, 100.0], "wind_nm_s": [0.01, -0.005],
            "orbit": {"cx_nm": 40.0, "cy_nm": 30.0, "radius_nm": 8.0, "speed_nm_s": 0.0778, "angle_rad": 1.0}}
    args.update(override)
    return json.dumps({"capability": "predict", "args": args})


@pytest.fixture
def setup(tmp_path):
    def make(replay=False, allowance=B.PREDICTION_ALLOWANCE, predict=predict_stub, name="log"):
        log = TrustedLog(tmp_path / f"{name}.jsonl", tmp_path / f"{name}.heads", "invented-run")
        ledger = Ledger()
        clock = iter(float(i) for i in range(10_000))
        br = B.Broker("invented-episode", log, ledger, predict, replay=replay, allowance=allowance,
                      clock=lambda: next(clock))
        return br, log, ledger
    return make


def test_a_valid_prediction_is_answered_counted_logged_and_charged(setup, tmp_path):
    br, log, ledger = setup()
    br.at(t=12, action_step=13, goal_version=0)
    with br.open_call("heading") as call:
        reply = json.loads(call.request(request()))
    assert reply["ok"] and reply["remaining"] == B.PREDICTION_ALLOWANCE - 1
    assert reply["result"]["closest_approach_nm"] == 1.5
    assert br.predictions_used == 1
    (c,) = ledger.calls
    assert c.kind == "prediction_tool" and c.action_step == 13 and c.status == "answered" and c.scope == "running"
    events = [json.loads(l)["payload"] for l in (tmp_path / "log.jsonl").read_text().splitlines()[1:]]
    assert [e["kind"] for e in events] == ["broker_open", "call_open", "prediction_start", "prediction_end", "reply",
                                           "call_close"]
    assert all(e["episode"] == "invented-episode" and "event_prefix_sha256" in e for e in events)
    assert events[2]["request"] == json.loads(request())             # the exact validated request (BR51.6)
    assert events[3]["status"] == "answered" and events[3]["request_id"] == events[2]["request_id"]
    assert events[4]["t"] == 12 and events[4]["action_step"] == 13
    assert verify(tmp_path / "log.jsonl", tmp_path / "log.heads")["ok"]


@pytest.mark.parametrize("text, code", [
    ("not json", "malformed"),
    ("[1, 2]", "malformed"),
    (json.dumps({"capability": "predict"}), "malformed"),
    (json.dumps({"capability": "predict", "args": {}, "extra": 1}), "malformed"),
    (json.dumps({"capability": 7, "args": {}}), "malformed"),
    ('{"capability": "predict", "args": {"goal_version": NaN}}', "malformed"),
    ('{"capability": "predict", "args": {"goal_version": 1e999}}', "malformed"),
    ("[" * 100_000 + "]" * 100_000, "too_large"),
    ("[" * 20_000 + "]" * 20_000, "malformed"),
    ("x" * (B.MAX_MESSAGE_BYTES + 1), "too_large"),
    (json.dumps({"capability": "read_truth", "args": {}}), "unknown_capability"),
    (json.dumps({"capability": "write_log", "args": {"text": "x"}}), "unknown_capability"),
    (json.dumps({"capability": "monitor_labels", "args": {}}), "unknown_capability"),
])
def test_malformed_and_unknown_requests_are_refused_and_not_counted(setup, text, code):
    br, _, ledger = setup()
    with br.open_call("heading") as call:
        reply = json.loads(call.request(text))
    assert reply == {"ok": False, "refusal": code}
    assert br.predictions_used == 0 and ledger.calls == []


@pytest.mark.parametrize("override", [
    {"true_wind_nm_s": [0.0, 0.0]},                       # a private field is not in the schema
    {"start": {"x_nm": 0.0, "y_nm": 0.0, "heading_deg": 90.0, "tas_nm_s": 0.125, "truth": True}},
    {"headings_deg": []},
    {"headings_deg": [90.0] * (B.MAX_PREDICTION_SECONDS + 1)},
    {"headings_deg": [90.0, "95"]},
    {"headings_deg": [90.0, True]},
    {"wind_nm_s": [0.0]},
    {"orbit": {"cx_nm": 40.0, "cy_nm": 30.0, "radius_nm": 0.0, "speed_nm_s": 0.07, "angle_rad": 1.0}},
    {"goal_version": -1},
    {"goal_version": True},
])
def test_requests_outside_the_public_schema_are_refused(setup, override):
    br, _, _ = setup()
    with br.open_call("heading") as call:
        assert json.loads(call.request(request(**override))) == {"ok": False, "refusal": "malformed"}


def test_stale_goal_version_is_refused_with_a_public_code(setup):
    br, _, _ = setup()
    br.at(t=300, action_step=301, goal_version=2)
    with br.open_call("heading") as call:
        assert json.loads(call.request(request(goal_version=1))) == {"ok": False, "refusal": "stale_goal_version"}
        assert json.loads(call.request(request(goal_version=2)))["ok"]


def test_the_allowance_is_enforced_across_calls(setup):
    br, _, ledger = setup(allowance=3)
    for _ in range(2):
        with br.open_call("heading") as call:
            call.request(request())
            call.request(request()) if br.predictions_used < 3 else None
    with br.open_call("heading") as call:
        assert json.loads(call.request(request())) == {"ok": False, "refusal": "over_allowance"}
    assert br.predictions_used == 3 and len(ledger.calls) == 3


def test_per_call_request_limit(setup, monkeypatch):
    monkeypatch.setattr(B, "MAX_REQUESTS_PER_CALL", 2)
    br, _, _ = setup()
    with br.open_call("heading") as call:
        call.request("bad")
        call.request("bad")
        assert json.loads(call.request(request())) == {"ok": False, "refusal": "request_limit"}
    assert br.predictions_used == 0


def test_a_failing_tool_is_charged_and_reveals_nothing(setup):
    def broken(args):
        raise RuntimeError("private detail that must not reach the procedure")
    br, _, ledger = setup(predict=broken)
    with br.open_call("heading") as call:
        reply = call.request(request())
    assert json.loads(reply) == {"ok": False, "refusal": "tool_failed"} and "private" not in reply
    assert br.predictions_used == 1 and ledger.calls[0].status == "failed"


def test_an_interrupted_call_is_recorded_and_closed(setup, tmp_path):
    br, _, _ = setup()
    with pytest.raises(KeyboardInterrupt):
        with br.open_call("heading") as call:
            call.request(request())
            raise KeyboardInterrupt("invented interruption")
    assert json.loads(call.request(request())) == {"ok": False, "refusal": "call_closed"}
    events = [json.loads(l)["payload"] for l in (tmp_path / "log.jsonl").read_text().splitlines()[1:]]
    close = [e for e in events if e["kind"] == "call_close"]
    assert close[0]["status"] == "interrupted" and "invented interruption" in close[0]["error"]
    assert br.predictions_used == 1                      # the answered prediction before the interruption counts


def test_live_and_replay_replies_are_byte_identical(setup):
    texts = [request(), "not json", request(goal_version=5), json.dumps({"capability": "x", "args": {}})]
    out = {}
    for replay in (False, True):
        br, _, ledger = setup(replay=replay, name=f"log{replay}")
        with br.open_call("heading") as call:
            out[replay] = [call.request(t) for t in texts]
        assert ledger.calls[0].scope == ("replay" if replay else "running")
    assert out[False] == out[True]
    assert all("replay" not in r for r in out[True])


def test_monitor_feedback_is_restricted_to_its_schema(setup):
    br, _, _ = setup()
    with pytest.raises(B.BrokerError):
        br.feedback("episode_end", goal_met=True, failed_part=None)          # not before the episode ends
    br.end_episode()
    br.feedback("episode_end", goal_met=False, failed_part="report")
    sha = "a" * 64
    br.feedback("replay_test", entry=sha, passed=True)
    br.feedback("undo", entry=sha, reason="trial on completed episodes failed")
    for bad in ({"kind": "episode_end", "goal_met": False, "failed_part": "report", "true_range_nm": 0.3},
                {"kind": "episode_end", "goal_met": "yes", "failed_part": None},
                {"kind": "label", "entry": sha, "label": "wrong"},
                {"kind": "replay_test", "entry": "short", "passed": True},
                {"kind": "undo", "entry": sha, "reason": "x" * 201},
                {"kind": "undo", "entry": sha, "reason": "label 4 was wrong"}):     # free text is refused (TURN 56)
        kind = bad.pop("kind")
        with pytest.raises(B.BrokerError):
            br.feedback(kind, **bad)
    assert br.take_feedback() == [{"kind": "episode_end", "goal_met": False, "failed_part": "report"},
                                  {"kind": "replay_test", "entry": sha, "passed": True},
                                  {"kind": "undo", "entry": sha, "reason": "trial on completed episodes failed"}]
    assert br.take_feedback() == []


def test_event_prefix_fingerprint_identifies_the_sequence_and_supports_ending_cuts(setup):
    br, _, ledger = setup()
    prefixes = [br.event_prefix]
    for step in (5, 9, 14):
        br.at(t=step - 1, action_step=step, goal_version=0)
        with br.open_call("heading") as call:
            call.request(request())
        prefixes.append(br.event_prefix)
    assert len(set(prefixes)) == 4
    cut = ledger.ending_calls("invented-episode", 10)
    assert len(cut["counted"]) == 2 and len(cut["excluded"]) == 1
    br2, _, _ = setup(name="again")
    for step in (5, 9, 14):
        br2.at(t=step - 1, action_step=step, goal_version=0)
        with br2.open_call("heading") as call:
            call.request(request())
    assert br2.event_prefix == br.event_prefix         # same events, same fingerprint


def test_time_and_action_step_never_go_back(setup):
    br, _, _ = setup()
    br.at(t=10, action_step=11, goal_version=0)
    for bad in ((9, 11, 0), (10, 10, 0), (10, 11, -1), (10.5, 11, 0)):
        with pytest.raises(B.BrokerError):
            br.at(*bad)
    br.end_episode()
    with pytest.raises(B.BrokerError):
        br.open_call("heading")


def test_procedure_side_public_surface_is_request_only(setup):
    """An interface check, not containment: inside one process the private attributes are still reachable. Model
    code never receives this object; the sandbox will carry only request and reply text (BROKER-CONTRACT.md)."""
    br, _, _ = setup()
    call = br.open_call("heading")
    public = {n for n in dir(call) if not n.startswith("_")}
    assert public == {"request", "call_id", "purpose", "requests", "closed"}


def test_nested_duplicate_keys_and_wrong_length_replies(setup):
    """BR51.7 at depth, and BR51.5 for a reply whose length does not match the request."""
    br, _, ledger = setup()
    nested = request().replace('"x_nm": 0.0', '"x_nm": 0.0, "x_nm": 5.0')
    with br.open_call("heading") as call:
        assert json.loads(call.request(nested)) == {"ok": False, "refusal": "malformed"}
    short = setup(predict=lambda a: {"positions_nm": [[0.0, 0.0]], "closest_approach_nm": 1.0, "closest_at_s": 1.0},
                  name="short")
    br2, _, ledger2 = short
    with br2.open_call("heading") as call:
        assert json.loads(call.request(request())) == {"ok": False, "refusal": "tool_failed"}
    assert ledger2.calls[0].status == "failed" and br.predictions_used == 0


@pytest.mark.parametrize("where", ["start clock", "end clock"])
def test_clock_interruption_records_one_row_without_wall_time(tmp_path, where):
    """BR53.RET at both clock reads: exactly one interrupted row, wall time None, the original interruption raised."""
    ticks = []

    def clock():
        ticks.append(1)
        if (where == "start clock" and len(ticks) == 1) or (where == "end clock" and len(ticks) == 2):
            raise KeyboardInterrupt(where)
        return float(len(ticks))
    log = TrustedLog(tmp_path / "l.jsonl", tmp_path / "l.heads", "invented-run")
    ledger = Ledger()
    br = B.Broker("invented-episode", log, ledger, predict_stub, clock=clock)
    with pytest.raises(KeyboardInterrupt, match=where):
        with br.open_call("heading") as call:
            call.request(request())
    (c,) = ledger.calls
    assert c.status == "interrupted" and c.wall_s is None and br.predictions_used == 1
    ev = [json.loads(l)["payload"] for l in (tmp_path / "l.jsonl").read_text().splitlines()[1:]]
    (stop,) = [e for e in ev if e["kind"] == "prediction_interrupted"]
    assert stop["phase"] == where and stop["request_id"] == c.request_id
    assert not any(e["kind"] == "prediction_end" for e in ev)
