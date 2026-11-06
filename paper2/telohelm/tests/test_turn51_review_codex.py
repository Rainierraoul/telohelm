"""TURN51 independent invented-input broker probes. No trajectory or model execution.

Strict expected failures retain demonstrated integration blockers for Claude to fix;
they are not passed requirements. Tested against exact 2b4b73f released in 96454a7.
"""
import hashlib
import json
from pathlib import Path

import pytest
import telohelm.runtime.broker as B
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.trusted_log import TrustedLog; from telohelm.runtime.trusted_log import verify


def message(version=0):
    return json.dumps({"capability": "predict", "args": {
        "goal_version": version,
        "start": {"x_nm": 1, "y_nm": 2, "heading_deg": 45, "tas_nm_s": .1},
        "headings_deg": [45, 50], "wind_nm_s": [0, .01],
        "orbit": {"cx_nm": 10, "cy_nm": 20, "radius_nm": 4,
                  "speed_nm_s": .08, "angle_rad": .1}}})


def result(args):
    return {"positions_nm": [[1, 2], [2, 3]], "closest_approach_nm": .5, "closest_at_s": 2}


@pytest.fixture
def make(tmp_path):
    count = 0
    def factory(tool=result, replay=False, allowance=2):
        nonlocal count
        count += 1
        path = tmp_path / str(count)
        head = tmp_path / (str(count) + ".heads")
        ledger = Ledger()
        ticks = iter(range(10000))
        broker = B.Broker("invented", TrustedLog(path, head, "turn51"), ledger, tool,
                          replay=replay, allowance=allowance, clock=lambda: float(next(ticks)))
        return broker, ledger, path, head
    return factory


def events(path):
    return [json.loads(line)["payload"] for line in path.read_text().splitlines()[1:]]


def test_positive_public_reply_accounting_and_independent_prefix(make):
    b, ledger, path, head = make()
    b.at(6, 8, 0)
    with b.open_call("heading") as c:
        assert json.loads(c.request(message())) == {"ok": True, "result": result(None), "remaining": 1}
    assert len(ledger.calls) == 1
    assert (ledger.calls[0].action_step, ledger.calls[0].status, ledger.calls[0].wall_s) == (8, "answered", 1)
    prefix = "0" * 64
    for event in events(path):
        claimed = event.pop("event_prefix_sha256")
        body = json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        prefix = hashlib.sha256((prefix + body).encode()).hexdigest()
        assert claimed == prefix
    assert prefix == b.event_prefix and verify(path, head)["ok"]
    assert ledger.ending_calls("invented", 8)["excluded"] == [ledger.calls[0].request_id]
    assert ledger.ending_calls("invented", 9)["counted"] == [ledger.calls[0].request_id]


@pytest.mark.parametrize("field,value", [("headings_deg", [True]), ("wind_nm_s", [0, float("inf")]),
                                         ("private_state", {"x": 1})])
def test_positive_untrusted_fields_refused_before_tool(make, field, value):
    called = []
    b, ledger, path, head = make(lambda a: called.append(a))
    obj = json.loads(message()); obj["args"][field] = value
    with b.open_call("heading") as c:
        assert json.loads(c.request(json.dumps(obj))) == {"ok": False, "refusal": "malformed"}
    assert not called and not ledger.calls and b.predictions_used == 0
    assert verify(path, head)["ok"]


def test_positive_failure_cap_and_live_replay(make):
    def broken(args):
        raise RuntimeError("invented private tool detail")
    replies = []
    for replay in (False, True):
        b, ledger, path, head = make(broken, replay, allowance=1)
        with b.open_call("heading") as c:
            replies.append([c.request(message()), c.request(message())])
        assert len(ledger.calls) == 1 and ledger.calls[0].status == "failed"
        assert ledger.calls[0].scope == ("replay" if replay else "running")
        assert b.predictions_used == 1 and "private" not in "".join(replies[-1])
        assert verify(path, head)["ok"]
    assert replies[0] == replies[1]


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit])
# Fixed in TURN 52 (Claude): BR51.1 prediction interruption has no ledger row or start event
def test_prediction_interruption_retains_started_and_interrupted_record(make, exc):
    def stopped(args):
        raise exc("invented")
    b, ledger, path, head = make(stopped)
    with pytest.raises(exc):
        with b.open_call("heading") as c:
            c.request(message())
    assert c.closed and b.predictions_used == 1
    assert json.loads(c.request(message()))["refusal"] == "call_closed"
    assert len(ledger.calls) == 1 and ledger.calls[0].status == "interrupted"
    assert any(e["kind"] == "prediction_start" for e in events(path))


# Fixed in TURN 52 (Claude): BR51.2 finite-integer conversion overflows outside request parsing guard
def test_huge_json_integer_is_a_public_malformed_refusal(make):
    b, ledger, _, _ = make()
    obj = json.loads(message()); obj["args"]["start"]["x_nm"] = 10**400
    with b.open_call("heading") as c:
        assert json.loads(c.request(json.dumps(obj))) == {"ok": False, "refusal": "malformed"}
    assert b.predictions_used == 0 and ledger.calls == []


# Fixed in TURN 52 (Claude): BR51.3 goal version may regress despite never-go-back contract
def test_goal_version_cannot_regress(make):
    b, _, _, _ = make()
    b.at(1, 2, 3)
    with pytest.raises(B.BrokerError):
        b.at(2, 3, 2)
    assert b.goal_version == 3


# Fixed in TURN 52 (Claude): BR51.4 outstanding call can predict after episode end
def test_episode_end_closes_existing_call_before_prediction(make):
    b, ledger, _, _ = make()
    with b.open_call("heading") as c:
        b.end_episode()
        assert json.loads(c.request(message())) == {"ok": False, "refusal": "call_closed"}
    assert not ledger.calls and b.predictions_used == 0


@pytest.mark.parametrize("bad", [None, 7, {"private_state": "invented"}, {"positions_nm": [[True, 0]]}])
# Fixed in TURN 52 (Claude): BR51.5 any finite JSON tool reply is forwarded without public result schema
def test_bad_trusted_reply_is_refused_and_charged(make, bad):
    b, ledger, _, _ = make(lambda a: bad)
    with b.open_call("heading") as c:
        assert json.loads(c.request(message())) == {"ok": False, "refusal": "tool_failed"}
    assert len(ledger.calls) == 1 and ledger.calls[0].status == "failed"


# Fixed in TURN 52 (Claude): BR51.6 only request digest is retained; exact public input absent
def test_prediction_input_is_retained_for_replay(make):
    b, _, path, _ = make()
    text = message()
    with b.open_call("heading") as c:
        c.request(text)
    assert any(e.get("request") == json.loads(text) for e in events(path))


# Fixed in TURN 52 (Claude): BR51.7 duplicate JSON keys are silently resolved by last key
def test_duplicate_capability_key_is_malformed(make):
    b, ledger, _, _ = make()
    text = message().replace('"capability": "predict"', '"capability": "read_truth", "capability": "predict"')
    with b.open_call("heading") as c:
        assert json.loads(c.request(text)) == {"ok": False, "refusal": "malformed"}
    assert b.predictions_used == 0 and ledger.calls == []
