"""The sandbox-to-broker channel (TURN 56): framing, bounds, one deadline over the whole exchange, interruption,
closing and live/replay parity. Invented procedures and a stub tool; CPU only; macOS sandbox route."""
from __future__ import annotations

import json
import time

import pytest

import telohelm.runtime.broker as B
import telohelm.runtime.sandbox as S
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.trusted_log import TrustedLog

pytestmark = pytest.mark.skipif(not S.available(), reason="no tested sandbox route on this platform")

ARGS = ('{"goal_version": 0, "start": {"x_nm": 0.0, "y_nm": 0.0, "heading_deg": 90.0, "tas_nm_s": 0.1}, '
        '"headings_deg": [90.0, 91.0], "wind_nm_s": [0.0, 0.0], '
        '"orbit": {"cx_nm": 1.0, "cy_nm": 1.0, "radius_nm": 1.0, "speed_nm_s": 0.1, "angle_rad": 0.0}}')


def tool(args):
    return {"positions_nm": [[0.0, 0.0]] * len(args["headings_deg"]), "closest_approach_nm": 1.0,
            "closest_at_s": 1.0}


@pytest.fixture
def make(tmp_path):
    n = [0]

    def factory(predict=tool, replay=False):
        n[0] += 1
        log = TrustedLog(tmp_path / f"l{n[0]}", tmp_path / f"h{n[0]}", "invented-run")
        ledger = Ledger()
        br = B.Broker("invented-episode", log, ledger, predict, replay=replay, condition="telohelm",
                      stream="invented-stream")
        return br, ledger, tmp_path / f"l{n[0]}"
    return factory


def events(path):
    return [json.loads(l)["payload"] for l in path.read_text().splitlines()[1:]]


def run(br, code, wall=5.0):
    return S.run_with_broker(code, "decide", [{"t": 3}], br, "heading:decide", wall_seconds=wall, context_args=True)


def test_predict_over_the_channel_is_answered_charged_and_closed(make):
    br, ledger, path = make()
    code = f"import json\ndef decide(ctx):\n    return predict(json.loads({ARGS!r}))\n"
    r = run(br, code)
    assert r.status == "ok" and r.result["ok"] and r.broker_requests == 1
    (c,) = ledger.calls
    assert (c.condition, c.stream, c.sim_second, c.answered_step) == ("telohelm", "invented-stream", 0, 0)
    assert c.cpu_s is not None and r.cpu_seconds is not None      # tool CPU (parent) and child CPU, kept apart
    ev = events(path)
    assert [e["kind"] for e in ev] == ["broker_open", "call_open", "prediction_start", "prediction_end", "reply",
                                       "call_close"]
    assert ev[-1]["status"] == "completed" and ev[1]["purpose"] == "heading:decide"


def test_replies_seen_by_the_procedure_are_identical_live_and_replay(make):
    code = (f"import json\ndef decide(ctx):\n    a = json.loads({ARGS!r})\n"
            "    return [predict(a), predict({'x': 1}), predict(dict(a, goal_version=3))]\n")
    results = []
    for replay in (False, True):
        br, ledger, _ = make(replay=replay)
        r = run(br, code)
        assert r.status == "ok"
        results.append(json.dumps(r.result, sort_keys=True))
        assert ledger.calls[0].scope == ("replay" if replay else "running")
    assert results[0] == results[1]


@pytest.mark.parametrize("body, error", [
    ("import sys\ndef decide(ctx):\n    sys.stdout.buffer.write(b'garbage!!!!!!'); sys.stdout.flush(); return 1\n",
     "protocol violation"),
    ("import sys\ndef decide(ctx):\n    sys.stdout.buffer.write(b'R9999999999'); sys.stdout.flush(); return 1\n",
     "frame too large"),
])
def test_malformed_frames_end_the_call(make, body, error):
    br, ledger, path = make()
    r = run(br, body)
    assert r.status == "bad_reply" and error in r.error
    assert ledger.calls == [] and br.predictions_used == 0
    assert events(path)[-1]["kind"] == "call_close" and events(path)[-1]["status"] == "bad_reply"


def test_bytes_after_the_final_frame_are_refused(make):
    br, _, _ = make()
    body = ("import sys, atexit\n"
            "atexit.register(lambda: (sys.stdout.buffer.write(b'X'), sys.stdout.flush()))\n"
            "def decide(ctx):\n    return 1\n")
    assert run(br, body).status == "bad_reply"


def test_a_child_that_never_reads_its_reply_hits_the_deadline(make):
    br, _, path = make()
    # Sends a request, then spins without reading the answer; the whole exchange stops at the deadline.
    body = (f"import json, sys\ndef decide(ctx):\n    b = json.dumps({{'capability': 'predict', 'args': json.loads({ARGS!r})}}).encode()\n"
            "    sys.stdout.buffer.write(b'R' + b'%010d' % len(b) + b); sys.stdout.flush()\n"
            "    while True: pass\n")
    t0 = time.perf_counter()
    r = run(br, body, wall=1.0)
    assert r.status == "timeout" and time.perf_counter() - t0 < 3.0
    assert br.predictions_used == 1 and events(path)[-1]["status"] == "timeout"


def test_a_child_that_exits_without_a_final_frame(make):
    br, _, _ = make()
    r = run(br, "import os\ndef decide(ctx):\n    os._exit(0)\n")
    assert r.status == "crashed" and "no final frame" in r.error


def test_a_slow_trusted_tool_counts_against_the_deadline(make):
    def slow(args):
        time.sleep(1.2)
        return tool(args)
    br, ledger, _ = make(predict=slow)
    code = f"import json\ndef decide(ctx):\n    return predict(json.loads({ARGS!r}))\n"
    r = run(br, code, wall=1.0)
    assert r.status == "timeout" and "broker request" in r.error
    assert ledger.calls[0].status == "answered" and br.predictions_used == 1


def test_an_interruption_in_the_parent_kills_the_child_and_closes_the_call(make):
    def stop(args):
        raise KeyboardInterrupt("invented interruption")
    br, ledger, path = make(predict=stop)
    code = f"import json\ndef decide(ctx):\n    return predict(json.loads({ARGS!r}))\n"
    with pytest.raises(KeyboardInterrupt):
        run(br, code)
    assert ledger.calls[0].status == "interrupted"
    assert events(path)[-1]["kind"] == "call_close" and events(path)[-1]["status"] == "interrupted"


def test_an_over_size_reply_is_refused_not_truncated(make, monkeypatch):
    monkeypatch.setattr(B, "MAX_REPLY_BYTES", 120)
    br, ledger, _ = make()
    code = f"import json\ndef decide(ctx):\n    return predict(json.loads({ARGS!r}))\n"
    r = run(br, code)
    assert r.status == "ok" and r.result == {"ok": False, "refusal": "reply_too_large"}
    assert ledger.calls[0].status == "failed"


def test_the_request_limit_applies_over_the_channel(make, monkeypatch):
    monkeypatch.setattr(B, "MAX_REQUESTS_PER_CALL", 2)
    br, _, _ = make()
    code = "def decide(ctx):\n    return [predict({'x': i}) for i in range(3)]\n"
    r = run(br, code)
    assert [x["refusal"] for x in r.result] == ["malformed", "malformed", "request_limit"]


def test_a_blocked_reply_write_still_ends_at_the_deadline(make):
    """The reply is larger than the pipe buffer and the child never reads it, so the parent's write blocks; the
    non-blocking loop must still stop at the deadline."""
    def big(args):
        n = len(args["headings_deg"])
        return {"positions_nm": [[1.2345678901234567, 7.654321098765432]] * n, "closest_approach_nm": 1.0,
                "closest_at_s": 1.0}
    br, _, _ = make(predict=big)
    body = (f"import json, sys\ndef decide(ctx):\n    a = json.loads({ARGS!r}); a['headings_deg'] = [90.0] * 3600\n"
            "    b = json.dumps({'capability': 'predict', 'args': a}).encode()\n"
            "    sys.stdout.buffer.write(b'R' + b'%010d' % len(b) + b); sys.stdout.flush()\n"
            "    while True: pass\n")
    t0 = time.perf_counter()
    r = run(br, body, wall=1.0)
    assert r.status == "timeout" and time.perf_counter() - t0 < 3.0
