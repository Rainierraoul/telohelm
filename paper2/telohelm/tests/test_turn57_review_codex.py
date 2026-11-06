"""Independent TURN57 integration controls. Invented inputs only; no planned collection."""
from __future__ import annotations
import json
import time
import pytest
import telohelm.runtime.broker as B; import telohelm.runtime.sandbox as S; import telohelm.runtime.episode_loop as L
from telohelm.runtime.ledger import Call; from telohelm.runtime.ledger import Ledger
from telohelm.runtime.library import Library
from telohelm.runtime.trusted_log import TrustedLog; from telohelm.runtime.trusted_log import verify
from .test_broker_channel import ARGS; from .test_broker_channel import tool
from .test_episode_loop import scenario

@pytest.fixture
def make(tmp_path):
    counter = [0]
    def create(replay=False, predict=tool, cpu_clock=None):
        counter[0] += 1
        root = tmp_path / str(counter[0]); root.mkdir()
        log = TrustedLog(root / "blog", root / "heads", "invented-turn57")
        ledger = Ledger()
        options = {} if cpu_clock is None else {"cpu_clock": cpu_clock}
        br = B.Broker("loop-test", log, ledger, predict, replay=replay,
                      condition="invented-condition", stream="invented-stream", **options)
        return br, ledger, root
    return create

def events(root):
    return [json.loads(x)["payload"] for x in (root / "blog").read_text().splitlines()[1:]]

def channel(br, code, wall=2):
    return S.run_with_broker(code, "decide", {}, br, "heading:decide", wall_seconds=wall)

needs_sandbox = pytest.mark.skipif(not S.available(), reason="no tested sandbox route")

@needs_sandbox
# Fixed in TURN 58 (Claude): BR57.FINAL: typed final envelope is not validated
@pytest.mark.parametrize("body", [b"null", b"[]", b'{"ok":"yes","result":1}',
                                  b'{"ok":true}', b'{"ok":true,"result":1,"extra":"x"}',
                                  b'{"ok":true,"result":NaN}', b'{"ok":false,"error":17}'])
def test_malformed_final_envelopes_fail_closed(make, body):
    br, ledger, root = make()
    frame = b"F" + ("%010d" % len(body)).encode() + body
    code = f"import os,sys\ndef decide(ctx):\n    sys.stdout.buffer.write({frame!r}); sys.stdout.flush(); os._exit(0)\n"
    r = channel(br, code)
    assert r.status == "bad_reply"
    assert ledger.calls == []
    assert events(root)[-1]["status"] == "bad_reply"

@needs_sandbox
# Fixed in TURN 58 (Claude): BR57.DRAIN: delayed trailing bytes are not drained after F
def test_delayed_trailing_bytes_fail_closed(make, monkeypatch):
    br, _, root = make()
    monkeypatch.setattr(S, "RUNNER_BROKER", S.RUNNER_BROKER +
                        "\nimport time\ntime.sleep(0.15)\n_out.write(b'X'); _out.flush()\n")
    r = channel(br, "def decide(ctx):\n    return 7\n")
    assert r.status == "bad_reply"
    assert events(root)[-1]["status"] == "bad_reply"

@needs_sandbox
# Fixed in TURN 58 (Claude): BR57.EOF: wait after no-final EOF raises TimeoutExpired
def test_stdout_closed_without_final_times_out_as_result(make):
    br, _, root = make()
    start = time.perf_counter()
    r = channel(br, "import os,time\ndef decide(ctx):\n    os.close(1); time.sleep(2)\n", wall=0.2)
    assert r.status == "timeout" and time.perf_counter() - start < 1.5
    assert events(root)[-1]["status"] == "timeout"

# Fixed in TURN 58 (Claude): BR57.PREFIX: ending prefix commits no world actions/decisions
def test_world_action_changes_change_ending_prefix(make, monkeypatch):
    results = []
    for heading in (0.0, 90.0):
        br, ledger, root = make()
        def choose(decision, ctx, library, **kw):
            return {"use": "invented", "output": {"heading_deg": heading} if decision == "heading" else {},
                    "tried": [], "checks": []}
        monkeypatch.setattr(L, "choose", choose)
        result = L.run_episode(scenario(), Library(root / "lib"), L.NoModels(),
                               TrustedLog(root / "episode", root / "eheads", "invented-turn57"),
                               ledger, max_steps=3, broker=br)
        results.append(result)
        assert verify(root / "episode", root / "eheads")["ok"]
    assert results[0]["actions"] != results[1]["actions"]
    assert results[0]["ending_cuts"][0]["event_prefix_sha256"] != results[1]["ending_cuts"][0]["event_prefix_sha256"]

# Fixed in TURN 58 (Claude): BR57.LEDGER: episode silently accepts a different ledger
def test_episode_refuses_a_different_broker_ledger(make):
    br, _, root = make()
    with pytest.raises(ValueError):
        L.run_episode(scenario(), Library(root / "lib"), L.NoModels(),
                      TrustedLog(root / "episode", root / "eheads", "invented-turn57"),
                      Ledger(), max_steps=0, broker=br)

@pytest.mark.parametrize("replay", [False, True])
def test_cpu_wall_identity_and_scope_are_separate(make, replay):
    ticks = iter([10.0, 10.25])
    br, ledger, root = make(replay=replay, cpu_clock=lambda: next(ticks))
    br.at(3, 2, 0)
    with br.open_call("invented") as c:
        reply = json.loads(c.request(json.dumps({"capability":"predict", "args":json.loads(ARGS)})))
    assert reply["ok"]
    (row,) = ledger.calls
    assert (row.condition, row.stream, row.sim_second, row.action_step, row.answered_step) == (
        "invented-condition", "invented-stream", 3, 2, 2)
    assert row.cpu_s == 0.25 and row.scope == ("replay" if replay else "running")
    linked = [e for e in events(root) if e["kind"] in ("prediction_start", "prediction_end")]
    assert len(linked) == 2 and all(e["request_id"] == row.request_id for e in linked)
    assert verify(root / "blog", root / "heads")["ok"]

@pytest.mark.parametrize("failed_read", [1,2])
def test_unavailable_cpu_clock_preserves_answer_and_null(make, failed_read):
    count = [0]
    def cpu():
        count[0] += 1
        if count[0] == failed_read: raise RuntimeError("invented unavailable CPU clock")
        return 2.0
    br, ledger, _ = make(cpu_clock=cpu)
    with br.open_call("invented") as c:
        assert json.loads(c.request(json.dumps({"capability":"predict", "args":json.loads(ARGS)})))["ok"]
    assert len(ledger.calls) == 1 and ledger.calls[0].cpu_s is None and ledger.calls[0].status == "answered"

def test_ledger_cuts_include_straddling_and_unknown():
    ledger = Ledger()
    for name, start, answer in [("before",1,1),("straddle",1,3),("unknown",2,None),("at",3,3),("after",4,4)]:
        ledger.add(Call(name,"reasoning_model","invented",start,None,None,None,None,"answered",answered_step=answer))
    # TURN 71 (Claude): the cut also carries charge_ids (MG70.CUT.CHARGE); with no recovery rows they equal counted.
    assert ledger.ending_calls("invented",3) == {
        "counted":["before","straddle","unknown"],"charge_ids":["before","straddle","unknown"],
        "excluded":["at","after"],"straddling":["straddle"],"straddling_unknown":["unknown"]}

@pytest.mark.parametrize("reason", B.UNDO_REASONS)
def test_monitor_reason_fixed_list_is_accepted(make, reason):
    br, _, _ = make()
    br.feedback("undo",entry="a"*64,reason=reason)
    assert br.take_feedback() == [{"kind":"undo","entry":"a"*64,"reason":reason}]
