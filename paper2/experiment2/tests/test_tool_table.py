"""The broker's tool table (packs/tools.py, packs/tanker_tools.py; Codex's TURN235 review).

A task supplies its tools once from trusted setup; the broker keeps the size limits, the request limit, the current-
goal check, one allowance shared by every tool and charging on failure or interruption. The tanker's prediction tool
is the table's first entry and behaves exactly as the broker's single branch did.
"""
import json

import pytest

from telohelm.taskpacks.tanker.tools import TANKER_TOOLS; from telohelm.taskpacks.tanker.tools import predict_tool
from telohelm.contracts.tools import Tool
import telohelm.runtime.broker as B
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.trusted_log import TrustedLog


def predict_stub(args):
    n = len(args["headings_deg"])
    return {"positions_nm": [[args["start"]["x_nm"] + k, args["start"]["y_nm"]] for k in range(1, n + 1)],
            "closest_approach_nm": 1.5, "closest_at_s": float(n)}


def predict_request(goal_version=0):
    return json.dumps({"capability": "predict", "args": {
        "goal_version": goal_version, "start": {"x_nm": 0.0, "y_nm": 0.0, "heading_deg": 90.0, "tas_nm_s": 0.125},
        "headings_deg": [90.0, 95.0], "wind_nm_s": [0.0, 0.0],
        "orbit": {"cx_nm": 40.0, "cy_nm": 30.0, "radius_nm": 8.0, "speed_nm_s": 0.0778, "angle_rad": 1.0}}})


def echo_request(v, goal_version=0):
    return json.dumps({"capability": "echo", "args": {"goal_version": goal_version, "v": v}})


def broker(tmp_path, tools, allowance=B.PREDICTION_ALLOWANCE, name="log"):
    log = TrustedLog(tmp_path / f"{name}.jsonl", tmp_path / f"{name}.heads", "invented-run")
    ledger = Ledger()
    clock = iter(float(i) for i in range(10_000))
    return B.Broker("invented-episode", log, ledger, None, allowance=allowance, clock=lambda: next(clock),
                    tools=tools), ledger


def echo(reply=None, raises=None):
    def run(args):
        if raises is not None:
            raise raises
        return {"v": args["v"]} if reply is None else reply
    return Tool(run=run, request={"goal_version": "int", "v": "number"},
                reply_ok=lambda r, a: isinstance(r, dict) and set(r) == {"v"}, charge="tool")


def test_the_tanker_table_is_the_prediction_tool_and_needs_goal_versions():
    assert set(TANKER_TOOLS) == {"predict"} and TANKER_TOOLS["predict"].charge == "prediction_tool"
    with pytest.raises(ValueError):
        Tool(run=lambda a: a, request={"v": "number"}, reply_ok=lambda r, a: True)


def test_two_tools_share_one_allowance_and_keep_their_own_forms_and_charges(tmp_path):
    br, ledger = broker(tmp_path, {"predict": predict_tool(predict_stub), "echo": echo()}, allowance=2)
    with br.open_call("heading") as call:
        a = json.loads(call.request(predict_request()))
        b = json.loads(call.request(echo_request(4.5)))
        c = json.loads(call.request(echo_request(1.0)))
        d = json.loads(call.request(json.dumps({"capability": "steer", "args": {"goal_version": 0}})))
        e = json.loads(call.request(json.dumps({"capability": "echo", "args": {"goal_version": 0, "v": "x"}})))
    assert a["ok"] and b == {"ok": True, "result": {"v": 4.5}, "remaining": 0}
    assert c == {"ok": False, "refusal": "over_allowance"} and d["refusal"] == "unknown_capability"
    assert e["refusal"] == "malformed"
    assert [r.kind for r in ledger.calls] == ["prediction_tool", "tool"]
    assert br.predictions_used == 2


def test_a_second_tools_bad_reply_stale_goal_and_interruption_are_handled_like_the_first(tmp_path):
    br, ledger = broker(tmp_path, {"echo": echo(reply={"wrong": 1})}, name="bad")
    with br.open_call("heading") as call:
        assert json.loads(call.request(echo_request(2.0))) == {"ok": False, "refusal": "tool_failed"}
        assert json.loads(call.request(echo_request(2.0, goal_version=3)))["refusal"] == "stale_goal_version"
    assert [(r.kind, r.status) for r in ledger.calls] == [("tool", "failed")]
    br2, ledger2 = broker(tmp_path, {"echo": echo(raises=KeyboardInterrupt())}, name="interrupted")
    with pytest.raises(KeyboardInterrupt):
        with br2.open_call("heading") as call:
            call.request(echo_request(1.0))
    assert [(r.kind, r.status) for r in ledger2.calls] == [("tool", "interrupted")] and br2.predictions_used == 1


def test_nested_schemas_and_the_registry_are_snapshots(tmp_path):
    form = {"goal_version": "int", "nested": {"v": "number"}}
    tool = Tool(lambda a: {}, form, lambda r, a: True)
    form["nested"]["v"] = "short_text"
    assert tool.request["nested"]["v"] == "number"
    with pytest.raises(TypeError):
        tool.request["nested"]["v"] = "short_text"
    registry = {"echo": echo()}
    br, ledger = broker(tmp_path, registry)
    registry.clear()
    with br.open_call("heading") as call:
        assert json.loads(call.request(echo_request(1)))["ok"]
    assert len(ledger.calls) == 1


@pytest.mark.parametrize("goal_rule", ["number", "bool", lambda value: True, None])
def test_a_tool_cannot_weaken_goal_version_binding(goal_rule):
    with pytest.raises(ValueError):
        Tool(lambda a: {}, {"goal_version": goal_rule}, lambda r, a: True)


def test_bad_setup_is_rejected_before_the_broker_opens(tmp_path):
    for charge in ("reasoning_model", "echo_tool"):
        with pytest.raises(ValueError):
            Tool(lambda a: {}, {"goal_version": "int"}, lambda r, a: True, charge=charge)
    with pytest.raises(ValueError):
        Tool(lambda a: {}, {"goal_version": "int", "v": "unknown"}, lambda r, a: True)
    with pytest.raises(ValueError):
        broker(tmp_path, {"echo": object()})


def test_validation_exceptions_fail_closed_without_invocation_or_charge(tmp_path):
    called = []
    def broken(value):
        raise RuntimeError("validator failed")
    tool = Tool(lambda a: called.append(a), {"goal_version": "int", "v": broken}, lambda r, a: True)
    br, ledger = broker(tmp_path, {"echo": tool})
    with br.open_call("heading") as call:
        assert json.loads(call.request(echo_request(1))) == {"ok": False, "refusal": "malformed"}
    assert called == [] and ledger.calls == [] and br.predictions_used == 0


@pytest.mark.parametrize("goal_version", [0.0, True, -1])
def test_non_integer_goal_versions_do_not_run_a_tool(tmp_path, goal_version):
    br, ledger = broker(tmp_path, {"echo": echo()})
    with br.open_call("heading") as call:
        assert json.loads(call.request(echo_request(1, goal_version)))["refusal"] == "malformed"
    assert ledger.calls == []


def test_a_generic_tool_never_gets_model_gpu_hours(tmp_path):
    from telohelm.runtime.ledger import Call
    br, ledger = broker(tmp_path, {"echo": echo()})
    with br.open_call("heading") as call:
        assert json.loads(call.request(echo_request(1)))["ok"]
    ledger.add(Call("model", "small_model", "e", 0, 0., 1800., 10, 20, "answered",
                    allocation="job", gpus=(0,)))
    shares = ledger.gpu_shares(1., [("job", 0., 3600., 1)])
    assert shares["calls_unknown_time_or_placement"] == []
    assert shares["shares_h"] == {"model": 0.5} and shares["idle_h"] == 0.5


def test_generic_tool_costs_are_read_as_cpu_work_not_model_calls(tmp_path, monkeypatch):
    from dataclasses import asdict
    import paper2.experiment2.accounting as A
    import telohelm.runtime.attribution as attribution
    br, ledger = broker(tmp_path, {"echo": echo()})
    with br.open_call("heading") as call:
        assert json.loads(call.request(echo_request(1)))["ok"]
    (tmp_path / "summary.json").write_text(json.dumps({"episode": "invented-episode", "outcome": {}}))
    (tmp_path / "ledger.jsonl").write_text("".join(json.dumps(asdict(c)) + "\n" for c in ledger.calls))
    (tmp_path / "episode.log").write_text("")
    monkeypatch.setattr(attribution, "derive", lambda *a, **kw: {})
    monkeypatch.setattr(A, "from_telohelm", lambda *a: {})
    rows = A.read_telohelm(tmp_path)["rows"]
    assert len(rows) == 1 and rows[0].process == "tool"
    assert rows[0].measured == ("cpu_s", "wall_s") and rows[0].covered == ("wall_s",)
    assert A.resource_vector(rows)["models"] == {}


@pytest.mark.parametrize("result", [object(), {"v": float("nan")}, {"v": "x" * (B.MAX_REPLY_BYTES + 1)}])
def test_a_tools_permissive_validator_cannot_bypass_json_or_reply_limits(tmp_path, result):
    tool = Tool(lambda a: result, {"goal_version": "int", "v": "number"}, lambda r, a: True)
    br, ledger = broker(tmp_path, {"echo": tool})
    with br.open_call("heading") as call:
        reply = json.loads(call.request(echo_request(1)))
    assert reply["ok"] is False and reply["refusal"] in ("tool_failed", "reply_too_large")
    assert len(ledger.calls) == 1 and ledger.calls[0].status == "failed" and br.predictions_used == 1
