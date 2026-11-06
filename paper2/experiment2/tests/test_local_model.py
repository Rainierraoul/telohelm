"""MVP step 2: the local model's prompt, parser and provider. An invented loopback server stands in for the model
server; no real model is called here."""
from __future__ import annotations

import json

import pytest

import telohelm.local_model as LM
import paper2.experiment2.run_stream as R
from paper2.telohelm.tests.test_local_adapter import ok_body
from paper2.telohelm.tests.test_transport import Invented; from paper2.telohelm.tests.test_transport import http


def test_the_prompt_sends_scalars_and_only_the_newest_reports():
    ctx = {"t": 50, "x": 1.0, "reports": [{"measured_at": i, "x": i, "y": i} for i in range(40)]}
    msgs = LM.messages_for("heading", ctx, "unfamiliar_but_solvable")
    assert msgs[0][0] == "system" and "heading_deg" in msgs[1][1]
    body = json.loads(msgs[1][1].split("Context:\n", 1)[1])
    assert "reports" not in body and [r["measured_at"] for r in body["recent_reports"]] == [35, 36, 37, 38, 39]


@pytest.mark.parametrize("text,part", [('{"heading_deg": 48.5}', {"heading_deg": 48.5}),
                                       ('Sure. {"heading_deg": 10}', {"heading_deg": 10}), ("{}", None)])
def test_the_parser_takes_the_first_json_object(text, part):
    assert LM.parse("heading", text) == part


@pytest.mark.parametrize("text", ["no json here", "[1, 2]", "{not json}"])
def test_a_reply_without_a_json_object_is_unparseable(text):
    with pytest.raises(ValueError):
        LM.parse("heading", text)


def test_a_short_stream_runs_through_a_loopback_model_server(tmp_path):
    body = ok_body(text='{"heading_deg": 45.0}')
    body["model"] = "invented-local"
    srv = Invented(lambda req: [(http(json.dumps(body).encode()), 0)])
    try:
        done = R.run_stream(1, 1, tmp_path / "out", max_steps=5,
                            local={"base_url": f"http://127.0.0.1:{srv.port}", "model": "invented-local"})
    finally:
        srv.close()
    s, e = done["summary"], done["episodes"][0]
    assert s["model"] == "invented-local" and s["stand_ins"] == []   # the prediction tool is real from MVP step 4
    assert e["model_calls"] == len(srv.requests) and e["seconds"] == 5
    if e["model_calls"]:
        assert e["tokens_in"] == 11 * e["model_calls"] and e["tokens_out"] == 4 * e["model_calls"]


def test_the_models_get_study_1s_settings():
    """Plan Appendix A, row 9 carries Study 1's models into Experiment 2, the reasoning model at medium effort. Until
    13 October these runs sent temperature 0 and no effort, so the reasoning model thought at its default 'xhigh'."""
    import json as _json
    from pathlib import Path
    import telohelm.local_model as LM
    study1 = _json.loads((next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'dcarr-runtime-0.1/study1/config.cluster.json').read_text())
    assert LM.SETTINGS["small_model"] == study1["model_settings"]["T1"]
    assert LM.SETTINGS["reasoning_model"] == study1["model_settings"]["T2"]
    small = LM.provider("http://127.0.0.1:8000", "m")
    big = LM.provider("http://127.0.0.1:8001", "m", role="reasoning_model")
    assert small.temperature == 0.7 and big.temperature == 1.0
    assert big._extra_body["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "medium"}
    assert LM.provider("http://127.0.0.1:8001", "m", role="reasoning_model",
                       thinking=False)._extra_body["chat_template_kwargs"] == {"enable_thinking": False}
