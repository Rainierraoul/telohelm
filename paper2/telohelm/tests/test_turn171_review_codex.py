"""Independent feedback-channel and current-call controls. No real model calls."""
import json
from pathlib import Path
import pytest
import telohelm.learning as G; import telohelm.token_count as T
import telohelm.runtime.replay as V; import telohelm.runtime.sandbox as S; import telohelm.runtime.models as M
from paper2.experiment2.tests.test_learning_feedback import coordinator; from paper2.experiment2.tests.test_learning_feedback import prompt; from paper2.experiment2.tests.test_learning_feedback import PROC


def feedback_symbols(tmp_path):
    symbols = set()
    for wrong in (False, True):
        for right in (94, 96, 100):
            def run(source, function, args, **kwargs):
                item = args[0]
                accepted = source == "old" or (wrong if item["wrong"] else item["index"] < right)
                return S.SandboxResult("ok", "accept" if accepted else "reject")
            r = V.replay_test("new", [{"context": {"wrong": True}, "output": {}}],
                              [{"context": {"wrong": False, "index": i}, "output": {}}
                               for i in range(100)], old_source="old", run=run)
            co, _ = coordinator(tmp_path / f"{wrong}-{right}", [])
            co._remember_result({"attempt": 1, "entry_kind": "check", "decision": "heading",
                                 "passed": r["passed"], "detail": {"replay": {**r, **G._two_facts(r)}}}, PROC)
            h = prompt(co)["previous_attempts"][-1]
            symbols.add(json.dumps({k: h[k] for k in ("adopted", "refused_at", "why") if k in h}, sort_keys=True))
    return symbols


@pytest.mark.xfail(strict=True, reason="FEED171.BITS: hidden replay feedback exceeds declared two-bit alphabet")
def test_declared_two_bit_replay_channel_has_at_most_four_symbols(tmp_path):
    assert len(feedback_symbols(tmp_path)) <= 4


def offline_source_prompt(tmp_path):
    co, _ = coordinator(tmp_path / "co", [])
    source = "def decide(ctx):\n" + "    x=1\n" * 650 + '    return {"heading_deg": 90.0}\n'
    co._remember(1, "procedure", "heading", "trial", ["the procedure failed its trial"], source)
    messages = co.messages([])
    assert len(messages[1][1]) < G.FEEDBACK_MAX_CHARS
    return co, messages


@pytest.mark.skipif(not (T.MODEL_FILES / "tokenizer.json").exists(), reason="pinned offline tokenizer unavailable")
def test_optional_source_is_removed_before_avoidable_input_token_refusal(tmp_path):
    co, messages = offline_source_prompt(tmp_path)
    kwargs = {"enable_thinking": True, "reasoning_effort": "medium"}
    assert T.count(messages, kwargs) > T.INPUT_TOKEN_CAP
    body = json.loads(messages[1][1])
    body["previous_attempts"][-1].pop("source")
    body["previous_attempts"][-1]["source_omitted"] = True
    shorter = [messages[0], ("user", json.dumps(body, sort_keys=True))]
    assert T.count(shorter, kwargs) <= T.INPUT_TOKEN_CAP
    proposer = G.GatewayProposer("http://127.0.0.1:1", "invented", tmp_path / "gateway", "4401")
    proposer.provider = M.ScriptedProvider(lambda request: M.ModelResponse("answered", "{}", 1, 1))
    # Fitting belongs to the coordinator; the gateway must preserve the exact
    # supplied message and refuse it rather than silently rewriting its identity.
    co.propose = proposer
    fitted = co.messages([])
    assert json.loads(fitted[1][1])["previous_attempts"][-1]["source_omitted"] is True
    assert T.count(fitted, kwargs) <= T.INPUT_TOKEN_CAP
    proposer(fitted, {"episode": "invented-feedback", "attempt": 2, "call": 1})
    assert proposer.provider.calls == 1
    proposer(messages, {"episode": "invented-feedback", "attempt": 3, "call": 1})
    assert proposer.provider.calls == 1 and proposer.last_status == "preflight_refused"


def test_current_preflight_refusal_does_not_inherit_old_timeout(tmp_path):
    proposer = G.GatewayProposer("http://127.0.0.1:1", "invented", tmp_path / "gateway", "4401",
                                 count_tokens=lambda messages: 1)
    proposer.provider = M.ScriptedProvider(lambda request: M.ModelResponse("timed_out"))
    proposer([("user", "old")], {"episode": "invented-feedback", "attempt": 1, "call": 1})
    assert proposer.provider.calls == 1 and proposer.ledger.calls[-1].status == "timed_out"
    proposer.count_tokens = lambda messages: T.INPUT_TOKEN_CAP + 1
    co, _ = coordinator(tmp_path / "co", [])
    co.propose = proposer
    co.attempts = 1
    co.attempt(co.completed[-1])
    rows = [json.loads(line) for line in (tmp_path / "gateway" / "learning-attempt-2" / "preflight.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and all(r["sent"] is False for r in rows)
    assert proposer.provider.calls == 1
    assert "timed_out" not in prompt(co)["previous_attempts"][-1]["why"]
