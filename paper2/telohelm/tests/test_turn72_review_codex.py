"""TURN72 exact stub review. Invented transports only; no network/model calls."""
from __future__ import annotations

import hashlib
import json

import pytest

import telohelm.runtime.models as M
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.trusted_log import verify
from .test_local_adapter import REQ; from .test_local_adapter import ROLES; from .test_local_adapter import TIMEOUTS; from .test_local_adapter import adapter; from .test_local_adapter import ok_body
from .test_turn65_review_codex import gateway; from .test_turn65_review_codex import submit
from .test_turn70_review_codex import execute; from .test_turn70_review_codex import flight; from .test_turn70_review_codex import records


# Fixed in TURN 73 (Claude): DL72.JSON: non-finite parsed action crashes trusted logging before world refusal
@pytest.mark.parametrize("key", ["heading_deg", "report_trigger_nm"])
def test_nonfinite_parsed_action_is_refused_before_checkpoint_serialisation(tmp_path, monkeypatch, key):
    # 1e999 is a JSON number; the supplied standard parser overflows it to infinity.
    parts = flight(tmp_path, monkeypatch, text='{"' + key + '":1e999}')
    result = execute(parts, tmp_path)
    assert len(result["actions"]) == 3 and key not in result["actions"][1]
    reply = next(e for e in records(tmp_path / "log") if e["kind"] == "model_reply")
    assert reply["part"] is None and reply["refusal"] is not None
    assert verify(tmp_path / "log", tmp_path / "heads")["ok"]
    assert verify(tmp_path / "blog", tmp_path / "bheads")["ok"]


# Fixed in TURN 73 (Claude): LA72.CONFIG: invalid timeout/model/reportability values accepted by constructor
@pytest.mark.parametrize("kind,value", [("timeout", 0.), ("timeout", float("nan")),
                                        ("timeout", float("inf")), ("model", ""),
                                        ("reportable", "False")])
def test_adapter_config_is_refused_before_transport_or_reportability(kind, value):
    names, limits, calls = dict(ROLES), dict(TIMEOUTS), []
    kwargs = {}
    if kind == "timeout":
        limits["small_model"] = value
    elif kind == "model":
        names["small_model"] = value
    else:
        kwargs[kind] = value
    with pytest.raises(M.GatewayError):
        M.LocalChatProvider("http://invented.local", names, lambda *a: calls.append(a), limits, **kwargs)
    assert calls == []


# Fixed in TURN 73 (Claude): LA72.MODEL: missing/unexpected response model accepted as answered
@pytest.mark.parametrize("model", ["invented-other-model", None])
def test_missing_or_wrong_returned_identity_cannot_be_an_answer(model):
    body = ok_body(); body["model"] = model
    # Adapted in TURN 83 (Claude): a reportable adapter now needs an exchange store and a checked configuration
    # fingerprint, so this identity check runs on an unreportable adapter; test_exchange_join.py repeats it with a store.
    response = adapter(lambda *a: (200, body)).complete(REQ)
    assert response.status != "answered" and response.text is None


# Fixed in TURN 73 (Claude): LA72.FINISH: unsupported or incomplete finish treated as usable answer
@pytest.mark.parametrize("finish", [None, "tool_calls", "invented-unknown-finish"])
def test_text_only_adapter_does_not_accept_unsupported_completion_outcome(finish):
    response = adapter(lambda *a: (200, ok_body(finish=finish))).complete(REQ)
    assert response.status != "answered" and response.text is None


# Fixed in TURN 73 (Claude): LA72.CUTOFF: known length outcome with null content loses status and usage
def test_length_with_no_content_keeps_cutoff_and_reported_tokens():
    response = adapter(lambda *a: (200, ok_body(text=None, finish="length"))).complete(REQ)
    assert response.status == "cut_off" and response.text is None
    assert (response.tokens_in, response.tokens_out) == (11, 4)


def test_all_cut_boundaries_keep_linked_charge_once_and_unknown_attempt(gateway, tmp_path):
    ledger, journal, ticks = Ledger(), tmp_path / "shared", [0]
    stop = KeyboardInterrupt("invented receipt stop")

    def clock():
        ticks[0] += 1
        if ticks[0] == 2:
            raise stop
        return 4.
    first, _, _, _ = gateway(journal=journal, ledger=ledger, clock=clock)
    with pytest.raises(KeyboardInterrupt) as caught:
        submit(first)
    assert caught.value is stop
    for _ in range(3):
        resumed, _, provider, _ = gateway(journal=journal, ledger=ledger)
        assert submit(resumed)["recovered"] and provider.calls == 0
    for cut_at, expected in [(9, 0), (10, 0), (11, 7), (13, 7), (14, 7)]:
        cut = ledger.ending_calls("episode-a", cut_at)
        assert len(cut["counted"]) == (1 if expected else 0)
        assert len(set(cut["charge_ids"])) == len(cut["charge_ids"])
        ids = set(cut["charge_ids"])
        assert sum(c.tokens_in or 0 for c in ledger.calls if c.request_id in ids) == expected
    assert ledger.calls[0].status == "interrupted" and ledger.calls[0].wall_s is None
    assert ledger.gpu_shares(1., [("invented", 0., 3600., 1)])["busy_h"] is None


@pytest.mark.parametrize("part,refusal", [(None, None), ([], "invalid_part"), ({}, "invalid_part"),
                                        ({"secret": "invented"}, "invalid_part")])
def test_parser_shapes_and_raw_reply_digest_are_public_without_error_text(tmp_path, monkeypatch, part, refusal):
    text = "invented raw reply"
    parts = flight(tmp_path, monkeypatch, text=text, parser=lambda d, t: part)
    result = execute(parts, tmp_path)
    assert "heading_deg" not in result["actions"][1]
    reply = next(e for e in records(tmp_path / "log") if e["kind"] == "model_reply")
    assert reply["refusal"] == refusal and reply["part"] is None
    assert reply["reply_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert reply["goal_version"] == 1 and reply["issued_second"] == 0
    assert verify(tmp_path / "blog", tmp_path / "bheads")["ok"]


def test_unrelated_parser_fault_preserves_exception_and_cached_response(tmp_path, monkeypatch):
    original = RuntimeError("invented unrelated implementation fault")

    def broken(decision, text):
        raise original
    parts = flight(tmp_path, monkeypatch, parser=broken)
    with pytest.raises(RuntimeError) as caught:
        execute(parts, tmp_path)
    assert caught.value is original
    journal = [json.loads(line) for line in (tmp_path / "journal").read_text().splitlines()]
    assert any(e.get("type") == "response" and e["response"]["text"] for e in journal)
    assert parts[2].calls == 1 and len(parts[1].calls) == 1
    assert verify(tmp_path / "glog", tmp_path / "gheads")["ok"]


@pytest.mark.parametrize("role", ["small_model", "reasoning_model"])
def test_role_specific_mapping_and_unknown_usage_are_retained(role):
    req = M.ModelRequest(REQ.request_id, REQ.attempt_id, REQ.episode, REQ.condition, REQ.stream, role,
                         REQ.messages, 64, 1, 0, 0, 2.)
    seen = []
    body = ok_body(); body["model"] = ROLES[role]; body["usage"] = {"prompt_tokens": True, "completion_tokens": -1}

    def transport(url, payload, timeout):
        seen.append((payload, timeout))
        return 200, body
    response = adapter(transport).complete(req)
    assert response.status == "answered" and response.tokens_in is None and response.tokens_out is None
    assert seen[0][0]["model"] == ROLES[role] and seen[0][1] == TIMEOUTS[role]
