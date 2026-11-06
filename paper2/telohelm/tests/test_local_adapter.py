"""The local chat-completions adapter stub (TURN 71) with invented transports. Nothing contacts a server."""
from __future__ import annotations

import json

import pytest

import telohelm.runtime.models as M
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.trusted_log import TrustedLog

ROLES = {"small_model": "invented-small", "reasoning_model": "invented-reasoning"}
TIMEOUTS = {"small_model": 30.0, "reasoning_model": 600.0}
REQ = M.ModelRequest('["s","e","c","small_model",1]', "attempt1", "e", "c", "s", "small_model",
                     (("system", "invented system"), ("user", "invented question")), 64, 1, 0, 0, 2.0)


def ok_body(text="invented answer", finish="stop"):
    return {"id": "cmpl-invented", "model": "invented-small", "usage": {"prompt_tokens": 11, "completion_tokens": 4},
            "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish}]}


def adapter(transport, **kw):
    return M.LocalChatProvider("http://invented.local:8000/", ROLES, transport, TIMEOUTS, **kw)


def test_construction_contacts_nothing_and_is_unreportable_by_default():
    calls = []
    a = adapter(lambda *x: calls.append(x))
    assert calls == [] and a.reportable is False and a.relays is False


def test_the_request_body_and_answer_mapping():
    seen = []

    def transport(url, body, timeout):
        seen.append((url, body, timeout))
        return 200, ok_body()
    r = adapter(transport).complete(REQ)
    url, body, timeout = seen[0]
    assert url == "http://invented.local:8000/v1/chat/completions" and timeout == 30.0
    assert body == {"model": "invented-small", "max_tokens": 64, "temperature": 0.0,
                    "messages": [{"role": "system", "content": "invented system"},
                                 {"role": "user", "content": "invented question"}]}
    assert (r.status, r.text, r.tokens_in, r.tokens_out) == ("answered", "invented answer", 11, 4)
    assert r.provider_ids == {"id": "cmpl-invented", "model": "invented-small"}


@pytest.mark.parametrize("transport, status", [
    (lambda *a: 200 and (200, ok_body(finish="length")), "cut_off"),
    (lambda *a: (_ for _ in ()).throw(TimeoutError()), "timed_out"),
    (lambda *a: (_ for _ in ()).throw(ConnectionRefusedError()), "failed"),
    (lambda *a: (500, {"error": "invented"}), "failed"),
    (lambda *a: (200, {"choices": []}), "failed"),
    (lambda *a: (200, {"choices": [{"message": {"content": 7}}]}), "failed"),
])
def test_failure_modes_map_to_statuses_and_pass_no_text(transport, status):
    r = adapter(transport).complete(REQ)
    assert r.status == status and r.text is None


def test_an_unknown_speaker_is_refused_before_the_transport():
    calls = []
    bad = M.ModelRequest(REQ.request_id, "attempt1", "e", "c", "s", "small_model", (("tool", "x"),), 64, 1, 0, 0, 2.0)
    r = adapter(lambda *a: calls.append(a)).complete(bad)
    assert r.status == "failed" and calls == []


def test_a_role_without_a_model_or_timeout_is_refused():
    with pytest.raises(M.GatewayError):
        M.LocalChatProvider("http://x", {"small_model": "a"}, lambda *a: None, TIMEOUTS)


def test_through_the_gateway_tokens_come_from_the_server_usage(tmp_path):
    ledger = Ledger()
    gw = M.ModelGateway("e", "c", "s", adapter(lambda *a: (200, ok_body())),
                        TrustedLog(tmp_path / "l", tmp_path / "h", "r"), ledger, tmp_path / "j")
    out = gw.submit("small_model", [("user", "invented")], 64, 1, 0, 0, 2.0)
    assert out["status"] == "answered" and out["reportable"] is False
    assert (ledger.calls[0].tokens_in, ledger.calls[0].tokens_out) == (11, 4)
    assert json.loads(json.dumps(out))["text"] == "invented answer"


def test_a_reviewed_served_alias_is_accepted_and_others_fail_closed():
    body = ok_body()
    body["model"] = "invented-small-alias"
    served = {"small_model": ("invented-small", "invented-small-alias"), "reasoning_model": ("invented-reasoning",)}
    a = M.LocalChatProvider("http://invented.local", ROLES, lambda *x: (200, body), TIMEOUTS, served_names_by_role=served)
    assert a.complete(REQ).status == "answered"
    assert adapter(lambda *x: (200, body)).complete(REQ).status == "failed"     # alias not in the default list


def test_unusable_finishes_keep_reported_usage_but_no_text():
    r = adapter(lambda *x: (200, ok_body(finish="content_filter"))).complete(REQ)
    assert (r.status, r.text, r.tokens_in, r.tokens_out) == ("failed", None, 11, 4)


@pytest.mark.parametrize("kw", [{"temperature": float("nan")}, {"temperature": -1.0},
                                {"served_names_by_role": {"small_model": (), "reasoning_model": ("x",)}},
                                {"served_names_by_role": {"small_model": ("",), "reasoning_model": ("x",)}}])
def test_other_invalid_settings_are_refused_at_construction(kw):
    with pytest.raises(M.GatewayError):
        M.LocalChatProvider("http://invented.local", ROLES, lambda *a: None, TIMEOUTS, **kw)
