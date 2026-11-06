"""TURN82 independent exact-repair boundaries; invented signed traces/socket stand-ins only."""
import json

import pytest

import telohelm.runtime.attribution as A; import telohelm.runtime.transport as T; import telohelm.runtime.models as M
from .test_attribution import signed; from .test_attribution import _set; from .test_attribution import fly; from .test_attribution import derive
from .test_turn80_review_codex import SocketStandIn
from .test_local_adapter import REQ; from .test_local_adapter import ROLES; from .test_local_adapter import TIMEOUTS


# Fixed in TURN 83 (Claude): AT82.GOAL: equality accepts boolean/float original goal tags as integer version1
@pytest.mark.parametrize("version", [True, 1.0])
def test_original_goal_version_is_a_typed_integer_before_model_credit(tmp_path, version):
    with pytest.raises(A.AttributionError):
        A.derive(*signed(tmp_path, _set(("reply", "goal_version"), version)), "inv")


# Fixed in TURN 83 (Claude): TR82.PARSE: oversized numeric HTTP fields escape as ValueError instead of transport refusal
@pytest.mark.parametrize("field", ["length", "status"])
def test_oversized_http_numeric_fields_fail_through_the_adapter_without_unrelated_exception(monkeypatch, field):
    digits = b"9" * 5000
    raw = (b'HTTP/1.0 200 OK\r\nContent-Length: ' + digits + b'\r\n\r\n{}' if field == "length"
           else b'HTTP/1.0 ' + digits + b' OK\r\nContent-Length: 2\r\n\r\n{}')
    fake = SocketStandIn([raw, b""])
    monkeypatch.setattr(T.socket, "create_connection", lambda *a, **kw: fake)
    seen = []
    t = T.LoopbackTransport("http://127.0.0.1:9", 100000, record=seen.append)
    adapter = M.LocalChatProvider("http://127.0.0.1:9", ROLES, t, TIMEOUTS)
    result = adapter.complete(REQ)
    assert result.status == "failed" and fake.closed
    assert seen[0]["outcome"] == "refused" and seen[0]["reply_bytes"] == len(raw)


# Fixed in TURN 83 (Claude): TR82.IPV6: the supported IPv6 origin writes an unbracketed HTTP Host authority
def test_supported_ipv6_origin_writes_bracketed_host_authority(monkeypatch):
    fake = SocketStandIn([b'HTTP/1.0 200 OK\r\nContent-Length: 2\r\n\r\n{}', b""])
    monkeypatch.setattr(T.socket, "create_connection", lambda *a, **kw: fake)
    try:
        t = T.LoopbackTransport("http://[::1]:9", 1000)
    except ValueError:
        return                   # explicitly narrowing supported origins to IPv4 is an acceptable declared alternative
    assert t("http://[::1]:9/x", {}, 1) == (200, {})
    assert b"\r\nHost: [::1]:9\r\n" in fake.sent


def test_delayed_issued_context_may_differ_under_the_same_typed_goal(tmp_path, monkeypatch):
    _, root, ep = fly(tmp_path, monkeypatch, lambda t: None, latency=1.0, steps=2)
    out = derive(root, ep)
    assert [s["category"] for s in out["steps"]] == ["holding_pending", "small_model"]
    detail = out["steps"][1]["set"]
    assert detail["issued"]["issued_second"] == 0 and detail["t"] == 1
    assert detail["issued"]["context_sha256"] != detail["context_sha256"]
    assert detail["issued"]["goal_version"] == detail["goal_version"]


@pytest.mark.parametrize("version", [0, 1, 2])
def test_integer_goal_tags_remain_valid_when_current_and_original_agree(tmp_path, version):
    def change(rows):
        rows["heading"]["goal_version"] = rows["report"]["goal_version"] = version
        rows["reply"]["goal_version"] = version
        rows["action"]["action"]["goal_version"] = version
    result = A.derive(*signed(tmp_path, change), "inv")
    assert result["complete"] and result["steps"][0]["set"]["issued"]["goal_version"] == version


@pytest.mark.parametrize("body", [b'{"a":{"x":1,"x":2}}', b'{"a":[-1e999]}', b'{"a":NaN}'])
def test_nested_ambiguous_or_nonfinite_json_is_refused(body):
    assert T._parse_json(body) is None


def test_numeric_only_origins_and_private_query_refusal_precede_contact(monkeypatch):
    def no_contact(*args, **kw):
        pytest.fail("an invalid configuration attempted contact")
    monkeypatch.setattr(T.socket, "create_connection", no_contact)
    with pytest.raises(ValueError):
        T.LoopbackTransport("http://localhost:9", 1000)
    seen = []
    t = T.LoopbackTransport("http://127.0.0.1:9", 1000, record=seen.append)
    with pytest.raises(T.TransportRefused):
        t("http://127.0.0.1:9/x?invented=private", {}, 1)
    assert seen[0]["reply_bytes"] == 0 and not seen[0]["reply_complete"]
