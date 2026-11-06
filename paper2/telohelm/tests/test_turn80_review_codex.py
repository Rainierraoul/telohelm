"""TURN80 independent boundary controls. Invented signed traces and socket stand-ins only."""
import hashlib
import json
import socket

import pytest

import telohelm.runtime.attribution as A; import telohelm.runtime.transport as T
from telohelm.runtime.trusted_log import TrustedLog; from telohelm.runtime.trusted_log import verify
from .test_attribution import signed; from .test_attribution import _set; from .test_attribution import fly; from .test_attribution import derive


# Fixed in TURN 81 (Claude): AT80.ISSUE: original issue/availability/goal/digest tags are copied without validating them
@pytest.mark.parametrize("key,value", [("goal_version", 2), ("issued_step", 5), ("available_second", 5),
                                      ("context_sha256", None), ("reply_sha256", None)])
def test_model_issuance_links_are_validated_before_credit(tmp_path, key, value):
    with pytest.raises(A.AttributionError):
        A.derive(*signed(tmp_path, _set(("reply", key), value)), "inv")


def rewrite(tmp_path, rows):
    w = TrustedLog(tmp_path / "rewritten", tmp_path / "heads", "invented")
    for r in rows:
        w.append(r)
    assert verify(tmp_path / "rewritten", tmp_path / "heads")["ok"]
    return tmp_path / "rewritten", tmp_path / "heads"


# Fixed in TURN 81 (Claude): AT80.CUT: an ending name/count is accepted despite missing actual ending metadata
def test_a_labelled_ending_needs_an_actual_reason_and_time(tmp_path):
    log, heads = signed(tmp_path)
    rows = [json.loads(l)["payload"] for l in log.read_text().splitlines()[1:]]
    rows[-1].update(end_reason=None, ended_at=None)
    with pytest.raises(A.AttributionError):
        A.derive(*rewrite(tmp_path, rows), "inv", ending="nominal")


# Fixed in TURN 81 (Claude): AT80.CUT: an unfinished second is silently returned as a known zero-step episode
def test_an_open_second_is_not_a_legitimate_empty_episode(tmp_path):
    log, heads = signed(tmp_path)
    rows = [json.loads(l)["payload"] for l in log.read_text().splitlines()[1:]][:1]
    assert rows[0]["kind"] == "bundle"
    with pytest.raises(A.AttributionError):
        A.derive(*rewrite(tmp_path, rows), "inv")


# Fixed in TURN 81 (Claude): TR80.JSON: exponent overflow and duplicate JSON names pass the strict-finite policy
@pytest.mark.parametrize("body", [b'{"outer": [1e999]}', b'{"model":"wrong","model":"accepted"}'])
def test_strict_finite_json_has_no_overflow_or_ambiguous_names(body):
    assert T._parse_json(body) is None


# Fixed in TURN 81 (Claude): TR80.HTTP: conflicting framing and unsupported transfer encoding are accepted
@pytest.mark.parametrize("reply", [
    b'HTTP/1.1 200 OK\r\nContent-Length: 500\r\nContent-Length: 2\r\n\r\n{}',
    b'HTTP/1.0 200 OK\r\nTransfer-Encoding: gzip\r\nContent-Length: 2\r\n\r\n{}',
])
def test_conflicting_or_unsupported_http_framing_is_refused(reply):
    with pytest.raises(T.TransportRefused):
        T._parse_http(reply)


class SocketStandIn:
    def __init__(self, pieces=(), connect=None):
        self.pieces = iter(pieces)
        self.connect_action = connect
        self.connected = []
        self.closed = False
        self.sent = b""
    def settimeout(self, t):
        self.timeout = t
    def connect(self, addr):
        self.connected.append(addr)
        if self.connect_action:
            self.connect_action(self)
    def sendall(self, data):
        self.sent = data
    def recv(self, n):
        v = next(self.pieces, b"")
        if isinstance(v, BaseException):
            raise v
        return v
    def close(self):
        self.closed = True


# Fixed in TURN 81 (Claude): TR80.CONNECT: localhost resolution is neither bounded nor rechecked for loopback before connect
def test_hostname_resolution_cannot_authorise_a_nonloopback_connect(monkeypatch):
    fake = SocketStandIn()
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("203.0.113.7", 9))])
    monkeypatch.setattr(socket, "socket", lambda *a, **kw: fake)
    try:
        t = T.LoopbackTransport("http://localhost:9", 1000)
        t("http://localhost:9/x", {}, 1.0)
    except (ValueError, OSError):
        pass                       # numeric-only constructor refusal is an acceptable alternative
    assert fake.connected == []    # entirely invented: no off-machine socket is opened


# Fixed in TURN 81 (Claude): TR80.CONNECT: create_connection reuses one timeout across resolution and every candidate
def test_connection_candidates_share_one_remaining_deadline(monkeypatch):
    clock = [0.0]
    made = []
    def resolve(*a, **kw):
        clock[0] += 0.6
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 9)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.2", 9))]
    def connect(s):
        clock[0] += s.timeout
        raise socket.timeout("invented connection timeout")
    def make(*a, **kw):
        s = SocketStandIn(connect=connect); made.append(s); return s
    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(socket, "socket", make)
    try:
        t = T.LoopbackTransport("http://localhost:9", 1000, clock=lambda: clock[0])
        t("http://localhost:9/x", {}, 1.0)
    except (ValueError, TimeoutError):
        pass
    assert clock[0] <= 1.0 and len(made) <= 1


# Fixed in TURN 81 (Claude): TR80.RECORD: partial reply evidence is discarded on deadline or size refusal
@pytest.mark.parametrize("fault", ["deadline", "cap"])
def test_failed_exchange_preserves_observed_reply_bytes_without_claiming_completion(monkeypatch, fault):
    partial = b'HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{' if fault == "deadline" else b"x" * 200
    fake = SocketStandIn([partial, socket.timeout("invented")])
    monkeypatch.setattr(socket, "create_connection", lambda *a, **kw: fake)
    seen = []
    t = T.LoopbackTransport("http://127.0.0.1:9", 100 if fault == "cap" else 1000, record=seen.append)
    with pytest.raises(OSError):
        t("http://127.0.0.1:9/x", {}, 1.0)
    assert fake.closed and len(seen) == 1 and seen[0]["outcome"] in ("deadline", "refused")
    assert seen[0].get("reply_bytes") == len(partial)
    assert seen[0].get("reply_sha256") == hashlib.sha256(partial).hexdigest()
    assert seen[0].get("reply_complete") is False
    # A separate explicitly partial digest/byte namespace is an acceptable marked schema adaptation.


def test_valid_original_tags_and_actual_ending_are_retained(tmp_path):
    out = A.derive(*signed(tmp_path), "inv", ending="nominal")
    original = out["steps"][0]["set"]["issued"]
    assert original["goal_version"] == 1 and original["issued_step"] == 0
    assert original["context_sha256"] == "b" * 64 and original["reply_sha256"] == "c" * 64
    assert out["complete"] and out["K"] == 1


def test_explicit_prefix_and_grouped_mixed_summaries_keep_denominators(tmp_path):
    log, heads = signed(tmp_path)
    complete = A.derive(log, heads, "inv")
    prefix = A.derive(log, heads, "inv", prefix=0)
    assert prefix["complete"] is False and prefix["K"] == 0
    result = A.pool([complete, prefix], mixed=True)
    assert result["completed"]["K"] == 1 and result["incomplete"]["K"] == 0
    assert result["incomplete"]["episodes_with_no_steps"] == 1
    assert result["mixed_completed_and_incomplete"]["pooled_shares"]["small_model"] == 1


def test_a_known_episode_with_no_action_steps_remains_explicitly_empty(tmp_path):
    rows = [{"episode": "inv", "kind": "briefing", "t": 0, "action_step": 0, "briefing": {}}]
    result = A.derive(*rewrite(tmp_path, rows), "inv")
    assert result["K"] == 0 and not result["complete"] and all(v is None for v in result["shares"].values())


def test_numeric_origin_normal_exchange_and_evidence_no_real_socket(monkeypatch):
    raw = b'HTTP/1.0 200 OK\r\nContent-Length: 2\r\n\r\n{}'
    fake = SocketStandIn([raw, b""])
    monkeypatch.setattr(socket, "create_connection", lambda *a, **kw: fake)
    seen = []
    t = T.LoopbackTransport("http://127.0.0.1:9", 1000, record=seen.append)
    assert t("http://127.0.0.1:9/x", {"model": "invented"}, 1) == (200, {})
    assert fake.closed and fake.sent.startswith(b"POST /x HTTP/1.0")
    assert seen[0]["reply_sha256"] == hashlib.sha256(raw).hexdigest()
    assert seen[0]["outcome"] == "replied"


def test_finite_nested_json_and_exact_content_length_remain_accepted():
    assert T._parse_json(b'{"outer":[1e3,null,true]}') == {"outer": [1000., None, True]}
    assert T._parse_http(b'HTTP/1.0 200 OK\r\nContent-Length: 2\r\n\r\n{}') == (200, b"{}")
