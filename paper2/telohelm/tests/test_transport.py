"""The draft loopback transport (TURN 79) against invented servers that the tests start on 127.0.0.1. No model server,
no other host and no real model text."""
from __future__ import annotations

import hashlib
import json
import socket
import threading
import time

import pytest

import telohelm.runtime.models as M
import telohelm.runtime.transport as T
from telohelm.runtime.transport import LoopbackTransport; from telohelm.runtime.transport import TransportRefused
from .test_local_adapter import REQ; from .test_local_adapter import ROLES; from .test_local_adapter import TIMEOUTS; from .test_local_adapter import ok_body


class Invented:
    """A one-connection-at-a-time server on 127.0.0.1 that answers each request with reply(request_bytes), a list of
    (bytes, pause_s) pieces sent in order before it closes."""

    def __init__(self, reply):
        self.reply, self.requests = reply, []
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen()
        self.port = self.sock.getsockname()[1]
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        self.sock.settimeout(0.05)
        while not self.stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except OSError:
                continue
            with conn:
                conn.settimeout(2.0)
                data = b""
                try:
                    while b"\r\n\r\n" not in data:
                        data += conn.recv(65536)
                    head, _, body = data.partition(b"\r\n\r\n")
                    n = int([l.split(b":")[1] for l in head.split(b"\r\n") if l.lower().startswith(b"content-length")][0])
                    while len(body) < n:
                        body += conn.recv(65536)
                    self.requests.append(head + b"\r\n\r\n" + body)
                    for piece, pause in self.reply(head + b"\r\n\r\n" + body):
                        if self.stop.is_set():
                            break
                        if piece:
                            conn.sendall(piece)
                        time.sleep(pause)
                except OSError:
                    pass

    def close(self):
        self.stop.set()
        self.sock.close()
        self.thread.join(2.0)


def http(body: bytes, status=b"200 OK", length=True):
    head = b"HTTP/1.1 " + status + b"\r\nContent-Type: application/json\r\n"
    if length:
        head += b"Content-Length: " + str(len(body)).encode() + b"\r\n"
    return head + b"\r\n" + body


@pytest.fixture
def server():
    made = []

    def start(reply):
        s = Invented(reply)
        made.append(s)
        return s
    yield start
    for s in made:
        s.close()


def transport(srv, **kw):
    return LoopbackTransport(f"http://127.0.0.1:{srv.port}", kw.pop("max_response_bytes", 1 << 20), **kw)


def test_a_normal_exchange_and_its_record(server):
    body = json.dumps(ok_body()).encode()
    srv = server(lambda req: [(http(body), 0)])
    seen = []
    t = transport(srv, record=seen.append)
    status, data = t(f"http://127.0.0.1:{srv.port}/v1/chat/completions", {"model": "m"}, 2.0)
    assert status == 200 and data == ok_body()
    rec = seen[0]
    assert rec["outcome"] == "replied" and rec["http_status"] == 200 and rec["reply_complete"] is True
    assert rec["request_sha256"] == hashlib.sha256(srv.requests[0]).hexdigest()
    assert rec["reply_sha256"] == hashlib.sha256(http(body)).hexdigest() and rec["reply_bytes"] == len(http(body))
    assert srv.requests[0].startswith(b"POST /v1/chat/completions HTTP/1.0\r\n")
    assert "invented answer" not in json.dumps(rec)                     # the record carries no model text


def test_one_deadline_covers_a_slow_drip(server):
    reply = http(json.dumps(ok_body()).encode())
    # One byte every 0.05 s: each read succeeds well inside any per-read timeout, but the whole reply takes seconds.
    srv = server(lambda req: [(reply[i:i + 1], 0.05) for i in range(len(reply))])
    seen = []
    t = transport(srv, record=seen.append)
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        t(f"http://127.0.0.1:{srv.port}/v1/chat/completions", {}, 0.4)
    elapsed = time.monotonic() - start
    assert 0.35 <= elapsed < 1.5 and len(reply) * 0.05 > 5           # stopped at the deadline, long before the end
    assert seen[0]["outcome"] == "deadline"


def test_a_server_that_never_answers_times_out(server):
    srv = server(lambda req: [(b"", 3.0)])
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        transport(srv)(f"http://127.0.0.1:{srv.port}/x", {}, 0.3)
    assert time.monotonic() - start < 1.5


def test_the_deadline_uses_the_clock_given():
    ticks = iter([0.0, 5.0, 5.0])
    t = LoopbackTransport("http://127.0.0.1:9", 100, clock=lambda: next(ticks))
    with pytest.raises(TimeoutError):                                   # already past the deadline before connecting
        t("http://127.0.0.1:9/x", {}, 1.0)


@pytest.mark.parametrize("origin", ["http://example.org:80", "https://127.0.0.1:8000", "http://127.0.0.1",
                                    "http://10.0.0.1:8000", "http://127.0.0.1:8000/v1", "http://u:p@127.0.0.1:8000",
                                    "http://localhost.example.org:8000", "http://localhost:8000"])
def test_only_a_loopback_origin_with_a_port_can_be_pinned(origin):
    with pytest.raises(ValueError):
        LoopbackTransport(origin, 100)


@pytest.mark.parametrize("url", ["http://127.0.0.2:{p}/x", "http://127.0.0.1:{q}/x", "http://127.0.0.1:{p}",
                                 "http://127.0.0.1:{p}/x y", "http://127.0.0.1:{p}/x\r\nEvil: 1",
                                 "https://127.0.0.1:{p}/x", "http://127.0.0.1:{p}@example.org/x",
                                 "http://127.0.0.1:{p}/x?key=invented"])
def test_a_url_outside_the_origin_is_refused_without_contact(server, url):
    srv = server(lambda req: [(http(b"{}"), 0)])
    with pytest.raises(TransportRefused):
        transport(srv)(url.format(p=srv.port, q=srv.port + 1), {}, 1.0)
    time.sleep(0.1)
    assert srv.requests == []


def test_a_long_reply_is_refused(server):
    srv = server(lambda req: [(http(b'{"a":"' + b"x" * 5000 + b'"}'), 0)])
    with pytest.raises(TransportRefused):
        transport(srv, max_response_bytes=1000)(f"http://127.0.0.1:{srv.port}/x", {}, 2.0)


@pytest.mark.parametrize("raw", [b"not json", b'{"a": NaN}', b'{"a": Infinity}', b"[1, 2]", b'"text"',
                                 b'{"a": "\xff"}', b'{"a": {"b": [{"c": -1e400}]}}', b'{"a": {"b": 1, "b": 2}}'])
def test_a_body_that_is_not_strict_finite_json_is_none(server, raw):
    srv = server(lambda req: [(http(raw), 0)])
    assert transport(srv)(f"http://127.0.0.1:{srv.port}/x", {}, 2.0) == (200, None)


@pytest.mark.parametrize("reply", [b"garbage with no header end", b"HTTP/1.1 OK\r\n\r\n{}",
                                   b"HTTP/1.1 200 OK\r\nContent-Length: 50\r\n\r\n{}",
                                   b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\n{}\r\n0\r\n\r\n",
                                   b"HTTP/1.1 200 OK\r\nbad header\r\n\r\n{}",
                                   b"HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\n\r\n{}",
                                   b"HTTP/1.1 200 OK\r\nContent-Length: +2\r\n\r\n{}"])
def test_malformed_http_is_refused(server, reply):
    srv = server(lambda req: [(reply, 0)])
    with pytest.raises(TransportRefused):
        transport(srv)(f"http://127.0.0.1:{srv.port}/x", {}, 2.0)


def test_a_redirect_is_not_followed(server):
    srv = server(lambda req: [(http(b"{}", status=b"307 Temporary Redirect"), 0)])
    assert transport(srv)(f"http://127.0.0.1:{srv.port}/x", {}, 2.0) == (307, {})
    assert len(srv.requests) == 1


def test_a_closed_port_is_a_transport_error():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    seen = []
    with pytest.raises(OSError):
        LoopbackTransport(f"http://127.0.0.1:{port}", 100, record=seen.append)(f"http://127.0.0.1:{port}/x", {}, 1.0)
    assert seen[0]["outcome"] == "transport error"


@pytest.mark.parametrize("body,timeout", [({"a": float("nan")}, 1.0), ({"a": object()}, 1.0), ({}, 0), ({}, float("inf")),
                                          ({}, True)])
def test_a_bad_request_or_timeout_is_refused_before_contact(server, body, timeout):
    srv = server(lambda req: [(http(b"{}"), 0)])
    with pytest.raises(TransportRefused):
        transport(srv)(f"http://127.0.0.1:{srv.port}/x", body, timeout)


def test_through_the_adapter(server):
    good = server(lambda req: [(http(json.dumps(ok_body()).encode()), 0)])
    a = M.LocalChatProvider(f"http://127.0.0.1:{good.port}", ROLES, transport(good), TIMEOUTS)
    r = a.complete(REQ)
    assert r.status == "answered" and r.text == "invented answer" and r.tokens_in == 11
    sent = json.loads(good.requests[0].partition(b"\r\n\r\n")[2])
    assert sent == a.body(REQ)
    slow = server(lambda req: [(b"", 3.0)])
    a = M.LocalChatProvider(f"http://127.0.0.1:{slow.port}", ROLES, transport(slow),
                            {"small_model": 0.3, "reasoning_model": 0.3})
    assert a.complete(REQ).status == "timed_out"
    bad = server(lambda req: [(b"HTTP/1.1 200 OK\r\nContent-Length: 99\r\n\r\n{}", 0)])
    a = M.LocalChatProvider(f"http://127.0.0.1:{bad.port}", ROLES, transport(bad), TIMEOUTS)
    assert a.complete(REQ).status == "failed"


def test_a_failed_exchange_keeps_what_it_received_marked_incomplete(server):
    # TURN 81 (Codex's TR80.RECORD), against a real loopback socket: a long reply refused at the cap, and a slow one
    # stopped at the deadline, each keep the bytes received and say the reply was not complete.
    srv = server(lambda req: [(b"x" * 5000, 0)])
    seen = []
    with pytest.raises(TransportRefused):
        transport(srv, max_response_bytes=1000, record=seen.append)(f"http://127.0.0.1:{srv.port}/x", {}, 2.0)
    assert seen[0]["reply_complete"] is False and seen[0]["reply_bytes"] > 1000
    slow = server(lambda req: [(b"HTTP/1.0 200 OK\r\n", 3.0)])
    seen = []
    with pytest.raises(TimeoutError):
        transport(slow, record=seen.append)(f"http://127.0.0.1:{slow.port}/x", {}, 0.3)
    assert seen[0]["reply_bytes"] == 17 and seen[0]["reply_complete"] is False
    assert seen[0]["reply_sha256"] == hashlib.sha256(b"HTTP/1.0 200 OK\r\n").hexdigest()
    seen = []
    with pytest.raises(TransportRefused):
        transport(slow, record=seen.append)("http://127.0.0.2:1/x", {}, 1.0)
    assert seen[0]["reply_bytes"] == 0 and seen[0]["reply_complete"] is False and "reply_sha256" not in seen[0]


def test_numbers_in_the_status_line_and_length_are_bounded_before_conversion():
    # TURN 83 (Codex's TR82.PARSE): a status outside 100-599 or not three digits, an unknown version, and a length too
    # long to be real are transport refusals; the digit limit of int() is never reached.
    for raw in [b"HTTP/1.0 099 X\r\n\r\n{}", b"HTTP/1.0 2000 X\r\n\r\n{}", b"HTTP/2 200 X\r\n\r\n{}",
                b"HTTP/1.0 200 OK\r\nContent-Length: 1234567890123456\r\n\r\n{}"]:
        with pytest.raises(TransportRefused):
            T._parse_http(raw)


def test_an_ipv6_loopback_origin_brackets_its_host():
    t = LoopbackTransport("http://[::1]:8000", 100)
    assert t.authority == "[::1]:8000" and t.origin == "http://[::1]:8000"
    assert LoopbackTransport("http://127.0.0.1:8000", 100).authority == "127.0.0.1:8000"
