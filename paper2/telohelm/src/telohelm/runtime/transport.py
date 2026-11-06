"""A draft HTTP transport for LocalChatProvider with one deadline for the whole request (paper2/notes/TRANSPORT-CONTRACT.md;
Codex's TURN 78 request). Draft for cross-review, TURN 79. Tested only against invented servers on the loopback
interface inside the tests; it has never contacted a model server.

The adapter calls transport(url, body, timeout_s) -> (http_status, json_body). This transport:
- reaches only the one origin it was built with, which must be http on a numeric loopback address (in 127.0.0.0/8, or
  ::1) with an explicit port. A name such as localhost is refused, so no name lookup can send the connection anywhere
  else and there is exactly one address to try (Codex's TR80.CONNECT). Any other URL is refused before a socket is
  opened. On the cluster the model server runs on the same node, so loopback is enough; any other route would be a
  reviewed change;
- spends at most timeout_s from the start of the call to the last byte read, across connecting, sending, waiting and
  reading. Before each socket operation the socket's timeout is set to the time left, so a server that sends one byte at
  a time cannot extend the request. Running out of time raises TimeoutError, which the adapter maps to timed_out;
- sends HTTP/1.0 with Connection: close, so the reply is a plain body ending when the server closes, with no chunked
  encoding and no kept-alive connection. Redirects are not followed;
- reads at most max_response_bytes, header included, and refuses a longer reply;
- refuses a reply whose framing is ambiguous or unsupported: more than one Content-Length, a length that disagrees with
  the body, any Transfer-Encoding, or a Content-Encoding other than identity (Codex's TR80.HTTP);
- returns the body as JSON only if it is strict UTF-8 JSON with finite numbers at every depth and no repeated name in
  any object; otherwise the body is None, which the adapter treats as failed with the HTTP status kept (TR80.JSON);
- reports each exchange to an optional record callback: the URL, the SHA-256 of the exact request bytes, the SHA-256
  and count of the reply bytes actually received, whether the reply was complete (the server closed the connection
  within the deadline and the byte limit), the HTTP status, the outcome and the elapsed seconds. A failed exchange
  keeps the bytes it did receive, marked incomplete (TR80.RECORD). Nothing in the record is model text.

Every refusal and transport fault raises TransportRefused or another OSError, which the adapter maps to failed.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import socket
import time
from typing import Any, Callable
from urllib.parse import urlsplit

class TransportRefused(OSError):
    """The transport would not make or finish this exchange: the URL is outside its origin, or the reply is too long
    or malformed at the HTTP level."""


def _loopback(host: str) -> bool:
    """A numeric loopback address. Names are refused, so nothing is looked up."""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


class LoopbackTransport:
    """transport(url, body, timeout_s) for one pinned loopback origin, with a whole-request deadline."""

    def __init__(self, origin: str, max_response_bytes: int, record: Callable[[dict], None] | None = None,
                 clock: Callable[[], float] = time.monotonic):
        parts = urlsplit(origin)
        if (parts.scheme != "http" or not parts.hostname or not _loopback(parts.hostname) or parts.port is None
                or parts.path not in ("", "/") or parts.query or parts.fragment or parts.username or parts.password):
            raise ValueError("the origin must be http://<numeric loopback address>:<port> with nothing after it")
        if not isinstance(max_response_bytes, int) or isinstance(max_response_bytes, bool) or max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be a positive integer")
        self.origin = f"http://{parts.netloc}"
        self.host, self.port = parts.hostname, parts.port
        # The Host header names the authority as in the URL, so an IPv6 address keeps its brackets (Codex's TR82.IPV6).
        self.authority = f"[{self.host}]:{self.port}" if ":" in self.host else f"{self.host}:{self.port}"
        self.max_response_bytes, self._record, self._clock = max_response_bytes, record, clock

    def __call__(self, url: str, body: dict, timeout_s: float) -> tuple[int, dict | None]:
        start = self._clock()
        rec: dict[str, Any] = {"url": url, "outcome": None}
        wire: dict[str, Any] = {}
        try:
            self._exchange(url, body, timeout_s, start, rec, wire)
            status, payload_bytes = _parse_http(wire["reply"])
            rec["http_status"] = status
            rec["outcome"] = "replied"
            return status, _parse_json(payload_bytes)
        except TimeoutError:
            rec["outcome"] = "deadline"
            raise
        except OSError as e:
            rec["outcome"] = "refused" if isinstance(e, TransportRefused) else "transport error"
            raise
        finally:
            rec.setdefault("reply_bytes", 0)              # refused or failed before any reply byte arrived
            rec.setdefault("reply_complete", False)
            rec["elapsed_s"] = self._clock() - start
            if self._record is not None:
                self._record(rec)

    def exchange(self, url: str, body: dict, timeout_s: float) -> dict[str, Any]:
        """The exchange's evidence with the reply left uninterpreted, so that it can be saved before anything parses it
        (Codex's TURN 82 requirement: pre-parse durability). A transport fault does not raise; the outcome names it:
        received (the server closed in time and within the byte limit), deadline, refused or transport error.
        request_wire and reply_wire are the exact bytes sent and received, reply_wire possibly partial, and either is
        None if nothing got that far. interpret() turns the evidence into (status, body) as __call__ would."""
        start = self._clock()
        rec: dict[str, Any] = {}
        wire: dict[str, Any] = {"request": None, "reply": None}
        try:
            self._exchange(url, body, timeout_s, start, rec, wire)
            outcome = "received"
        except TimeoutError:
            outcome = "deadline"
        except TransportRefused:
            outcome = "refused"
        except OSError:
            outcome = "transport error"
        return {"url": url, "outcome": outcome, "request_wire": wire["request"], "reply_wire": wire["reply"],
                "reply_complete": rec.get("reply_complete", False), "elapsed_s": self._clock() - start}

    def _exchange(self, url: str, body: dict, timeout_s: float, start: float, rec: dict, wire: dict) -> None:
        if not _finite(timeout_s) or timeout_s <= 0:
            raise TransportRefused("the timeout must be finite and positive")
        if not isinstance(url, str) or not url.startswith(self.origin + "/"):
            raise TransportRefused("the URL is outside the pinned origin")
        path = url[len(self.origin):]
        if any(c in path for c in "\r\n #?"):
            # No query string either: the record keeps the URL, and the adapter never needs one.
            raise TransportRefused("the path is not a plain request path")
        try:
            payload = json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as e:
            raise TransportRefused("the request body is not finite JSON") from e
        head = (f"POST {path} HTTP/1.0\r\nHost: {self.authority}\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n").encode("ascii")
        request = head + payload
        wire["request"] = request
        rec.update(request_sha256=hashlib.sha256(request).hexdigest(), request_bytes=len(request))
        deadline = start + timeout_s

        def left() -> float:
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise TimeoutError("the request deadline passed")
            return remaining

        # A numeric address resolves to itself with no lookup, so there is one candidate and it gets the time left.
        sock = socket.create_connection((self.host, self.port), timeout=left())
        reply, complete = bytearray(), False
        try:
            sock.settimeout(left())
            sock.sendall(request)
            while True:
                sock.settimeout(left())
                try:
                    chunk = sock.recv(65536)
                except socket.timeout as e:
                    raise TimeoutError("the request deadline passed") from e
                if not chunk:
                    complete = True
                    break
                reply += chunk
                if len(reply) > self.max_response_bytes:
                    raise TransportRefused("the reply is longer than the limit")
        finally:
            sock.close()
            # What was received is kept on every exit, marked complete only if the server closed in time and in bounds.
            wire["reply"] = bytes(reply)
            rec.update(reply_sha256=hashlib.sha256(bytes(reply)).hexdigest(), reply_bytes=len(reply),
                       reply_complete=complete)


def interpret(evidence: dict) -> tuple[int, dict | None]:
    """(status, body) from saved exchange evidence, raising as the transport call would: TimeoutError for the deadline,
    TransportRefused for a refusal or a malformed reply, OSError for another transport fault."""
    outcome = evidence.get("outcome")
    if outcome == "deadline":
        raise TimeoutError("the request deadline passed")
    if outcome == "refused":
        raise TransportRefused("the exchange was refused")
    if outcome != "received" or not isinstance(evidence.get("reply_wire"), bytes):
        raise OSError("the exchange failed")
    status, payload_bytes = _parse_http(evidence["reply_wire"])
    return status, _parse_json(payload_bytes)


def _parse_http(reply: bytes) -> tuple[int, bytes]:
    head, sep, body = reply.partition(b"\r\n\r\n")
    if not sep:
        raise TransportRefused("the reply has no complete header")
    lines = head.split(b"\r\n")
    status_line = lines[0].split(b" ", 2)
    # Numbers are checked for form and size before conversion, so an absurd field is a transport refusal and never
    # reaches int(), whose own digit limit would raise an unrelated ValueError (Codex's TR82.PARSE).
    if (len(status_line) < 2 or status_line[0] not in (b"HTTP/1.0", b"HTTP/1.1") or len(status_line[1]) != 3
            or not status_line[1].isdigit() or not 100 <= int(status_line[1]) <= 599):
        raise TransportRefused("the reply's status line is malformed")
    headers: dict[bytes, list[bytes]] = {}
    for line in lines[1:]:
        name, colon, value = line.partition(b":")
        if not colon or not name.strip():
            raise TransportRefused("a reply header is malformed")
        headers.setdefault(name.strip().lower(), []).append(value.strip())
    if b"transfer-encoding" in headers:
        raise TransportRefused("a transfer encoding in a reply to an HTTP/1.0 request")
    if any(v.lower() != b"identity" for v in headers.get(b"content-encoding", [])):
        raise TransportRefused("an encoded reply body")
    if b"content-length" in headers:
        values = headers[b"content-length"]
        if len(values) != 1 or not values[0].isdigit() or len(values[0]) > 15:
            raise TransportRefused("the reply's length header is repeated, malformed or absurdly large")
        if int(values[0]) != len(body):
            raise TransportRefused("the reply's body is not the length its header gives")
    return int(status_line[1]), body


def _parse_json(body: bytes) -> dict | None:
    def no_constant(name: str) -> Any:
        raise ValueError(name)

    def finite_float(text: str) -> float:
        v = float(text)
        if not math.isfinite(v):          # 1e999 parses to infinity
            raise ValueError(text)
        return v

    def unique(pairs: list) -> dict:
        names = [k for k, _ in pairs]
        if len(set(names)) != len(names):
            raise ValueError("a repeated name")
        return dict(pairs)
    try:
        data = json.loads(body.decode("utf-8"), parse_constant=no_constant, parse_float=finite_float,
                          object_pairs_hook=unique)
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    return data if isinstance(data, dict) else None
