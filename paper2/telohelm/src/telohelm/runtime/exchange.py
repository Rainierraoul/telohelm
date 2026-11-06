"""The private exchange store: the exact bytes of each model exchange, saved before the reply is parsed
(paper2/notes/TRANSPORT-CONTRACT.md section 3; Codex's TURN 82 and TURN 84 requirements). Draft for cross-review,
TURN 83, revised TURN 85 and TURN 87. Used only by LocalChatProvider when it is given a store; nothing in the runtime gives it one.

One record per exchange, appended as one JSON line and synced to disk before the call returns. A record names the model
request it answered: the request identifier and attempt, and (from the adapter) the full original request identity,
which is the episode, condition and stream, the content fingerprint, the request's details (goal version, issue second
and step, modelled latency, attempt, output limit) and the endpoint. It also holds the configuration fingerprint, the
role, the SHA-256 of the request body, whether the adapter that made the exchange was reportable, the exact request and
reply bytes (base64) with their digests and lengths, whether the reply was complete, the transport's outcome and the
wire time. Completeness on the wire is kept apart from usability: a complete reply can still be one the adapter refuses.

Fail closed:
- loading raises ExchangeError on any record that is malformed, incoherent (for example received but incomplete), whose
  bytes do not match their digests, or that repeats an exchange, and on a line that is not a record;
- a lookup for the same request identifier and attempt with any other identity, body or configuration raises, so stored
  bytes are never reused for another question or relabelled with other issue details (Codex's EX84.IDENTITY);
- a record becomes visible to get() only after its line is synced, and the whole save holds the store's lock, so a
  reader never sees a record that may not be on disk (EX84.DURABLE);
- if writing or syncing a line fails, the store is tainted: every later save and get raises, because the line may or
  may not be on disk. Reconciling means opening the file again in a new store, which admits a complete line as a
  committed record and refuses a torn one. The store never appends after a failure it cannot account for;
- a file whose last line has no terminator is refused on opening, even if that line is complete JSON, because the next
  append would join onto it (Codex's EX86.TAIL). A person reconciles it;
- records handed out are copies, so a caller cannot change a committed record in memory (EX86.COPY).

Anchoring (TURN 87, Codex's ANCH86.GATE): after the gateway has written an exchange's anchor into its trusted log, the
adapter marks the exchange as anchored in a sidecar file next to the store (the store's path plus ".anchored"), synced
in the same way. A stored exchange without that mark was saved but may never have been anchored, for example because the
run stopped between the two writes. Such an exchange is quarantined on reuse: its evidence and observed usage are kept,
but it gives no usable text and is never reportable. A later reuse never adds the mark. The mark is a runtime gate in
unprotected storage; the authority remains models.verify_anchors() against the protected logs.

From TURN 89 (Codex's ANCH88.PREUSE) the mark is only a cache: it also names the gateway log and protected heads that
hold the original anchor, and before a stored exchange's text is used the adapter verifies that log and finds the
original anchor in it. A mark without those references, or one whose log does not verify or does not hold a matching
anchor joined to the original request, gives no admission, and the exchange is quarantined.

The store is private run data. It holds model text, so it is never a public artefact. Its records carry their own
digests, which show only that a record agrees with itself; they do not show where it came from. The anchors the gateway
writes into its trusted log when an exchange is saved or reused carry the whole original record except the bytes, whose
digests it holds, and are what authenticate it (models.verify_anchors). The store itself is not hash-chained. That is a
choice of this draft, with the protected anchors doing the authenticating; a chained store that a resumed run reopens
and extends would also be possible (an earlier note gave resuming as the reason, which was wrong). Its lock covers
threads in one process; two processes must not share a store, and nothing here enforces that.
"""
from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import json
import math
import os
import re
import threading
from pathlib import Path
from typing import Any

from .manifest import ROLES

_HEX64 = re.compile(r"[0-9a-f]{64}")
OUTCOMES = ("received", "deadline", "refused", "transport error")
FIELDS = frozenset({"exchange_id", "request_id", "attempt_id", "request", "config_sha256", "role", "body_sha256",
                    "reportable", "outcome", "request_wire", "request_sha256", "reply_wire", "reply_sha256",
                    "reply_bytes", "reply_complete", "elapsed_s"})


class ExchangeError(ValueError):
    """The store is corrupt or tainted, or a lookup does not match what was stored."""


def _sha(b: bytes | None) -> str | None:
    return None if b is None else hashlib.sha256(b).hexdigest()


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def body_sha256(body: dict) -> str:
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


def _b64(b: bytes | None) -> str | None:
    return None if b is None else base64.b64encode(b).decode("ascii")


def _unb64(s: Any) -> bytes | None:
    if s is None:
        return None
    if not isinstance(s, str):
        raise ExchangeError("stored bytes are not base64 text")
    try:
        return base64.b64decode(s.encode("ascii"), validate=True)
    except (binascii.Error, UnicodeEncodeError) as e:
        raise ExchangeError("stored bytes are not valid base64") from e


def summary(rec: dict) -> dict:
    """A stored record without its bytes: every field the anchor carries, including the bytes' digests."""
    return {k: copy.deepcopy(v) for k, v in rec.items() if k not in ("request_wire", "reply_wire")}


def record_sha256(rec: dict) -> str:
    return hashlib.sha256(_canonical(summary(rec)).encode("utf-8")).hexdigest()


def _lines(path: Path) -> list[bytes]:
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ExchangeError(f"{path.name} ends without a line terminator; reconcile it before any append")
    return data.splitlines()


def _text(v: Any) -> bool:
    if not isinstance(v, str) or not v:
        return False
    try:
        v.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


class ExchangeStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._records: dict[str, dict] = {}
        self._tainted: str | None = None
        self.anchor_path = self.path.with_name(self.path.name + ".anchored")
        self._anchored: dict[str, dict] = {}         # exchange id -> {"log", "heads"} references, possibly empty
        if self.path.exists():
            for n, line in enumerate(_lines(self.path), start=1):
                try:
                    rec = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, ValueError) as e:
                    raise ExchangeError(f"line {n} of the exchange store is not a record") from e
                self._check(rec, n)
                self._records[rec["exchange_id"]] = rec
        if self.anchor_path.exists():
            for n, line in enumerate(_lines(self.anchor_path), start=1):
                try:
                    mark = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, ValueError) as e:
                    raise ExchangeError(f"line {n} of the anchor marks is not a mark") from e
                xid = mark.get("exchange_id") if isinstance(mark, dict) else None
                keys = set(mark) if isinstance(mark, dict) else set()
                if (not {"exchange_id", "record_sha256"} <= keys <= {"exchange_id", "record_sha256", "log", "heads"}
                        or xid not in self._records or xid in self._anchored):
                    raise ExchangeError(f"anchor mark {n} does not name a stored record once")
                if mark["record_sha256"] == record_sha256(self._records[xid]):
                    self._anchored[xid] = {k: mark[k] for k in ("log", "heads")
                                           if k in mark and isinstance(mark[k], str)}
                # A mark for a record that has since changed does not count: the exchange is then unanchored and is
                # quarantined on reuse, and verify_anchors reports the change against the protected log.

    @staticmethod
    def exchange_id(request_id: str, attempt_id: str) -> str:
        return json.dumps([request_id, attempt_id], ensure_ascii=False, separators=(",", ":"))

    def _check(self, rec: Any, n: int) -> None:
        """Every rule a stored record must meet (Codex's EX84.SCHEMA). Raises on the first it fails."""
        def bad(why: str) -> ExchangeError:
            return ExchangeError(f"record {n}: {why}")
        if not isinstance(rec, dict) or set(rec) != FIELDS:
            raise bad("the wrong fields")
        if not _text(rec["request_id"]) or not _text(rec["attempt_id"]):
            raise bad("the request identifier and attempt must be non-empty UTF-8 text")
        if rec["exchange_id"] != self.exchange_id(rec["request_id"], rec["attempt_id"]):
            raise bad("an exchange identifier that does not match its request")
        if rec["request"] is not None and not isinstance(rec["request"], dict):
            raise bad("a request identity that is not an object")
        if rec["role"] not in ROLES:
            raise bad(f"an unknown role {rec['role']!r}")
        if not isinstance(rec["reportable"], bool):
            raise bad("reportability that is not True or False")
        if not _HEX64.fullmatch(str(rec["config_sha256"])) or not _HEX64.fullmatch(str(rec["body_sha256"])):
            raise bad("a malformed configuration or body digest")
        el = rec["elapsed_s"]
        if isinstance(el, bool) or not isinstance(el, (int, float)) or not math.isfinite(el) or el < 0:
            raise bad("a wire time that is not a finite non-negative number")
        req, rep = _unb64(rec["request_wire"]), _unb64(rec["reply_wire"])
        if _sha(req) != rec["request_sha256"] or _sha(rep) != rec["reply_sha256"]:
            raise bad("bytes that do not match their digests")
        if rec["reply_bytes"] != (0 if rep is None else len(rep)):
            raise bad("a reply length that does not match its bytes")
        outcome, complete = rec["outcome"], rec["reply_complete"]
        if outcome not in OUTCOMES or not isinstance(complete, bool):
            raise bad("an unknown outcome")
        # Coherence: received means the whole reply arrived; anything else is incomplete. Only a refusal can come
        # before any request bytes existed.
        if complete != (outcome == "received") or (outcome == "received" and rep is None):
            raise bad(f"outcome {outcome!r} with reply_complete {complete}")
        if req is None and outcome != "refused":
            raise bad(f"outcome {outcome!r} with no request bytes")
        if rec["exchange_id"] in self._records:
            raise bad(f"a repeat of exchange {rec['exchange_id']}")

    def _usable(self) -> None:
        if self._tainted is not None:
            raise ExchangeError(f"the store is tainted ({self._tainted}); reopen it to reconcile")

    def get(self, request_id: str, attempt_id: str, config_sha256: str, body_sha: str,
            request: dict | None = None) -> dict | None:
        """The committed evidence for this request, with its bytes decoded, or None. A stored exchange for the same
        identifier and attempt with any other request identity, body or configuration raises ExchangeError."""
        with self._lock:
            self._usable()
            rec = self._records.get(self.exchange_id(request_id, attempt_id))
        if rec is None:
            return None
        if (rec["config_sha256"] != config_sha256 or rec["body_sha256"] != body_sha
                or rec["request"] != (None if request is None else json.loads(_canonical(request)))):
            raise ExchangeError("a stored exchange for this request was made with another identity, body or "
                                "configuration")
        return self._copy(rec)

    def _copy(self, rec: dict) -> dict:
        return {**copy.deepcopy(rec), "request_wire": _unb64(rec["request_wire"]),
                "reply_wire": _unb64(rec["reply_wire"]), "anchored": rec["exchange_id"] in self._anchored}

    def anchor_ref(self, exchange_id: str) -> dict | None:
        """The log references the mark gives for this exchange ({} if it gives none), or None if it is unmarked."""
        with self._lock:
            ref = self._anchored.get(exchange_id)
            return None if ref is None else dict(ref)

    def mark_anchored(self, exchange_id: str, log: Path | None = None, heads: Path | None = None) -> None:
        """Record, synced, that this exchange's anchor is in the protected log named. Raises if it is unknown or
        already marked; a failed write taints the store."""
        with self._lock:
            self._usable()
            if exchange_id not in self._records or exchange_id in self._anchored:
                raise ExchangeError(f"exchange {exchange_id} is unknown or already marked as anchored")
            mark = {"exchange_id": exchange_id, "record_sha256": record_sha256(self._records[exchange_id])}
            if log is not None and heads is not None:
                mark.update(log=str(log), heads=str(heads))
            line = json.dumps(mark, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
            try:
                with self.anchor_path.open("ab") as f:
                    f.write(line + b"\n")
                    f.flush()
                    os.fsync(f.fileno())
            except BaseException as e:
                self._tainted = f"{type(e).__name__} while marking {exchange_id}"
                raise
            self._anchored[exchange_id] = {k: mark[k] for k in ("log", "heads") if k in mark}

    def save(self, request_id: str, attempt_id: str, config_sha256: str, role: str, body_sha: str,
             evidence: dict, request: dict | None = None, reportable: bool = False) -> dict:
        """Append one exchange, sync it, and only then make it visible; returns the stored record without the bytes."""
        req, rep = evidence.get("request_wire"), evidence.get("reply_wire")
        rec = {"exchange_id": self.exchange_id(request_id, attempt_id), "request_id": request_id,
               "attempt_id": attempt_id, "request": None if request is None else json.loads(_canonical(request)),
               "config_sha256": config_sha256, "role": role, "body_sha256": body_sha, "reportable": reportable,
               "outcome": evidence.get("outcome"), "request_wire": _b64(req), "request_sha256": _sha(req),
               "reply_wire": _b64(rep), "reply_sha256": _sha(rep), "reply_bytes": 0 if rep is None else len(rep),
               "reply_complete": evidence.get("reply_complete"), "elapsed_s": evidence.get("elapsed_s")}
        with self._lock:
            self._usable()
            self._check(rec, len(self._records) + 1)
            line = json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
            try:
                with self.path.open("ab") as f:
                    f.write(line + b"\n")
                    f.flush()
                    os.fsync(f.fileno())
            except BaseException as e:
                self._tainted = f"{type(e).__name__} while saving {rec['exchange_id']}"
                raise
            self._records[rec["exchange_id"]] = rec
        return {**summary(rec), "anchored": False}

    def records(self) -> list[dict]:
        with self._lock:
            self._usable()
            return [self._copy(r) for r in self._records.values()]
