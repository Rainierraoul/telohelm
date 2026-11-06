"""Shared model servers: at most n requests in flight to one server, across every process of a job.

Several conditions can fly at once against one pair of model servers (paper2/notes/PATH-TO-FREEZE.md, item 8). Each
server takes requests up to its own limit and queues the rest, and a queued request would spend its time limit
waiting. SlotTransport wraps a transport so that a request first takes one of n slots, which are lock files in a
directory every process of the job shares. The wrapped transport's deadline starts only once a slot is held, so time
spent waiting never counts against a request's timeout and never changes an episode: simulated time runs by fixed
modelled call times, whatever the wall clock does.

Queue entry, acquisition and release are appended to waits.jsonl with one unique wait_id. A killed waiter therefore
keeps its queue entry; absence of a later event is censored, never zero wait. These are client lease intervals, not
GPU execution intervals. The gateway's measured wall time includes waiting. Polling is not FIFO and may starve a
waiter; the job's fixed allocation limit bounds the rehearsal, not an individual queue timeout.
The release outcome names a Python return/exception, not the model's completion status.

Configuration is one environment variable for the job, TELOHELM_MODEL_SLOTS, a JSON object mapping each server's base
URL to {"dir": <slot directory>, "n": <slots>}. local_model.provider() reads it. Unset, nothing is wrapped.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

ENV = "TELOHELM_MODEL_SLOTS"
POLL_S = 0.5


def configured(base_url: str) -> dict | None:
    """This job's slot setting for a server, or None. A malformed setting raises, so a job never runs unlimited by
    mistake."""
    raw = os.environ.get(ENV)
    if not raw:
        return None
    table = json.loads(raw)
    if not isinstance(table, dict):
        raise ValueError(f"{ENV} must be a JSON object of base URL to settings")
    entry = table.get(base_url.rstrip("/"))
    if entry is None:
        return None
    if (not isinstance(entry, dict) or set(entry) != {"dir", "n"} or not isinstance(entry["dir"], str)
            or type(entry["n"]) is not int or entry["n"] < 1):
        raise ValueError(f"{ENV} entry for {base_url} must be {{\"dir\": <path>, \"n\": <positive integer>}}")
    return entry


class SlotTransport:
    """transport(url, body, timeout_s) that holds one of n shared slots while the wrapped transport runs."""

    def __init__(self, inner: Callable, directory: Path, n: int, clock: Callable[[], float] = time.monotonic):
        if type(n) is not int or n < 1:
            raise ValueError("n must be a positive integer")
        self.inner, self.dir, self.n, self._clock = inner, Path(directory), n, clock
        self.dir.mkdir(parents=True, exist_ok=True)
        # The checked-configuration binding in models.py reads these from the transport.
        self.origin = getattr(inner, "origin", None)
        self.max_response_bytes = getattr(inner, "max_response_bytes", None)
        # A shared directory denotes one immutable quota and server. Otherwise a process using n+1 silently defeats n.
        with (self.dir / "config.lock").open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            expected = {"n": n, "origin": self.origin}
            config = self.dir / "config.json"
            if config.exists():
                if json.loads(config.read_text()) != expected:
                    raise ValueError("shared slot configuration differs")
            else:
                with config.open("x") as out:
                    json.dump(expected, out, sort_keys=True)
                    out.flush()
                    os.fsync(out.fileno())
            fcntl.flock(f, fcntl.LOCK_UN)

    @contextmanager
    def _slot(self, body: dict):
        began_wall, began = time.time(), self._clock()
        identity = {"wait_id": uuid.uuid4().hex, "pid": os.getpid(), "n": self.n,
                    "arm": os.environ.get("TELOHELM_ARM"), "stream": os.environ.get("TELOHELM_STREAM"),
                    "origin": self.origin, "began": began_wall, "began_monotonic_s": began,
                    "body_sha256": hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()}
        self._log({**identity, "event": "queued"})
        while True:
            for k in range(self.n):
                f = (self.dir / f"slot-{k}.lock").open("a")
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    f.close()
                    continue
                try:
                    acquired = self._clock()
                    self._log({**identity, "event": "acquired", "slot": k, "wait_s": acquired - began,
                               "at": time.time(), "monotonic_s": acquired})
                    outcome = "returned"
                    try:
                        yield k
                    except BaseException as exc:
                        outcome = type(exc).__name__
                        raise
                    finally:
                        ended = self._clock()
                        self._log({**identity, "event": "released", "slot": k, "at": time.time(),
                                   "monotonic_s": ended, "held_s": ended - acquired, "outcome": outcome})
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
                    f.close()
                return
            time.sleep(POLL_S)

    def _log(self, row: dict) -> None:
        with (self.dir / "waits.jsonl").open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f, fcntl.LOCK_UN)

    def __call__(self, url: str, body: dict, timeout_s: float) -> tuple[int, Any]:
        with self._slot(body):
            return self.inner(url, body, timeout_s)

    def exchange(self, url: str, body: dict, timeout_s: float) -> dict:
        """Preserve the underlying pre-parse raw exchange API, subject to the same quota."""
        with self._slot(body):
            return self.inner.exchange(url, body, timeout_s)


def wrap(transport: Callable, base_url: str) -> Callable:
    """The transport, wrapped if this job shares the server at base_url."""
    entry = configured(base_url)
    return transport if entry is None else SlotTransport(transport, Path(entry["dir"]), entry["n"])
