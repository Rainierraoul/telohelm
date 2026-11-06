"""A chained log written by a trusted process, with its head copied to protected storage on every record
(architecture section 10).

Each record carries the SHA-256 of the record before it and a sequence number. The first record names the
run and is copied to the protected store before anything else, so no chain can be replaced from its start.
After each record the writer copies the head fingerprint and the record count to the protected store.
verify() checks a log file against that store, so any later change, cut or replacement of the log is found.

What this does not do: it cannot protect against a fault in the trusted writer itself, and it relies on
the sandbox (section 6) to keep Telohelm's code from writing either file. Those are tested elsewhere.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


class TrustedLog:
    def __init__(self, path: Path, protected: Path, run_id: str):
        self.path, self.protected = Path(path), Path(protected)
        if self.path.exists() or self.protected.exists():
            raise FileExistsError("a trusted log is written once; use a new path for a new run")
        self.count, self.head = 0, "0" * 64
        # Both files stay open for the run (MVP, TURN 113): reopening them for every record cost most of an episode's
        # time. Each record is still written, flushed and synced before the next.
        self._log_file = self.path.open("ab")
        self._heads_file = self.protected.open("ab")
        self._write({"kind": "anchor", "run_id": run_id})

    def _write(self, payload: dict[str, Any]) -> None:
        record = {"seq": self.count + 1, "prev": self.head, "payload": payload}
        body = _canonical(record)
        f = self._log_file
        f.write(body + b"\n")
        f.flush()
        os.fsync(f.fileno())
        self.count, self.head = self.count + 1, _digest(body)
        entry = {"count": self.count, "head": self.head}
        if self.count == 1:
            entry["anchor"] = self.head
        f = self._heads_file                              # append-only copy of every head, written in order
        f.write(_canonical(entry) + b"\n")
        f.flush()
        os.fsync(f.fileno())

    def close(self) -> None:
        for f in (self._log_file, self._heads_file):
            if not f.closed:
                f.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def append(self, payload: dict[str, Any]) -> None:
        self._write(payload)


def verify(path: Path, protected: Path) -> dict[str, Any]:
    """Check the log against its protected heads. Returns ok and, if not, the first problem found."""
    heads = [json.loads(line) for line in Path(protected).read_text().splitlines() if line]
    lines = Path(path).read_bytes().splitlines()
    if not heads:
        return {"ok": False, "problem": "no protected heads"}
    anchor = heads[0].get("anchor")
    prev = "0" * 64
    for i, line in enumerate(lines, start=1):
        record = json.loads(line)
        if record["seq"] != i or record["prev"] != prev:
            return {"ok": False, "problem": f"chain broken at record {i}"}
        prev = _digest(line)
        if i == 1 and prev != anchor:
            return {"ok": False, "problem": "first record does not match the protected anchor"}
        if i <= len(heads) and heads[i - 1]["head"] != prev:
            return {"ok": False, "problem": f"record {i} differs from its protected head"}
    if len(lines) != heads[-1]["count"]:
        return {"ok": False, "problem": f"log has {len(lines)} records, protected count is {heads[-1]['count']}"}
    return {"ok": True, "records": len(lines)}
