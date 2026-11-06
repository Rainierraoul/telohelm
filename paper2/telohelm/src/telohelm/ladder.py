"""The shared reasoning allowance and the automatic pause (B36, decision D26; paper2/notes/B36-FLIGHT-LADDER-PLAN.md).

ReasoningAllowance is the one authority for an original episode's reasoning requests, across flight and the rewrite
attempt that follows it. Each episode, keyed by (condition, stream, original episode), has at most 4 allocations; a
rewrite attempt takes at most min(3, what remains). Every allocation is written to an append-only file and synced
before it is used, so a restart or a fresh ledger reads back the spent room and cannot restore it. Allocating the
same request again returns its first allocation: recovery never adds one. A request refused before sending (over the
input cap, or uncountable) takes an allocation and costs nothing; a request with no room left takes none.

pause_due applies the protocol's pause (section 10, D26 item 7) to ledger rows: distinct started reasoning-provider
calls, flight and rewrite (each request counted once), anchored to the condition's first 20 scheduled positions;
timed-out and cut-off calls are bad; recovery rows and trial, replay, planting and setup calls are not counted;
unknown statuses are counted apart. It is due when 10 x bad > started. A condition-wide pause across streams, with a
shared authority and restart reconciliation, is not built yet.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

EPISODE_LIMIT = 4
ATTEMPT_LIMIT = 3
PAUSE_WINDOW = 20
BAD = ("timed_out", "cut_off")
GOOD = ("answered",)


class ReasoningAllowance:
    def __init__(self, path: Path, limit: int = EPISODE_LIMIT):
        self.path, self.limit = Path(path), limit
        self._rows: list[dict] = []
        if self.path.exists():
            self._rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    @staticmethod
    def key(condition: str, stream: str, episode: str) -> str:
        return json.dumps([condition, stream, episode])

    def allocations(self, key: str) -> list[dict]:
        return [r for r in self._rows if r["key"] == key]

    def remaining(self, key: str) -> int:
        return self.limit - len(self.allocations(key))

    def attempt_budget(self, key: str) -> int:
        return max(0, min(ATTEMPT_LIMIT, self.remaining(key)))

    def allocate(self, key: str, request: str, purpose: str, kind: str) -> int | None:
        """The ordinal (1 to 4) for this request, or None if no room is left. `kind` is 'provider' for a request
        that will be sent and 'preflight_refused' for one refused before sending."""
        for r in self.allocations(key):
            if r["request"] == request:
                if (r["purpose"], r["kind"]) != (purpose, kind):
                    # An allocation's identity is fixed: the same request cannot come back as another purpose or
                    # kind (Codex's ALLOW143.IDENTITY).
                    raise ValueError(f"request {request!r} was allocated as {r['purpose']}/{r['kind']}, "
                                     f"not {purpose}/{kind}")
                return r["ordinal"]
        if self.remaining(key) <= 0:
            return None
        row = {"key": key, "ordinal": len(self.allocations(key)) + 1, "request": request, "purpose": purpose,
               "kind": kind}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._rows.append(row)
        return row["ordinal"]


def pause_due(rows: Iterable[dict], schedule: list[str], window: int = PAUSE_WINDOW) -> dict[str, Any]:
    """The pause over ledger rows (dicts with kind, episode, status, recovers), for the first `window` scheduled
    episodes of a condition."""
    first = set(schedule[:window])
    # Distinct original provider requests only: flight and rewrite calls, each once by its request identity, never a
    # recovery row; trial, replay, planting and setup calls are not counted (Codex's PAUSE143.DISTINCT and PURPOSE).
    # Duplicates that disagree are refused, whichever comes first (PAUSE145.CONFLICT); a row with no purpose is listed
    # as unplaced, never guessed to be flying (PAUSE145.UNKNOWN.PURPOSE).
    started, seen, unplaced = [], {}, []
    for r in rows:
        if r.get("kind") != "reasoning_model" or r.get("recovers") is not None or r.get("episode") not in first:
            continue
        if "scope" not in r or r["scope"] is None:
            unplaced.append(r.get("request_id"))
            continue
        if r["scope"] not in ("running", "build"):
            continue
        rid = r.get("request_id")
        if rid is not None:
            fields = (r.get("status"), r.get("episode"), r.get("scope"), r.get("kind"))
            if rid in seen:
                if seen[rid] != fields:
                    raise ValueError(f"request {rid!r} appears twice with different records")
                continue
            seen[rid] = fields
        started.append(r)
    bad = sum(r.get("status") in BAD for r in started)
    unknown = sum(r.get("status") not in BAD + GOOD for r in started)
    n = len(started)
    return {"started": n, "bad": bad, "unknown_status": unknown, "unplaced_requests": unplaced,
            "fraction": (bad / n) if n else None, "due": bool(n) and 10 * bad > n}
