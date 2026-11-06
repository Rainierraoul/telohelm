"""Telohelm's procedure library: immutable entries, an append-only event log and a projected index
(architecture section 7).

Every change follows one order: the entry's content is written first, by a temporary file and a rename;
then the event that refers to it is appended to the log and flushed to disk; only then is the index
updated. The index is a projection of the log, rebuilt from it after a crash. Each log record carries its
length and SHA-256, so a record cut short is recognised, removed, and the removal itself is logged.
Statuses are never stored in entries, so the library can be projected to any earlier point.

One process writes a library. This module does not lock against a second writer.

The default task keeps its historical implicit schema. Other tasks record their candidate schema in contract.json
before entries can be added; their add events carry its fingerprint. They must reopen with the same schema.

Status changes that must happen together (a pair adopted, a check retired with the procedure that depends on it)
are written as a group: an opening record that lists every planned change, the changes themselves, then a closing
record. Until the closing record is on disk the projection ignores the group's changes, so a reader sees the state
from before the group and never half of it (Codex's PAIR153.VISIBLE). complete_open_groups() carries an
interrupted group through from its plan, once (PAIR153.UNDO.CUT); a group cannot be opened while another is open.

What the tests show is recovery from a process stopped at any of the three points. Power loss is a different
model: files and the folder entries for renames are synced to disk, but whether a given disk and file system
honour those syncs is not tested here (Codex's R22.9).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ..contracts import PublicContract
from ..taskpacks.tanker.public import TANKER

SCHEMA_VERSION = 1
STATUSES = ("candidate", "adopted", "suspended", "retired")


class LibraryError(RuntimeError):
    pass


class Crash(Exception):
    """Raised by the test hook to stop the writer at a chosen point."""


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)               # make the rename itself durable, not only the file's bytes (R22.9)


def _contract_record(pack: PublicContract) -> dict:
    """The task identity and candidate schema; model instructions are recorded separately."""
    return {"schema": 1, "pack_id": pack.pack_id, "schema_version": pack.schema_version,
            "decisions": list(pack.decisions), "pair_decisions": list(pack.pair_decisions),
            "context_fields": sorted(pack.context_fields), "output_fields": sorted(pack.output_fields),
            "decision_fields": {d: sorted(fields) for d, fields in pack.decision_fields.items()}}


class Library:
    def __init__(self, root: Path, crash_at: str | None = None, *, pack: PublicContract = TANKER):
        self.root = Path(root)
        self.content = self.root / "content"
        self.log_path = self.root / "events.log"
        self.index_path = self.root / "index.json"
        self.contract_path = self.root / "contract.json"
        self.bind_contract(pack)
        self.content.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self.log_path.touch()
            _fsync_dir(self.root)
        self.crash_at = crash_at          # test hook: 'after_content', 'mid_event' or 'after_event'
        self.events = self._read_log()
        if any(e.get("contract_sha256") != self._contract_sha
               for e in self.events if e["kind"] == "add"):
            raise LibraryError("library entries do not match the selected task contract")
        self.recovered: dict[str, Any] | None = None

    def bind_contract(self, pack: PublicContract) -> None:
        """Bind an empty library, or confirm its existing task. Unmarked stores are legacy tanker stores.

        Binding is host configuration, not learned knowledge or an operating-system security boundary. A stored
        binding never changes. Legacy tanker bytes stay unchanged; a different schema needs a fresh library.
        """
        wanted = _contract_record(pack)
        if self.contract_path.exists():
            try:
                saved = json.loads(self.contract_path.read_bytes())
                matches = canonical(saved) == canonical(wanted)
            except (ValueError, UnicodeError) as exc:
                raise LibraryError("the library contract is damaged") from exc
            if not matches:
                raise LibraryError("the library contract does not match the selected task")
        elif wanted != _contract_record(TANKER):
            if ((self.log_path.exists() and self.log_path.stat().st_size)
                    or (self.content.exists() and any(self.content.iterdir()))):
                raise LibraryError("a populated legacy library cannot be rebound to another task")
            self.root.mkdir(parents=True, exist_ok=True)
            _write_atomic(self.contract_path, canonical(wanted))
        self.pack = pack
        self._contract_sha = (hashlib.sha256(canonical(wanted)).hexdigest()
                              if wanted != _contract_record(TANKER) else None)

    # -- the log -------------------------------------------------------------------------------
    def _read_log(self) -> list[dict[str, Any]]:
        """Read every whole record. An unfinished last line (no newline) is a torn write, removable by
        recover(). A finished line that fails its length or fingerprint is damage, which stops recovery."""
        events = []
        data = self.log_path.read_bytes()
        pos = 0
        self._torn, self._damaged_at = 0, None
        while pos < len(data):
            end = data.find(b"\n", pos)
            if end < 0:
                self._torn = len(data) - pos
                break
            try:
                length, digest, body = data[pos:end].split(b" ", 2)
                if len(body) != int(length) or hashlib.sha256(body).hexdigest().encode() != digest:
                    raise ValueError("fingerprint")
                events.append(json.loads(body))
            except ValueError:
                self._damaged_at = pos
                break
            pos = end + 1
        return events

    def _append(self, event: dict[str, Any]) -> None:
        if self._damaged_at is not None:
            raise LibraryError("the log is damaged; nothing more is written until it is examined")
        if self._torn and event["kind"] != "recovery":
            raise LibraryError("the log ends in a torn record; run recover() first")
        body = canonical(event)
        record = b"%d %s %s\n" % (len(body), hashlib.sha256(body).hexdigest().encode(), body)
        with self.log_path.open("ab") as f:
            if self.crash_at == "mid_event":
                f.write(record[: len(record) // 2])
                f.flush()
                raise Crash("mid_event")
            f.write(record)
            f.flush()
            os.fsync(f.fileno())
        self.events.append(event)

    def _event(self, kind: str, **fields: Any) -> dict[str, Any]:
        return {"schema": SCHEMA_VERSION, "seq": len(self.events) + 1, "kind": kind, **fields}

    # -- changes -------------------------------------------------------------------------------
    def add(self, entry: dict[str, Any], episode: str | None = None) -> str:
        """Store a new entry as a candidate and return its fingerprint."""
        for key in ("kind", "decision", "conditions", "source", "origin"):
            if key not in entry:
                raise LibraryError(f"entry lacks {key}")
        # The library applies the reply checks itself, so no caller can store an entry that bypasses them (R22.7).
        from .procedures import reply_errors
        errors = reply_errors({"kind": entry["kind"], "decision": entry["decision"], "conditions": entry["conditions"],
                               "source": entry["source"], "reason": {"code": "missing_tool", "text": ""}}, pack=self.pack)
        if errors:
            raise LibraryError(f"entry refused: {errors}")
        body = canonical({"schema": SCHEMA_VERSION, **entry})
        sha = hashlib.sha256(body).hexdigest()
        path = self.content / f"{sha}.json"
        if not path.exists():
            _write_atomic(path, body)
        if self.crash_at == "after_content":
            raise Crash("after_content")
        binding = {"contract_sha256": self._contract_sha} if self._contract_sha is not None else {}
        self._append(self._event("add", entry=sha, kind_of_entry=entry["kind"], decision=entry["decision"],
                                 episode=episode, **binding))
        self._after_event()
        return sha

    def set_status(self, sha: str, status: str, reason: str, episode: str | None = None,
                   group: int | None = None) -> None:
        if status not in STATUSES:
            raise LibraryError(f"unknown status {status!r}")
        if sha not in self.projection()["entries"]:
            raise LibraryError(f"no entry {sha}")
        fields: dict[str, Any] = {"entry": sha, "status": status, "reason": reason, "episode": episode}
        if group is not None:
            if group not in self.open_groups():
                raise LibraryError(f"group {group} is not open")
            fields["group"] = group
        self._append(self._event("status", **fields))
        self._after_event()

    # -- groups of changes that become visible together -----------------------------------------
    def open_group(self, plan: list[dict[str, Any]], reason: str, episode: str | None = None) -> int:
        """Record every planned change ({entry, status, reason}) before making any; return the group's number."""
        if self.open_groups():
            raise LibraryError("an earlier group is still open; complete it first")
        entries = self.projection()["entries"]
        for change in plan:
            if change["entry"] not in entries or change["status"] not in STATUSES:
                raise LibraryError(f"bad planned change {change!r}")
        event = self._event("group_open", plan=[{k: change[k] for k in ("entry", "status", "reason")}
                                                for change in plan], reason=reason, episode=episode)
        self._append(event)
        self._after_event()
        return event["seq"]

    def close_group(self, group: int) -> None:
        if group not in self.open_groups():
            raise LibraryError(f"group {group} is not open")
        self._append(self._event("group_close", group=group))
        self._after_event()

    def open_groups(self) -> dict[int, dict[str, Any]]:
        opened = {e["seq"]: e for e in self.events if e["kind"] == "group_open"}
        for e in self.events:
            if e["kind"] == "group_close":
                opened.pop(e["group"], None)
        return opened

    def complete_open_groups(self) -> list[int]:
        """Finish every interrupted group from its recorded plan and close it. A change already made is made again
        harmlessly; the plan, written before any change, is the only source, so nothing is re-derived."""
        done = []
        for group, event in sorted(self.open_groups().items()):
            for change in event["plan"]:
                self.set_status(change["entry"], change["status"], change["reason"], episode=event.get("episode"),
                                group=group)
            self.close_group(group)
            done.append(group)
        return done

    def _after_event(self) -> None:
        if self.crash_at == "after_event":
            raise Crash("after_event")
        _write_atomic(self.index_path, canonical(self.projection()))

    # -- reading -------------------------------------------------------------------------------
    def projection(self, upto_seq: int | None = None) -> dict[str, Any]:
        """Statuses as they stood after event upto_seq (all events if None)."""
        entries: dict[str, dict[str, Any]] = {}
        last = 0
        closed = {e["group"] for e in self.events
                  if e["kind"] == "group_close" and (upto_seq is None or e["seq"] <= upto_seq)}
        for e in self.events:
            if upto_seq is not None and e["seq"] > upto_seq:
                break
            last = e["seq"]
            if e["kind"] == "add":
                entries[e["entry"]] = {"kind": e["kind_of_entry"], "decision": e["decision"], "status": "candidate"}
            elif e["kind"] == "status" and ("group" not in e or e["group"] in closed):
                entries[e["entry"]]["status"] = e["status"]
        return {"schema": SCHEMA_VERSION, "seq": last, "entries": entries}

    def entry(self, sha: str) -> dict[str, Any]:
        body = (self.content / f"{sha}.json").read_bytes()
        if hashlib.sha256(body).hexdigest() != sha:
            raise LibraryError(f"content of {sha} does not match its fingerprint")
        entry = json.loads(body)
        if self.contract_path.exists():
            from .procedures import reply_errors
            errors = reply_errors({k: entry.get(k) for k in ("kind", "decision", "conditions", "source")} |
                                  {"reason": {"code": "missing_tool", "text": ""}}, pack=self.pack)
            if errors:
                raise LibraryError(f"stored entry refused by its task contract: {errors}")
        return entry

    def adopted(self, decision: str, kind: str) -> list[str]:
        p = self.projection()["entries"]
        return [s for s, v in p.items() if v["decision"] == decision and v["kind"] == kind and v["status"] == "adopted"]

    # -- recovery ------------------------------------------------------------------------------
    def recover(self) -> dict[str, Any]:
        """Remove a torn last record, log the removal, rebuild the index and list unreferenced content."""
        self.crash_at = None
        if self._damaged_at is not None:
            raise LibraryError(f"log record at byte {self._damaged_at} is damaged; recovery will not cut the log")
        removed = self._torn
        if removed:
            data = self.log_path.read_bytes()
            with self.log_path.open("r+b") as f:
                f.truncate(len(data) - removed)
                f.flush()
                os.fsync(f.fileno())
            self._torn = 0
            self._append(self._event("recovery", removed_bytes=removed))
        referenced = {e["entry"] for e in self.events if e["kind"] == "add"}
        stored = {p.stem for p in self.content.glob("*.json")}
        missing = sorted(referenced - stored)
        if missing:
            raise LibraryError(f"the log refers to content that is missing: {missing[:3]}")
        for sha in referenced:
            self.entry(sha)                   # each referenced entry must match its fingerprint
        _write_atomic(self.index_path, canonical(self.projection()))
        self.recovered = {"removed_bytes": removed, "unreferenced": sorted(stored - referenced)}
        return self.recovered
