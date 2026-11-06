"""The goal monitor's store of right and wrong cases (note B8, decision D29, protocol 0.8 section 6).

Claude's half of the D29 split: the typed case record, its storage, the audit of every acquisition attempt, the
per-stream schedule with its seeds and caps, and the routing of cases to a check's replay test. Codex's half
supplies the labels: a case is admitted only after a trusted label verifier, passed in by the monitor, accepts its
label evidence. A caller's boolean, a fingerprint or a field in the case never authorises admission by itself.

Each case belongs to one stream, one split and one decision. The splits are:
- stored_wrong: 200 per stream, 40 of each of the 5 kinds, across both decisions;
- stored_right: 100 per stream for each live decision, heading and report (D29);
- fresh_wrong: 100 per rewrite, keyed by the stream and the rewrite's ordinal, never reused;
- confirmed_wrong: wrong answers confirmed during live episodes, added as they are found;
- eligibility: the separate pool for judging planted weakenings, never used for replay or adoption;
- final: 200 per stream, held back from every adoption decision for the final evaluation.

A check's replay test sees only cases of its own decision: stored and confirmed wrong cases, its rewrite's fresh
wrong cases, and the 100 stored right cases. An unrelated or skipped case never counts as accepted. A decision with
no settled wrong case, fewer than 100 right cases, or no fresh cases for the rewrite cannot pass; the reason is
returned. The eligibility and final splits are never handed to a replay.

Storage follows the library's rules: one append-only log of framed records (length, SHA-256, body), each flushed to
disk before it counts; a torn last record is removed by recover(); a damaged record stops further writing. Every
acquisition attempt, admitted or not, is logged with its outcome, reason and cost, so excluding an unknown or failed
draw from a pool never removes it from the denominator.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

from ..taskpacks.tanker.public import TANKER

SCHEMA = 1
SPLITS = ("stored_wrong", "stored_right", "fresh_wrong", "confirmed_wrong", "eligibility", "final")
DECISIONS = TANKER.decisions                           # from the task's public contract
KINDS = ("geometry", "identity", "stale", "goal_change", "combined")      # the 5 kinds of note B8, in its order
WRONG_SPLITS = ("stored_wrong", "fresh_wrong", "confirmed_wrong", "final")
DISPOSITIONS = ("right", "wrong", "unknown")
OUTCOMES = ("admitted", "unknown", "error", "cap", "refused")

STORED_WRONG = 200
PER_KIND = 40
RIGHT_PER_DECISION = 100
FRESH_PER_REWRITE = 100
FINAL = 200
ELIGIBILITY_PER_DECISION = 100              # proposed in protocol 0.9, for Codex's review
ATTEMPTS_PER_CASE = 20                      # draw cap per required case; revised only after the yield diagnostic
SCORED_STREAMS = (4301, 4302, 4303, 4304)
# A development store (Simon, 14 October) holds cases for development streams only, labelled under a development
# rule (dev.*), such as B8's named alternative 'wrong relative to the reference pilot'. A scored store refuses both, and
# a development store refuses scored streams and any other rule, so the two can never mix.
DEVELOPMENT_STREAMS = tuple(range(4400, 4500))
DEVELOPMENT_RULE_PREFIX = "dev."
MARKER = {"record": "store", "development": True}

Verifier = Callable[[dict[str, Any]], bool]

# Codex's conditional label arithmetic (study2/monitor_labels.py, D30) returns proposals that are unverified and
# inadmissible by design. Until an independently verified numerical producer and its admission path exist, a label
# that carries a proposal, or the proposal object itself, is refused here, whatever a verifier says.
UNVERIFIED_RULES = ("B8.heading.action.v1",)


class StoreError(RuntimeError):
    pass


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def base_seed(stream: int, split: str) -> int:
    """Monitor data uses 5001 to 5004 and eligibility 7001 to 7004, one per scored stream (protocol section 4)."""
    if stream not in SCORED_STREAMS:
        raise StoreError(f"stream {stream} is not a scored stream")
    offset = SCORED_STREAMS.index(stream) + 1
    return (7000 if split == "eligibility" else 5000) + offset


def draw_seed(stream: int, split: str, decision: str, kind: str | None, ordinal: int, attempt: int,
              rewrite: int | None = None) -> int:
    """Each attempt's seed: the first 8 bytes of SHA-256 of its full identity, read big-endian, as streams.py does.
    Fresh cases are keyed by the rewrite's ordinal, never by which candidate it produced."""
    text = f"telohelm-b8|{base_seed(stream, split)}|{split}|{decision}|{kind}|{rewrite}|{ordinal}|{attempt}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def schedule(stream: int) -> list[dict[str, Any]]:
    """The fixed acquisition schedule for one stream, before any draw: every required case with its cap. Stored
    wrong and final cases are stream-wide quotas by kind (40 each); which decision each falls to is whatever the
    kind's generator yields, and it is reported, not fixed. Fresh cases are drawn per rewrite, not scheduled here."""
    rows = []
    for kind in KINDS:
        for i in range(PER_KIND):
            rows.append({"split": "stored_wrong", "decision": None, "kind": kind, "ordinal": i})
    for decision in DECISIONS:
        for i in range(RIGHT_PER_DECISION):
            rows.append({"split": "stored_right", "decision": decision, "kind": None, "ordinal": i})
        for i in range(ELIGIBILITY_PER_DECISION):
            rows.append({"split": "eligibility", "decision": decision, "kind": None, "ordinal": i})
    for i in range(FINAL):
        rows.append({"split": "final", "decision": None, "kind": KINDS[i % len(KINDS)], "ordinal": i})
    for r in rows:
        r.update({"stream": stream, "cap": ATTEMPTS_PER_CASE})
    return rows


def _output_errors(decision: str, output: Any) -> list[str]:
    if not isinstance(output, dict):
        return ["output is not an object"]
    if decision == "heading":
        h = output.get("heading_deg")
        if set(output) != {"heading_deg"} or isinstance(h, bool) or not isinstance(h, (int, float)) \
                or not 0 <= h < 360:
            return ["a heading output is exactly {'heading_deg': number in [0, 360)}"]
        return []
    if not output:
        return []
    if set(output) == {"report"} and isinstance(output["report"], str):
        return []
    if set(output) == {"report_trigger_nm"}:
        v = output["report_trigger_nm"]
        if v is None or (not isinstance(v, bool) and isinstance(v, (int, float)) and 0 < v <= 1000):
            return []
    return ["a report output is {}, {'report': text} or {'report_trigger_nm': number in (0, 1000] or null}"]


def _unverified(value: Any) -> str | None:
    """Why a label, or anything inside it, is an unverified proposal; None if nothing marks it so."""
    if isinstance(value, dict):
        if value.get("admissible") is False:
            return "it is marked inadmissible"
        q = value.get("qualification")
        if isinstance(q, str) and q.startswith(("unverified", "conditional")):   # monitor_labels, contact_bounds
            return f"its qualification is {q!r}"
        if value.get("rule") in UNVERIFIED_RULES:
            return f"its rule {value['rule']!r} only proposes labels"
        for v in value.values():
            why = _unverified(v)
            if why:
                return why
    elif isinstance(value, (list, tuple)):
        for v in value:
            why = _unverified(v)
            if why:
                return why
    return None


def case_errors(case: dict[str, Any], development: bool = False) -> list[str]:
    """Structural checks before any label verifier or check runs. Label truth is the verifier's job, but a label
    that is only a proposal is refused here first (D30)."""
    if not isinstance(case, dict):
        return [f"a case is a typed record, not {type(case).__name__}; a label proposal is not a case"]
    errors = []
    need = ("schema", "stream", "split", "decision", "kind", "interface", "goal", "context", "output", "origin",
            "label")
    missing = [k for k in need if k not in case]
    if missing:
        return [f"missing {missing}"]
    if case["schema"] != SCHEMA:
        errors.append(f"schema {case['schema']!r}")
    if case["stream"] not in (DEVELOPMENT_STREAMS if development else SCORED_STREAMS):
        errors.append(f"stream {case['stream']!r} is not a {'development' if development else 'scored'} stream")
    if case["split"] not in SPLITS:
        errors.append(f"split {case['split']!r}")
    if case["decision"] not in DECISIONS:
        errors.append(f"decision {case['decision']!r}")
    if case["interface"] != "action":
        errors.append("only the one-second action interface is stored; sequences need their own review")
    label = case["label"]
    if not isinstance(label, dict) or label.get("disposition") not in DISPOSITIONS \
            or not isinstance(label.get("reason"), str) or not label.get("rule") or not label.get("evidence"):
        errors.append("label needs a disposition, a reason, a rule and non-empty evidence")
    elif label["disposition"] == "unknown":
        errors.append("an unknown case is audited, never stored")
    elif str(label.get("rule")).startswith(DEVELOPMENT_RULE_PREFIX) != development:
        errors.append(f"rule {label.get('rule')!r} does not belong in a {'development' if development else 'scored'} "
                      "store")
    elif _unverified(label):
        errors.append(f"the label is an unverified proposal ({_unverified(label)}); proposals are never admitted "
                      "until a verified producer and admission path exist")
    else:
        if case["split"] in WRONG_SPLITS and label["disposition"] != "wrong":
            errors.append(f"a {case['split']} case must be labelled wrong")
        if case["split"] == "stored_right" and label["disposition"] != "right":
            errors.append("a stored_right case must be labelled right")
    if case["split"] in WRONG_SPLITS and case["kind"] not in KINDS:
        errors.append(f"a wrong case needs one of the kinds {KINDS}")
    if case["split"] == "stored_right" and case["kind"] is not None:
        errors.append("a right case has no kind")
    goal = case["goal"]
    if not isinstance(goal, dict) or "version" not in goal:
        errors.append("the goal is the complete original goal record, with its version")
    ctx = case["context"]
    if not isinstance(ctx, dict):
        errors.append("the context is the exact payload the live check receives")
    elif isinstance(goal, dict) and "goal_version" in ctx and ctx["goal_version"] != goal.get("version"):
        errors.append("the context's goal version differs from the case's original goal")
    origin = case["origin"]
    if not isinstance(origin, dict) or not all(k in origin for k in ("episode", "second", "generator", "seed",
                                                                       "ordinal", "attempt")):
        errors.append("origin needs episode, second, generator, seed, ordinal and attempt")
    if case["split"] == "fresh_wrong" and not (isinstance(origin, dict) and isinstance(origin.get("rewrite"), int)):
        errors.append("a fresh case records the rewrite ordinal it was drawn for")
    if case["decision"] in DECISIONS:
        errors += _output_errors(case["decision"], case["output"])
    return errors


def _marker_line() -> bytes:
    body = canonical(MARKER)
    return b"%d %s %s\n" % (len(body), hashlib.sha256(body).hexdigest().encode(), body)


def snapshot(case: Any) -> dict[str, Any]:
    """A detached copy decoded from the case's canonical bytes; a case JSON cannot encode is refused here, before any
    caller acts on it (CONFIRM159.JSON)."""
    if not isinstance(case, dict):
        raise StoreError(f"case refused: {case_errors(case)}")
    try:
        return json.loads(canonical(case))
    except (TypeError, ValueError) as e:
        raise StoreError(f"case refused: not a JSON record ({e})") from None


def case_id(case: dict[str, Any]) -> str:
    return hashlib.sha256(canonical({k: v for k, v in case.items() if k != "cost"})).hexdigest()


def typed_key(case: dict[str, Any]) -> tuple:
    o = case["origin"]
    return (case["stream"], case["split"], case["decision"], case["kind"], o.get("rewrite"), o["generator"],
            o["seed"], o["ordinal"])


class CaseStore:
    def __init__(self, root: Path, verifier: Verifier | None = None, verifier_name: str | None = None,
                 development: bool = False):
        self.development = development
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.root / "cases.log"
        if not self.log_path.exists():
            self.log_path.touch()
        self.verifier, self.verifier_name = verifier, verifier_name
        self.records = self._read()
        # A development store marks its log on creation; a log is opened only in the mode it was made in. A
        # development store may mark only a log that holds no case or attempt record. If the log ends in a torn record
        # (such as a torn first marker), the mark is written by recover(), after the cut and before any usable record
        # can be written (DEV161.MARKER), so no development case is ever stored in an unmarked log.
        if development and not self._marked():
            if any(r.get("record") in ("case", "attempt") for r in self.records):
                raise StoreError("this case log was not made as a development store")
            if not self._torn and self._damaged_at is None:
                self._append(MARKER)
        elif not development and self._marked():
            raise StoreError("this case log is a development store; a scored store cannot open it")
        elif not development and self._torn and not self.records and _marker_line().startswith(self._torn_bytes()):
            raise StoreError("this case log begins with a torn development store marker; a scored store cannot "
                             "open it")

    def _marked(self) -> bool:
        return any(r.get("record") == "store" and r.get("development") is True for r in self.records)

    def _torn_bytes(self) -> bytes:
        return self.log_path.read_bytes()[-self._torn:] if self._torn else b""

    # -- the log -------------------------------------------------------------------------------------------------
    def _read(self) -> list[dict[str, Any]]:
        out, data, pos = [], self.log_path.read_bytes(), 0
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
                out.append(json.loads(body))
            except ValueError:
                self._damaged_at = pos
                break
            pos = end + 1
        return out

    def _append(self, record: dict[str, Any]) -> None:
        if self.development and record.get("record") in ("case", "attempt") and not self._marked():
            raise StoreError("a development store writes no case or attempt before its log is marked")
        if self._damaged_at is not None:
            raise StoreError("the case log is damaged; nothing more is written until it is examined")
        if self._torn:
            raise StoreError("the case log ends in a torn record; run recover() first")
        body = canonical(record)
        line = b"%d %s %s\n" % (len(body), hashlib.sha256(body).hexdigest().encode(), body)
        with self.log_path.open("ab") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        # What is kept in memory is decoded from the bytes just written, so no caller holds a reference to it and
        # it is the same value a restart reads (B8.STORE.IMMUTABLE).
        self.records.append(json.loads(body))

    def recover(self) -> int:
        if self._damaged_at is not None:
            raise StoreError(f"case log record at byte {self._damaged_at} is damaged; recovery will not cut it")
        removed = self._torn
        if removed:
            data = self.log_path.read_bytes()
            with self.log_path.open("r+b") as f:
                f.truncate(len(data) - removed)
                f.flush()
                os.fsync(f.fileno())
            self._torn = 0
            self._append({"record": "recovery", "removed_bytes": removed})
        if self.development and not self._marked():
            if any(r.get("record") in ("case", "attempt") for r in self.records):
                raise StoreError("this case log was not made as a development store")
            self._append(MARKER)              # the mode is restored before any usable record (DEV161.MARKER)
        return removed

    # -- admission and audit -------------------------------------------------------------------------------------
    def attempt(self, row: dict[str, Any], outcome: str, reason: str, cost: dict[str, Any] | None = None,
                case: str | None = None) -> None:
        """Log one acquisition attempt whatever its outcome, with its cost (Codex's B8 denominators)."""
        if outcome not in OUTCOMES:
            raise StoreError(f"unknown outcome {outcome!r}")
        self._append({"record": "attempt", "row": row, "outcome": outcome, "reason": reason, "cost": cost or {},
                      "case": case})            # _append keeps a copy decoded from the written bytes

    def admit(self, case: dict[str, Any]) -> str:
        """Store a labelled case after its structure and the trusted verifier both accept it. The same case again
        is a no-op; a different case under the same typed identity is refused."""
        # Everything below works on one detached snapshot, decoded from the canonical bytes that will be written:
        # the value checked, verified, identified and stored is the same, and later changes to the caller's object
        # cannot reach it (B8.STORE.IMMUTABLE).
        snap = snapshot(case)
        body = canonical(snap)
        errors = case_errors(snap, self.development)
        if errors:
            raise StoreError(f"case refused: {errors}")
        if self.verifier is None:
            raise StoreError("no trusted label verifier; nothing can be admitted")
        cid = case_id(snap)
        for r in self.records:
            if r.get("record") != "case":
                continue
            if r["case_id"] == cid:
                return cid
            if typed_key(r["case"]) == typed_key(snap):
                raise StoreError(f"conflicting case for identity {typed_key(snap)}")
        if self.verifier(json.loads(body)) is not True:
            raise StoreError("the label verifier did not accept the case's evidence")
        self._append({"record": "case", "case_id": cid, "verifier": self.verifier_name, "case": snap})
        return cid

    # -- reading -------------------------------------------------------------------------------------------------
    def cases(self, stream: int | None = None, split: str | None = None, decision: str | None = None,
              rewrite: int | None = None) -> list[dict[str, Any]]:
        out = []
        for r in self.records:
            if r.get("record") != "case":
                continue
            c = r["case"]
            if stream is not None and c["stream"] != stream:
                continue
            if split is not None and c["split"] != split:
                continue
            if decision is not None and c["decision"] != decision:
                continue
            if rewrite is not None and c["origin"].get("rewrite") != rewrite:
                continue
            out.append(copy.deepcopy(c))     # a reader's changes never reach the stored original
        return out

    def coverage(self, stream: int) -> dict[str, Any]:
        """Admitted counts by split, decision and kind, and attempt counts by outcome, for the report."""
        counts: dict[str, int] = {}
        for c in self.cases(stream):
            key = f"{c['split']}|{c['decision']}|{c['kind']}"
            counts[key] = counts.get(key, 0) + 1
        attempts: dict[str, int] = {}
        for r in self.records:
            if r.get("record") == "attempt" and r["row"].get("stream") == stream:
                attempts[r["outcome"]] = attempts.get(r["outcome"], 0) + 1
        return {"admitted": counts, "attempts": attempts}

    def replay_cases(self, stream: int, decision: str, rewrite: int | None
                     ) -> tuple[list[dict], list[dict], list[str]]:
        """The wrong and right cases a check of this decision is replayed on, as the exact payloads the sandbox
        receives, and the reasons the test cannot run (empty when it can)."""
        if decision not in DECISIONS:
            return [], [], [f"decision {decision!r} has no stored cases"]
        stored = self.cases(stream, "stored_wrong", decision) + self.cases(stream, "confirmed_wrong", decision)
        fresh = self.cases(stream, "fresh_wrong", decision, rewrite) if rewrite is not None else []
        right = self.cases(stream, "stored_right", decision)
        reasons = []
        if not stored:
            reasons.append(f"no settled wrong case for {decision}")
        if len(right) != RIGHT_PER_DECISION:
            reasons.append(f"{len(right)} right cases for {decision}, not {RIGHT_PER_DECISION}")
        if rewrite is not None and len(fresh) != FRESH_PER_REWRITE:
            reasons.append(f"{len(fresh)} fresh wrong cases for rewrite {rewrite}, not {FRESH_PER_REWRITE}")
        payload = lambda c: {"context": c["context"], "output": c["output"]}
        return [payload(c) for c in stored + fresh], [payload(c) for c in right], reasons

    def fresh_ordinals(self, stream: int, decision: str) -> list[int]:
        """The rewrite ordinals that hold fresh wrong cases for this decision, in order: the fresh sets a run may use."""
        return sorted({c["origin"].get("rewrite") for c in self.cases(stream, "fresh_wrong", decision)
                       if isinstance(c["origin"].get("rewrite"), int)})

    def fresh_set_digest(self, stream: int, decision: str, rewrite: int) -> str:
        """The identity of one fresh set: SHA-256 over its cases' identities in order, so an allocation can be checked
        against the store it was made from (Codex's FRESH213.NAMESPACE)."""
        ids = sorted(r["case_id"] for r in self.records if r.get("record") == "case"
                     and r["case"]["stream"] == stream and r["case"]["split"] == "fresh_wrong"
                     and r["case"]["decision"] == decision and r["case"]["origin"].get("rewrite") == rewrite)
        return hashlib.sha256(json.dumps(ids).encode()).hexdigest()

    def undo_cases(self, stream: int, decision: str) -> tuple[list[dict], list[dict]]:
        """Cases for re-checking an adopted check after a confirmed wrong answer: stored and confirmed wrong cases
        and the right cases of its decision; fresh cases belong to their own rewrite only."""
        stored = self.cases(stream, "stored_wrong", decision) + self.cases(stream, "confirmed_wrong", decision)
        right = self.cases(stream, "stored_right", decision)
        payload = lambda c: {"context": c["context"], "output": c["output"]}
        return [payload(c) for c in stored], [payload(c) for c in right]
