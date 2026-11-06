"""Protected learning monitor: episode records, qualification, adoption and recovery.

The proposal coordinator receives public episode summaries, source snapshots, verified public examples and bounded
verdicts. Scenarios, reference labels, case sets and library writes stay here. This is a responsibility boundary
inside the trusted process, not a new operating-system security boundary. Saved event names and formats are kept.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

from .qualification.store import CaseStore; from .qualification.store import StoreError; from .qualification.store import case_errors; from .qualification.store import snapshot
from .runtime import broker as B; from .runtime import episode_loop as L
from .runtime.library import Library; from .runtime.library import _write_atomic
from .runtime.ledger import Ledger
from .runtime.replay import replay_test
from .runtime.trusted_log import TrustedLog
from .taskpacks.tanker.world import episode as E
from .taskpacks.tanker.world.tanker_environment import component_trial  # compatibility export
from .runtime.environment import environment_for

FLOOR_WINDOW, FLOOR_RATE = 20, 0.95
RIGHT_CASES, MIN_RIGHT_CASES = 100, 95


@dataclass
class Completed:
    ordinal: int
    episode: str
    scenario: Any
    goal_met: bool
    failed: str | None
    solvable: bool | None         # the reference pilot's label, recorded for analysis only; no selection reads it
    helped: bool = False          # a help request was made; the floor counts only goals met without help
    alone: bool = False           # the goal was met with no model call and no help: the library did it by itself
    eligible: bool = True         # a fair trial case by Telohelm's own record (see Monitor.record)


@dataclass
class Corpus:
    """The monitor's stored cases for the check replay test. Kept on the trusted side; the learner never sees them."""
    wrong: list[dict] = field(default_factory=list)
    right: list[dict] = field(default_factory=list)


def _trial_ok(r: dict) -> bool:
    """One trial flight's verdict: its own decision's component trial, or, for a flight function that returns only a
    whole-goal result (the invented ones in tests), the goal met without help."""
    if "component" in r:
        return r["component"]["passed"]
    return bool(r["goal_met"]) and r["help"] == 0


def _regression_ok(r: dict) -> bool:
    """A regression episode's verdict: the whole goal met again without help."""
    # Real flights keep nominal success separate from numerical certainty and natural completion. A nominal
    # success must not certify retention when the saved outcome is unsettled or only a prefix (TURN221).
    if "outcome" in r:
        outcome = r["outcome"]
        if outcome.get("settled") is not True or outcome.get("end_reason") is None:
            return False
    return bool(r.get("goal_met")) and not r.get("help")


def _two_facts(r: dict) -> dict:
    """What the proposer may later be told about a replay: whether any wrong answer got through, and whether too few
    right answers, or fewer than the replaced check's, did. Booleans only."""
    return {"wrong_accepted_any": bool(r.get("wrong_accepted")),
            "right_short": r.get("right_accepted", 0) < MIN_RIGHT_CASES,
            "fewer_than_old": r.get("old_right_accepted") is not None
                              and r.get("right_accepted", 0) < r["old_right_accepted"]}


@dataclass(frozen=True)
class PublicEpisode:
    ordinal: int
    episode: str
    goal_met: bool
    failed: str | None
    helped: bool


@dataclass(frozen=True)
class Verdict:
    passed: bool
    candidate: str | None
    feedback: Any
    event_index: int

    def __post_init__(self):
        if self.feedback is not None:
            item = dict(self.feedback)
            if "why" in item:
                item["why"] = tuple(item["why"])
            object.__setattr__(self, "feedback", MappingProxyType(item))

class Monitor:
    def __init__(self, library, workdir, solvable, tool, corpus, *, pack, goal, flight_pack,
                 learn_decisions, trial_set, shadow_max_steps, fly, stream, regression,
                 floor_after_cold_start, no_replay):
        self.library, self.workdir = library, Path(workdir)
        self.solvable, self.tool, self.corpus = solvable, tool, corpus
        self.pack, self.goal, self.flight_pack = pack, goal, flight_pack
        self.library.bind_contract(pack)
        self.learn_decisions = tuple(learn_decisions)
        self.trial_set_size, self.shadow_max_steps = trial_set, shadow_max_steps
        self.fly, self.stream = fly or self._fly, stream
        self.regression, self.floor_after_cold_start, self.no_replay = regression, floor_after_cold_start, no_replay
        self.completed, self.events = [], []
        self._event_session = uuid.uuid4().hex
        self.armed, self.breaches = True, 0
        self._regression_set, self._attempt_trial = [], []

    def record(self, scenario, outcome, complete):
        if not complete:
            self._event("prefix_excluded", episode=scenario.episode_id)
            return None
        failed = None if outcome["goal_met"] else self.goal.failed(outcome)
        helped = bool(outcome.get("help_requests"))
        # The trial set, the retention set and the floor take only episodes that are a fair test by Telohelm's own
        # record: no accepted help request said the evidence was missing or stale. Such a request is Telohelm's own
        # judgement that its information ran out, and a procedure that sees the same information cannot be held to the
        # goal there. A request for want of a usable method stays in, because that
        # is the gap learning closes. Until 27 October the reference pilot's label chose these sets; it is still
        # recorded, but a deployed system could not compute it (paper2/notes/PRE-FREEZE-DESIGN-CHANGES-CLAUDE.md).
        # A flown episode's summary carries a count at its top level and the world's accepted requests, with their
        # reasons, in its nested outcome (Codex's TURN295 witness); a bare world outcome carries the list itself.
        nested = outcome.get("outcome")
        requests = nested.get("help_requests") if isinstance(nested, dict) else outcome.get("help_requests")
        eligible = not (isinstance(requests, (list, tuple)) and any(
            isinstance(h, dict) and h.get("reason") == "missing_or_stale_evidence" for h in requests))
        c = Completed(len(self.completed) + 1, scenario.episode_id, scenario, bool(outcome["goal_met"]), failed,
                      self.solvable(scenario), helped=helped,
                      alone=bool(outcome["goal_met"]) and outcome.get("settled") is True
                      and outcome.get("model_calls") == 0 and not helped, eligible=eligible)
        self.completed.append(c)
        self._event("completed", episode=c.episode, ordinal=c.ordinal, goal_met=c.goal_met, failed=failed,
                    solvable=c.solvable, helped=c.helped, alone=c.alone, eligible=c.eligible)
        return PublicEpisode(c.ordinal, c.episode, c.goal_met, c.failed, c.helped)

    def begin_attempt(self, n, boundary, reasons):
        self._attempt_trial = [c for c in self.completed if c.eligible][-self.trial_set_size:]
        self._regression_set = self.regression_set(self._attempt_trial)
        self._event("attempt_start", attempt=n, ordinal=boundary.ordinal, reasons=list(reasons),
                    trial_set=[c.episode for c in self._attempt_trial],
                    **({"regression_set": [c.episode for c in self._regression_set]} if self.regression else {}))
        return bool(self._attempt_trial)

    def episode_names(self):
        return tuple(c.episode for c in self.completed)

    def host_event(self, index):
        """Detailed output for the host's result log, never part of a proposal request."""
        return self.events[index]

    def _verdict(self, event, remember=True):
        return Verdict(bool(event.get("passed")), event.get("candidate"),
                       self.feedback(event) if remember else None, len(self.events) - 1)

    def examples(self, n, reasons):
        from . import examples as X
        if self.flight_pack.example_policy is None:
            raise ValueError("the selected task has no example policy")
        shown, receipts, omitted = X.select(self.workdir, self.completed, reasons, self.flight_pack.hidden, self.goal,
                                              context_builder=self.flight_pack.context_builder,
                                              policy=self.flight_pack.example_policy)
        self._event("examples_selected", attempt=n, selected=len(shown), omitted=len(omitted))
        with (self.workdir / "learning-examples.jsonl").open("a") as f:
            f.write(json.dumps({"attempt": n, "stage": "selected", "receipts": receipts, "omitted": omitted},
                               sort_keys=True) + "\n")
        return shown, receipts

    @staticmethod
    def feedback(event):
        """Project the existing feedback; detailed trial/replay data never crosses this interface."""
        n, kind, decision = event["attempt"], event["entry_kind"], event["decision"]
        base = {"attempt": n, "kind": kind, "decision": decision, "adopted": bool(event["passed"])}
        if event["passed"]:
            return base
        detail = event.get("detail") or {}
        check = detail.get("check") if kind == "pair" else (detail if kind == "check" else None)
        if check is not None and not (check.get("replay") or {}).get("passed", False):
            r = check.get("replay") or {}
            why = []
            if "refused" in check:
                why.append("the monitor could not run its replay test")
            if r.get("wrong_accepted_any"):
                why.append("the check accepted at least one stored wrong answer")
            if r.get("right_short"):
                why.append(f"the check accepted fewer than {MIN_RIGHT_CASES} of the {RIGHT_CASES} right answers")
            if r.get("fewer_than_old"):
                why.append("the check accepted fewer right answers than the check it would replace")
            return {**base, "stage": "replay_test", "why": why or ["the check failed the replay test"]}
        trial = ((detail.get("procedure") if kind == "pair" else detail) or {}).get("trial") or []
        failed = [t["episode"] for t in trial if not ((t.get("component") or {}).get("passed")
                                                      if "component" in t else t.get("goal_met"))]
        why = [f"the procedure did not do its part of the goal in trial episode {failed[0]}"] if failed else \
            ["the procedure failed its trial"]
        return {**base, "stage": "trial", "why": why}


    def _event(self, kind: str, **fields: Any) -> dict:
        e = {"kind": kind, **fields}
        # Keep full private results even if the next model request never returns.
        # A new instance gets a new session; this journal does not resume a stream.
        self.workdir.mkdir(parents=True, exist_ok=True)
        row = {"session": self._event_session, "event_number": len(self.events) + 1, "event": e}
        with (self.workdir / "learning-events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.events.append(e)
        return e


    @property
    def replaced(self) -> dict[str, str | None]:
        """Each adopted rewrite and the entry it replaced, from the entry's own recorded origin."""
        out: dict[str, str | None] = {}
        for sha, p in self.library.projection()["entries"].items():
            if p["status"] == "adopted":
                origin = self.library.entry(sha).get("origin") or {}
                if "predecessor" in origin:
                    out[sha] = origin["predecessor"]
        return out


    @property
    def pair_of(self) -> dict[str, str]:
        """Each adopted paired check and the procedure adopted with it, while that procedure is adopted."""
        proj = self.library.projection()["entries"]
        out: dict[str, str] = {}
        for sha, p in proj.items():
            if p["status"] == "adopted" and p["kind"] == "check":
                origin = self.library.entry(sha).get("origin") or {}
                proc = origin.get("procedure")
                if origin.get("pair") == "check" and proj.get(proc, {}).get("status") == "adopted":
                    out[sha] = proc
        return out


    def _change(self, changes: list[tuple[str, str, str]], reason: str, episode: str | None) -> None:
        """Make status changes that must be seen together, as one library group: all are planned on disk first,
        and none is visible until all are made (PAIR153.VISIBLE)."""
        plan = [{"entry": sha, "status": status, "reason": why} for sha, status, why in changes]
        group = self.library.open_group(plan, reason, episode=episode)
        for change in plan:
            self.library.set_status(change["entry"], change["status"], change["reason"], episode=episode,
                                    group=group)
        self.library.close_group(group)


    def floor_reason(self) -> dict | None:
        eligible = [c for c in self.completed if c.eligible]
        if self.floor_after_cold_start:
            # Development alternative: report the first 20 eligible episodes as cold start, then assess a full
            # 20-episode post-cold-start window (first at eligible episode 40). Keep all history and ordinary
            # miss feedback; adoption never resets this window. The legacy/protocol default below is unchanged.
            eligible = eligible[FLOOR_WINDOW:]
            if len(eligible) < FLOOR_WINDOW:
                return
        elif len(eligible) <= FLOOR_WINDOW:
            return
        window = eligible[-FLOOR_WINDOW:]
        # Met without help; an episode that failed only in parts no learned decision affects counts as met (with the
        # heading learned alone, a missed report).
        rate = sum(self.goal.met_for(c.goal_met, c.failed, self.learn_decisions) and not c.helped
                   for c in window) / FLOOR_WINDOW
        if rate < FLOOR_RATE and self.armed:
            self.armed, self.breaches = False, self.breaches + 1
            self._event("floor_breach", rate=rate)
            return {"episode": window[-1].episode, "failed": "floor"}
        elif rate >= FLOOR_RATE and not self.armed:
            self.armed = True
            self._event("floor_rearmed", rate=rate)


    def _is_guard(self, sha: str) -> bool:
        """A trusted guard installed at stream start (steps 2 to 4): permanent, never a predecessor, never replaced."""
        origin = self.library.entry(sha).get("origin") or {}
        return any(k.startswith("trusted_") and k.endswith("_guard") for k in origin)


    def _learned(self, decision: str, kind: str) -> list[str]:
        """Adopted entries a rewrite may replace: everything but the permanent guards (Codex's TURN 185 finding)."""
        return [s for s in self.library.adopted(decision, kind) if not (kind == "check" and self._is_guard(s))]


    def sources(self) -> dict:
        out = {}
        for decision in self.pack.decisions:
            guards = [s for s in self.library.adopted(decision, "check") if self._is_guard(s)]
            if guards:
                out[f"{decision} guard (fixed)"] = self.library.entry(guards[0])["source"]
            for kind in ("procedure", "check"):
                shas = self._learned(decision, kind)
                if shas:
                    out[f"{decision} {kind}"] = self.library.entry(shas[0])["source"]
        return out


    def trial(self, sha: str, predecessor: str | None, decision: str, trial: list[Completed],
              candidate_check: str | None = None, predecessor_check: str | None = None) -> tuple[bool, dict]:
        """Shadow flights of the trial scenarios with the candidate in place, no model calls. The candidate passes
        only if it passes its own decision's component trial in every one (decision D27, `component_trial`); a
        flight function that returns only a whole-goal result (the invented ones in tests) is judged on that."""
        self._candidate = (sha, decision)
        results = []
        root = Path(tempfile.mkdtemp(prefix="shadow-", dir=self.workdir))
        try:
            shadow = Library(root / "library", pack=self.pack)
            for d in self.pack.decisions:
                for kind in ("procedure", "check"):
                    for s in self.library.adopted(d, kind):
                        if s in (predecessor, predecessor_check):
                            continue
                        e = self.library.entry(s)
                        shadow.set_status(shadow.add(e), "adopted", "shadow copy")
            if candidate_check is not None:
                # A paired check is in place for its procedure's trial, in the shadow only (D27, item 2).
                shadow.set_status(shadow.add(self.library.entry(candidate_check)), "adopted", "paired check under trial")
            shadow.set_status(shadow.add(self.library.entry(sha)), "adopted", "candidate under trial")
            for c in trial:
                results.append(self.fly(shadow, c, root))
            # The regression set is flown only if the recent episodes all pass, so a failing candidate costs no more
            # than before; it is marked so that what failed can be told apart. Its episodes were met whole, so each
            # must meet the whole goal again without help, not only the candidate's own part (Codex, TURN220).
            if self._regression_set and results and all(_trial_ok(r) for r in results):
                for c in self._regression_set:
                    results.append({**self.fly(shadow, c, root), "regression": True})
        finally:
            shutil.rmtree(root, ignore_errors=True)
        passed = bool(results) and all(_regression_ok(r) if r.get("regression") else _trial_ok(r) for r in results)
        return passed, {"trial": results}


    def regression_set(self, recent: list[Completed]) -> list[Completed]:
        """Up to self.regression earlier finished, eligible episodes the library met by itself (no model call, no
        help), not in the recent trial set, spread evenly over that history and keeping the newest. From Telohelm's own
        records only: it never says which disturbance an episode held (decision D8)."""
        if not self.regression:
            return []
        recent_ids = {c.episode for c in recent}
        pool = [c for c in self.completed if c.eligible and c.alone and c.episode not in recent_ids]
        k = self.regression
        if len(pool) <= k:
            return pool
        if k == 1:
            return [pool[-1]]
        return [pool[round(i * (len(pool) - 1) / (k - 1))] for i in range(k)]


    def _fly(self, shadow: Library, c: Completed, root: Path) -> dict:
        """One shadow flight: the trial scenario from its start, the shadow library, no model calls."""
        environment = environment_for(self.flight_pack)
        d = root / c.episode
        d.mkdir()
        ledger = Ledger()
        br = B.Broker(c.episode, TrustedLog(d / "broker.log", d / "broker.heads", "shadow"), ledger, self.tool,
                      goal=self.goal,
                      condition="shadow", stream="shadow", tools=environment.tools)
        steps = self.shadow_max_steps if self.shadow_max_steps is not None else environment.max_steps
        r = L.run_episode(c.scenario, shadow, L.NoModels(), TrustedLog(d / "log", d / "heads", "shadow"), ledger,
                          max_steps=steps, broker=br, pack=self.flight_pack)
        o = r["outcome"]
        decisions = [p for p in (json.loads(l)["payload"] for l in (d / "log").read_text().splitlines())
                     if p.get("kind") == "decision"]
        sha, decision = getattr(self, "_candidate", (None, None))
        # The evidence the verdict rests on is kept: the candidate decision's records in the result, and the shadow
        # flight's protected log and heads copied out before the temporary folder goes (Codex's TRIAL147.AUDIT).
        keep = _new_evidence_dir(self.workdir / "trial-evidence" / str(sha)[:16] / c.episode)
        for name in ("log", "heads", "broker.log", "broker.heads"):
            if (d / name).exists():
                shutil.copy2(d / name, keep / name)
        # What reproducing the trial needs beyond the logs (which hold the briefing and every public bundle): the
        # scenario, the shadow library as flown (the candidate's source included), the trial's ledger and its
        # settings (Codex's TRIAL149). Everything is then made read-only; a later trial never writes here.
        (keep / "scenario.json").write_text(json.dumps(_plain(c.scenario), indent=1, default=str))
        if getattr(shadow, "root", None) is not None and Path(shadow.root).exists():
            shutil.copytree(shadow.root, keep / "shadow-library")
        from dataclasses import asdict
        (keep / "ledger.jsonl").write_text("".join(json.dumps(asdict(x), default=str) + "\n" for x in ledger.calls))
        (keep / "settings.json").write_text(json.dumps({
            "candidate": sha, "decision": decision, "max_steps": steps, "tool": getattr(self.tool, "__qualname__",
                                                                                         repr(self.tool)),
            "trial_ordinal": keep.name, "completed_ordinal": c.ordinal,
            **({"environment": environment.record()} if self.flight_pack.environment is not None else {})},
            indent=1, default=str))
        result = {"episode": c.episode, "goal_met": bool(o["goal_met"]), "help": len(o.get("help_requests") or []),
                "end_reason": o["end_reason"], "first_actions": r["actions"][:3],
                "component": environment.trial(decision, sha, o, decisions), "outcome": o,
                "decision_evidence": [p for p in decisions if p.get("decision") == decision],
                "evidence_dir": str(keep), "candidate_source_sha256": _source_sha(shadow, sha)}
        _write_atomic(keep / "result.json", json.dumps(result, sort_keys=True, default=str).encode())
        for f in keep.rglob("*"):
            if f.is_file():
                f.chmod(0o444)
        return result


    def check_replay(self, source: str, predecessor: str | None, decision: str | None = None,
                     rewrite: int | None = None, candidate: str | None = None) -> tuple[bool, dict]:
        """The replay test for a check. With a candidate (its library entry) and a case store, the check takes its
        own fresh set (fresh_set); otherwise the given rewrite's, as before."""
        if self.flight_pack.hidden:
            # The stored cases hold full contexts; a check must not qualify on fields hidden in flight, with or
            # without the replay test (Codex's TURN237 review).
            return False, {"refused": "checks are not qualified when the pack hides fields"}
        if self.no_replay:
            return True, {"replay": {"passed": True, "skipped": True,
                                     "reason": "not run (Telohelm without the replay test, D5)"}}
        old = self.library.entry(predecessor)["source"] if predecessor else None
        if isinstance(self.corpus, CaseStore):
            allocation = None
            if candidate is not None:
                allocation = self.fresh_set(candidate, decision)
                if allocation is None:
                    return False, {"refused": f"no unused fresh set is left for {decision} checks"}
                rewrite = allocation
            # D29: the check meets only cases of its own decision, with its own fresh wrong cases; a decision
            # without settled wrong cases, 100 right cases or the fresh cases cannot pass.
            wrong, right, missing = self.corpus.replay_cases(self.stream, decision, rewrite)
            if missing:
                return False, {"refused": "; ".join(missing)}
            r = replay_test(source, wrong, right, old_source=old)
            return r["passed"], {"replay": {**{k: r[k] for k in ("passed", "right_accepted", "old_right_accepted",
                                                                 "wrong_cases", "right_cases")}, **_two_facts(r),
                                            "fresh_set": rewrite}}
        # The protocol's replay test uses the monitor's 100 right cases; fewer is a missing corpus, not a pass
        # (Codex's MVP119.RIGHT100).
        if self.corpus is None or not self.corpus.wrong or len(self.corpus.right) != RIGHT_CASES:
            return False, {"refused": f"the replay test needs stored wrong cases and exactly {RIGHT_CASES} right cases"}
        r = replay_test(source, self.corpus.wrong, self.corpus.right, old_source=old)
        return r["passed"], {"replay": {**{k: r[k] for k in ("passed", "right_accepted", "old_right_accepted")},
                                        **_two_facts(r)}}


    def fresh_set(self, candidate: str, decision: str) -> int | None:
        """The fresh wrong-case set this check candidate is replayed on (Codex's TURN207; agreed TURN211): the next
        unused one for its decision, so that no fresh case is seen by two candidates. The allocation is written and
        synced before any replay, bound to the candidate's library entry, its stream, its decision and the identity of
        the set's cases, and found again after a restart, so a candidate never gets a second set. A folder holding
        another stream's allocations, or an allocation whose decision or cases no longer match, is refused rather than
        reused (Codex's FRESH213.NAMESPACE). A torn last line is not repaired here: reading fails and nothing is
        replayed until someone inspects and truncates the file. None when every set for the decision is taken."""
        path = self.workdir / "fresh-allocations.jsonl"
        rows = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
        others = {r.get("stream") for r in rows} - {self.stream}
        if others:
            raise ValueError(f"fresh allocations in {path} belong to stream(s) {sorted(map(str, others))}, "
                             f"not {self.stream}")
        for r in rows:                              # every earlier reservation must still name the same cases
            if r.get("fresh_cases") != self.corpus.fresh_set_digest(self.stream, r["decision"], r["fresh_set"]):
                raise ValueError(f"fresh set {r['fresh_set']} for {r['decision']} has changed since it was allocated")
        for r in rows:
            if r["candidate"] == candidate:
                if r["decision"] != decision:
                    raise ValueError(f"the allocation of {candidate} does not match its decision or fresh cases")
                return r["fresh_set"]
        used = {r["fresh_set"] for r in rows if r["decision"] == decision}
        free = [o for o in self.corpus.fresh_ordinals(self.stream, decision) if o not in used]
        if not free:
            self._event("fresh_sets_exhausted", candidate=candidate, decision=decision, used=len(used))
            return None
        row = {"candidate": candidate, "decision": decision, "fresh_set": free[0], "stream": self.stream,
               "fresh_cases": self.corpus.fresh_set_digest(self.stream, decision, free[0])}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._event("fresh_set_allocated", **row)
        return free[0]


    def _attempt_pair(self, n: int, calls: int, reply: dict, boundary: Completed, trial: list[Completed],
                      reasons: list, call: int | None = None) -> Verdict:
        """Decision D27, item 2. The check takes the replay test first; if it passes, the procedure is trialled with
        that check in place. Both are adopted together only if both pass; until then neither replaces anything, and
        every failure is kept. An intent record is written before adoption, so a crash part-way is completed by
        `reconcile_pairs`. If the check is later retired, its procedure is suspended with it (`confirm_wrong`)."""
        decision = reply["decision"]
        preds = {k: (self._learned(decision, k) or [None])[0] for k in ("procedure", "check")}
        origin = {"calls": [], "attempt": n, "episode": boundary.episode, **({"candidate": call} if call is not None else {})}
        proc = self.library.add({"kind": "procedure", "decision": decision, "conditions": reply["procedure"]["conditions"],
                                 "source": reply["procedure"]["source"],
                                 "origin": {**origin, "predecessor": preds["procedure"], "pair": "procedure"}},
                                episode=boundary.episode)
        check = self.library.add({"kind": "check", "decision": decision, "conditions": reply["check"]["conditions"],
                                  "source": reply["check"]["source"],
                                  "origin": {**origin, "predecessor": preds["check"], "pair": "check",
                                             "procedure": proc}}, episode=boundary.episode)
        detail: dict[str, Any] = {}
        passed = False
        check_ok, detail["check"] = self.check_replay(reply["check"]["source"], preds["check"], decision, n, check)
        if check_ok:
            proc_ok, detail["procedure"] = self.trial(proc, preds["procedure"], decision, trial,
                                                      candidate_check=check, predecessor_check=preds["check"])
            if proc_ok:
                record = {"procedure": proc, "check": check, "decision": decision, "attempt": n,
                          **({"candidate": call} if call is not None else {}),
                          "predecessor_procedure": preds["procedure"], "predecessor_check": preds["check"]}
                self._pair_log({**record, "state": "adopting"})
                self._adopt_pair(record, boundary.episode)
                self._pair_log({**record, "state": "adopted"})
                passed = True
        return self._event("attempt_result", attempt=n, calls=calls, candidate=proc, paired_check=check,
                           entry_kind="pair", decision=decision, predecessor=preds["procedure"],
                           predecessor_check=preds["check"], passed=passed, detail=detail, reasons=reasons)


    def _adopt_pair(self, record: dict, episode: str | None) -> None:
        proj = self.library.projection()["entries"]
        changes = []
        for kind in ("check", "procedure"):
            sha, pred = record[kind], record[f"predecessor_{kind}"]
            if proj.get(sha, {}).get("status") != "adopted":
                changes.append((sha, "adopted", f"pair from attempt {record['attempt']} passed"))
            if pred is not None and proj.get(pred, {}).get("status") == "adopted":
                changes.append((pred, "suspended", f"replaced by {sha}"))
        if changes:
            self._change(changes, f"pair from attempt {record['attempt']} adopted", episode)


    def _pair_log(self, row: dict) -> None:
        path = self.workdir / "pairs.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())


    def reconcile_pairs(self) -> list[dict]:
        """Complete any pair adoption that a crash interrupted after both tests had passed: an 'adopting' record with
        no 'adopted' record is carried through, so the library never holds only half a pair."""
        finished = [self._event("group_completed", group=g) for g in self.library.complete_open_groups()]
        path = self.workdir / "pairs.jsonl"
        if not path.exists():
            return finished
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        done = {(r["procedure"], r["check"]) for r in rows if r["state"] == "adopted"}
        for r in rows:
            if r["state"] == "adopting" and (r["procedure"], r["check"]) not in done:
                record = {k: v for k, v in r.items() if k != "state"}
                self._adopt_pair(record, None)
                self._pair_log({**record, "state": "adopted"})
                done.add((r["procedure"], r["check"]))
                finished.append(self._event("pair_reconciled", procedure=r["procedure"], check=r["check"]))
        return [e for e in finished if e["kind"] == "pair_reconciled"]


    def confirm_wrong(self, case: dict) -> list[dict]:
        """A newly confirmed wrong answer joins the store; rewritten checks that now accept a wrong answer are
        retired and their predecessors restored. Nothing else triggers an undo."""
        store = isinstance(self.corpus, CaseStore)
        if store:
            # Only a confirmed wrong answer of this coordinator's own stream is live evidence. A held-back final or
            # eligibility case, or another stream's case, is refused before any reconciliation, admission or replay
            # (B8.CONFIRM.SCOPE); its split and stream are never rewritten to make it fit.
            # The checks run on one detached snapshot from the case's canonical bytes, and that snapshot is what is
            # admitted, so a case JSON cannot encode, or one the caller changes later, is refused or ignored before
            # reconciliation (CONFIRM159.JSON).
            case = snapshot(case)
            if case.get("split") != "confirmed_wrong" or case.get("stream") != self.stream:
                raise StoreError(f"live confirmation takes a confirmed_wrong case of stream {self.stream}, not "
                                 f"{(case.get('split'), case.get('stream'))}")
            errors = case_errors(case, self.corpus.development)
            if errors:
                raise StoreError(f"case refused: {errors}")
        self.reconcile_pairs()            # an interrupted undo or adoption is finished before this one starts
        start = len(self.events)
        if store:
            self.corpus.admit(case)       # a confirmed_wrong case record, admitted only with verified evidence
        else:
            if self.corpus is None:
                self.corpus = Corpus()
            self.corpus.wrong.append(case)
        if self.no_replay:                # D5: replay evidence never retires a check in this arm
            self._event("undo_skipped", reason="Telohelm without the replay test (D5)")
            return self.events[start:]
        for sha, predecessor in list(self.replaced.items()):
            proj = self.library.projection()["entries"]
            p = proj.get(sha, {})
            if p.get("kind") != "check" or p.get("status") != "adopted":
                continue
            if store:
                wrong, right = self.corpus.undo_cases(self.stream, self.library.entry(sha)["decision"])
            else:
                wrong, right = self.corpus.wrong, self.corpus.right
            r = replay_test(self.library.entry(sha)["source"], wrong, right, old_source=None, min_right=0)
            if not r["wrong_accepted"]:
                continue
            # The check is retired and what it replaced is restored. A procedure adopted with it depends on it, so
            # that procedure is suspended and the one it replaced restored (D27, item 2). All of it is one group, so
            # a stop part-way is finished from the plan, once (PAIR153.UNDO.CUT). Only a suspended entry is
            # restored: a retired or never-adopted one is not revived.
            changes = [(sha, "retired", "replay test shows it accepts a confirmed wrong answer")]
            restored = predecessor if proj.get(predecessor, {}).get("status") == "suspended" else None
            if restored is not None:
                changes.append((restored, "adopted", f"restored after {sha} was retired"))
            proc = self.pair_of.get(sha)
            prior = None
            if proc is not None:
                changes.append((proc, "suspended", f"its paired check {sha} was retired"))
                before = (self.library.entry(proc).get("origin") or {}).get("predecessor")
                if before is not None and proj.get(before, {}).get("status") == "suspended":
                    prior = before
                    changes.append((prior, "adopted", f"restored after {proc} was suspended"))
            self._change(changes, f"undo of {sha}", None)
            self._event("undo", retired=sha, restored=restored)
            if proc is not None:
                self._event("undo", retired=proc, restored=prior, reason="paired check retired")
        return self.events[start:]


    def submit(self, n: int, calls: int, reply: dict, boundary: PublicEpisode,
                       reasons: list, call: int | None = None) -> Verdict:
        """One valid reply through its test (the trial for a procedure, the replay test for a check, both for a pair),
        adopted if it passes, with its outcome remembered for later attempts' feedback."""
        trial = self._attempt_trial
        if reply["kind"] in ("check", "pair") and self.flight_pack.hidden:
            # Never reached while the constructor refuses this combination; kept so no path can qualify such a check.
            event = self._event("candidate_refused", attempt=n, call=call, stage="unsupported",
                                errors=["learned checks are not qualified when the pack hides fields"])
            return self._verdict(event, remember=False)
        if reply["kind"] == "pair":
            event = self._attempt_pair(n, calls, reply, boundary, trial, reasons, call=call)
            return self._verdict(event)
        kind, decision = reply["kind"], reply["decision"]
        predecessors = self._learned(decision, kind)
        predecessor = predecessors[0] if predecessors else None
        origin = {"calls": [], "attempt": n, "episode": boundary.episode, "predecessor": predecessor}
        if call is not None:
            origin["candidate"] = call
        sha = self.library.add({"kind": kind, "decision": decision, "conditions": reply["conditions"],
                                "source": reply["source"], "origin": origin}, episode=boundary.episode)
        if kind == "procedure":
            passed, detail = self.trial(sha, predecessor, decision, trial)
        else:
            passed, detail = self.check_replay(reply["source"], predecessor, decision, n, sha)
        if passed:
            changes = [(sha, "adopted", f"attempt {n} passed")]
            if predecessor is not None:
                changes.append((predecessor, "suspended", f"replaced by {sha}"))
            self._change(changes, f"attempt {n} adopted", boundary.episode)
        event = self._event("attempt_result", attempt=n, calls=calls, candidate=sha, entry_kind=kind,
                            decision=decision, predecessor=predecessor, passed=passed, detail=detail, reasons=reasons,
                            **({"candidate_call": call} if call is not None else {}))
        return self._verdict(event)


def reference_solvable(scenario: Any) -> bool | None:
    """The independent solvable label, recorded for analysis only: the reference pilot flies the scenario without
    help. True only for a settled success, False for a settled failure, None (unknown) when the world could not settle
    the outcome (Codex's GEN141). Since 27 October no trial set, retention set or floor reads it (Monitor.record)."""
    from .taskpacks.tanker.world import phase5 as phase5
    o = phase5.fly(scenario, ask_help=False)
    if o.get("settled") is not True:
        return None
    return o.get("goal_met") is True




def _source_sha(library: Library, sha: str | None) -> str | None:
    """SHA-256 of a library entry's source, or None if there is no such entry."""
    if not sha:
        return None
    try:
        return hashlib.sha256(library.entry(sha)["source"].encode()).hexdigest()
    except Exception:
        return None


def _new_evidence_dir(parent: Path) -> Path:
    """A fresh, numbered folder for one trial flight's evidence; an existing one is never reused (TRIAL149.REPEAT)."""
    parent.mkdir(parents=True, exist_ok=True)
    n = len([p for p in parent.iterdir() if p.is_dir()])
    while True:
        d = parent / f"trial-{n:04d}"
        try:
            d.mkdir()
            return d
        except FileExistsError:
            n += 1


def _plain(obj: Any) -> Any:
    """A scenario as plain data for its evidence file, or its text if it is not a dataclass."""
    from dataclasses import asdict, is_dataclass
    return asdict(obj) if is_dataclass(obj) else repr(obj)

