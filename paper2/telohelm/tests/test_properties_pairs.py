"""Stateful property test of paired adoption, undo, restart and crash recovery (study2/learning.py, decision D27
item 2; Codex's PAIR153 findings), with Hypothesis. Suggested in paper2/notes/HYPOTHESIS-TESTING-NOTE.md, item 1.

Development test only, with invented entries and no model, world or sandbox. Hypothesis generates sequences of: a pair
proposed whose check and trial each pass or fail; a confirmed wrong answer that one adopted rewritten check accepts;
a restart from disk; and a stop part-way through either change, either before a status write or as a torn record.
After every step it checks the rules the code promises. Runs are derandomised so they repeat exactly.
"""
import json
import shutil
import tempfile
from pathlib import Path

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import settings, strategies as st                                   # noqa: E402
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule  # noqa: E402

import telohelm.learning as G                                                     # noqa: E402
from telohelm.runtime.library import Library                                           # noqa: E402

BOUNDARY = G.Completed(5, "invented-e", None, False, "both", True)
CUTS = st.one_of(st.none(), st.tuples(st.sampled_from(["stop", "torn"]), st.integers(0, 3)))


class Pairs(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.dir = Path(tempfile.mkdtemp(prefix="pairs-"))
        self.n, self.crashed, self.retired_ever = 0, False, set()
        self.check_ok = self.trial_ok = True
        lib = Library(self.dir / "library")
        for kind, source in (("procedure", 'def decide(ctx):\n    return {"heading_deg": 12.0}\n'),
                             ("check", "def check(ctx, out):\n    return 'reject'\n")):
            sha = lib.add({"kind": kind, "decision": "heading", "conditions": [], "source": source,
                           "origin": {"invented": True}})
            lib.set_status(sha, "adopted", "invented starting library")
        self._open(lib)

    def teardown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    # -- plumbing -------------------------------------------------------------------------------------------------
    def _fly(self, shadow, c, root):
        return {"goal_met": self.trial_ok, "help": 0, "episode": c.episode}

    def _open(self, lib):
        self.lib = lib
        self.co = G.Coordinator(lib, self.dir, None, lambda s: True, lambda a: {}, fly=self._fly)
        self.co.check_replay = lambda *a: (self.check_ok, {"invented": True})

    def _cut(self, cut):
        """Make the k-th status write of the next change stop the process, before it writes or half way through."""
        if cut is None:
            return
        kind, k = cut
        real, calls = self.lib.set_status, [0]

        def set_status(*a, **kw):
            calls[0] += 1
            if calls[0] == k + 1:
                if kind == "stop":
                    raise RuntimeError("invented stop")
                self.lib.crash_at = "mid_event"
            return real(*a, **kw)
        self.lib.set_status = set_status

    def _run(self, change, cut):
        self._cut(cut)
        try:
            return change()
        except Exception as e:                  # the invented stop, or the library's torn-write Crash
            if cut is None:
                raise
            self.crashed = True
            return e
        finally:
            self.lib.__dict__.pop("set_status", None)

    def _states(self):
        return {s: v["status"] for s, v in self.lib.projection()["entries"].items()}

    def _origin(self, sha):
        return self.lib.entry(sha).get("origin") or {}

    # -- steps ----------------------------------------------------------------------------------------------------
    @precondition(lambda self: not self.crashed)
    @rule(check_ok=st.booleans(), trial_ok=st.booleans(), cut=CUTS)
    def propose_pair(self, check_ok, trial_ok, cut):
        self.n += 1
        self.check_ok, self.trial_ok = check_ok, trial_ok
        reply = {"kind": "pair", "decision": "heading",
                 "procedure": {"conditions": [], "source": f'def decide(ctx):\n    return {{"heading_deg": {self.n}.0}}\n'},
                 "check": {"conditions": [], "source": f"def check(ctx, out):\n    return 'accept' if {self.n} else 'reject'\n"},
                 "reason": {"code": "missing_tool", "text": "invented"}}
        self._run(lambda: self.co._attempt_pair(self.n, 1, reply, BOUNDARY, [BOUNDARY], [{"failed": "both"}]), cut)

    @precondition(lambda self: not self.crashed)
    @rule(pick=st.integers(0, 10), cut=CUTS)
    def confirm_wrong(self, pick, cut):
        rewritten = sorted(s for s in self.co.replaced if self._states().get(s) == "adopted"
                           and self.lib.projection()["entries"][s]["kind"] == "check")
        target = self.lib.entry(rewritten[pick % len(rewritten)])["source"] if rewritten else None
        real = G.replay_test
        G.replay_test = lambda source, *a, **k: {"wrong_accepted": [0] if source == target else []}
        try:
            self._run(lambda: self.co.confirm_wrong({"context": {"t": 0}, "output": {"heading_deg": 1.0}}), cut)
        finally:
            G.replay_test = real

    @rule()
    def restart(self):
        before = None if self.crashed else (dict(self.co.pair_of), dict(self.co.replaced))
        lib = Library(self.dir / "library")
        lib.recover()
        self._open(lib)
        self.co.reconcile_pairs()
        self.crashed = False
        assert not lib.open_groups(), "every interrupted group is finished on restart"
        rows = [json.loads(l) for l in (self.dir / "pairs.jsonl").read_text().splitlines()] \
            if (self.dir / "pairs.jsonl").exists() else []
        done = {(r["procedure"], r["check"]) for r in rows if r["state"] == "adopted"}
        assert all((r["procedure"], r["check"]) in done for r in rows if r["state"] == "adopting"), \
            "a pair whose adoption was interrupted is finished on restart"
        if before is not None:
            assert (self.co.pair_of, self.co.replaced) == before, "a clean restart keeps the pairs and replacements"

    # -- rules after every step -----------------------------------------------------------------------------------
    @invariant()
    def one_adopted_procedure_and_check(self):
        for kind in ("procedure", "check"):
            assert len(self.lib.adopted("heading", kind)) <= 1, kind

    @invariant()
    def a_paired_procedure_flies_only_with_its_check(self):
        states = self._states()
        for sha, status in states.items():
            if status == "adopted" and self._origin(sha).get("pair") == "procedure":
                checks = [c for c, s in states.items() if s == "adopted" and self._origin(c).get("procedure") == sha]
                assert checks, f"procedure {sha[:8]} adopted without its paired check"

    @invariant()
    def a_retired_entry_is_never_adopted_again(self):
        states = self._states()
        assert not {s for s, v in states.items() if v == "adopted"} & self.retired_ever
        self.retired_ever |= {s for s, v in states.items() if v == "retired"}

    @invariant()
    def a_reader_never_sees_part_of_an_open_group(self):
        for seq in self.lib.open_groups():
            assert self.lib.projection()["entries"] == self.lib.projection(upto_seq=seq)["entries"]

Pairs.TestCase.settings = settings(derandomize=True, max_examples=60, stateful_step_count=25, deadline=None)
TestPairs = Pairs.TestCase
