"""The adaptive learning budget (Simon, 15 October; paper2/notes/ADAPTIVE-LEARNING-BUDGET.md): two unadopted attempts
in a row raise the budget one level to a ceiling, two adoptions in a row lower it one level, format refusals are
ignored, and every attempt records the level it used. Invented proposers; no model."""
from __future__ import annotations

import json

import telohelm.learning as G
from telohelm.runtime.library import Library
from .test_learning_feedback import pair


def test_levels_move_up_on_misses_and_down_on_adoptions():
    b = G.AdaptiveBudget()
    assert b.current() == {"level": 0, **G.BUDGET_LEVELS[0]}
    assert b.observe("merit") is None
    assert b.observe("format") is None                       # neither counts nor resets
    assert b.observe("budget") == {"from": 0, "to": 1, "after": "budget"}
    assert b.observe("merit") is None and b.observe("merit") == {"from": 1, "to": 2, "after": "merit"}
    assert b.observe("budget") is None and b.observe("budget") is None   # the ceiling holds
    assert b.current()["level"] == 2
    assert b.observe("adopted") is None
    assert b.observe("merit") is None                        # a miss resets the run of adoptions
    assert b.observe("adopted") is None and b.observe("adopted") == {"from": 2, "to": 1, "after": "adopted"}


class Proposer:
    """An invented proposer that records the budget it is given before each attempt."""

    def __init__(self, answers):
        self.answers, self.used = iter(answers), []

    def use(self, level):
        self.used.append(dict(level))

    def __call__(self, messages, meta):
        return next(self.answers)


def test_the_coordinator_applies_and_records_the_level_of_each_attempt(tmp_path):
    lib = Library(tmp_path / "library")
    refuse = (False, {"replay": {"passed": False, "wrong_accepted_any": True, "right_short": False,
                                 "fewer_than_old": False}})
    proposer = Proposer([json.dumps(pair())] * 3)
    co = G.Coordinator(lib, tmp_path, proposer, lambda s: True, lambda a: {}, budget=G.AdaptiveBudget())
    co.completed = [G.Completed(5, "s4401-p1-e04", None, False, "both", True)]
    co.check_replay = lambda *a, **k: refuse
    for _ in range(3):
        co.attempt(co.completed[-1])
    assert [u["level"] for u in proposer.used] == [0, 0, 1]
    changes = [e for e in co.events if e["kind"] == "budget_changed"]
    assert changes == [{"kind": "budget_changed", "attempt": 2, "from": 0, "to": 1, "after": "merit",
                        "now": {"level": 1, **G.BUDGET_LEVELS[1]}}]
    assert [e["level"] for e in co.events if e["kind"] == "attempt_budget"] == [0, 0, 1]


def test_timeouts_count_as_budget_misses_and_format_refusals_do_not(tmp_path):
    lib = Library(tmp_path / "library")

    class Ledgered(Proposer):
        def __init__(self, answers, statuses):
            super().__init__(answers)
            self.statuses = iter(statuses)
            self.ledger = type("L", (), {"calls": []})()

        def __call__(self, messages, meta):
            status = next(self.statuses)
            self.ledger.calls.append(type("C", (), {"status": status})())
            return next(self.answers)
    long_reason = json.dumps(pair("x" * 340))
    proposer = Ledgered(["", "", "", "", "", "", long_reason, long_reason, long_reason],
                        ["timed_out"] * 6 + ["answered"] * 3)
    budget = G.AdaptiveBudget()
    co = G.Coordinator(lib, tmp_path, proposer, lambda s: True, lambda a: {}, budget=budget)
    co.completed = [G.Completed(5, "s4401-p1-e04", None, False, "both", True)]
    co.attempt(co.completed[-1])
    co.attempt(co.completed[-1])
    assert budget.current()["level"] == 1                     # two timed-out attempts
    co.attempt(co.completed[-1])
    assert budget.current()["level"] == 1 and budget.misses == 0   # a format refusal moves nothing


def test_the_proposer_rebuilds_its_provider_for_each_level(tmp_path):
    p = G.GatewayProposer("http://127.0.0.1:1", "Qwen/Qwen3.8-27B", tmp_path, "dev")
    p.use({"level": 2, **G.BUDGET_LEVELS[2]})
    assert p.max_output_tokens == 24_576 and p.provider.timeout_s_by_role["reasoning_model"] == 1_800.0
    assert p.provider._extra_body["chat_template_kwargs"]["reasoning_effort"] == "xhigh"
    assert p._config()["effort"] == "xhigh" and p._config()["timeout_s"] == 1_800.0


def test_the_pause_leaves_out_bad_learning_calls_below_the_ceiling(tmp_path):
    """Protocol 0.11 (proposed): a timed-out learning call below the ceiling is the signal the budget acts on and is
    not counted by the pause; at the ceiling it is."""
    from types import SimpleNamespace
    import paper2.experiment2.run_stream as RS
    from telohelm.runtime.ledger import Call

    def call(episode, status, n):
        return Call(request_id=f"r{episode}{n}", kind="reasoning_model", episode=episode, action_step=0,
                    started_s=None, wall_s=1.0, tokens_in=1, tokens_out=1, status=status, scope="build")
    budget = G.AdaptiveBudget()
    coordinator = SimpleNamespace(budget=budget, events=[
        {"kind": "attempt_budget", "attempt": 1, "episode": "e04", "level": 0},
        {"kind": "attempt_budget", "attempt": 2, "episode": "e09", "level": 2},
        {"kind": "attempt_budget", "attempt": 3, "episode": "e12", "level": 1},
        {"kind": "attempt_budget", "attempt": 4, "episode": "e12", "level": 2}])
    proposer = SimpleNamespace(ledger=SimpleNamespace(calls=[
        call("e04", "timed_out", 1), call("e04", "answered", 2), call("e04", "cut_off", 3), call("e04", "failed", 4),
        call("e09", "timed_out", 1), call("e12", "timed_out", 1)]))
    excluded = []
    rows = RS._ledger_rows(tmp_path, [], proposer, coordinator, excluded)
    assert [r["request_id"] for r in rows] == ["re042", "re044", "re091", "re121"]   # e12 reached the ceiling
    assert excluded == ["re041", "re043"]
    assert len(RS._ledger_rows(tmp_path, [], proposer, SimpleNamespace(budget=None, events=[]))) == 6   # fixed budget
