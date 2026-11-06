"""Several separate candidates per learning attempt (study2/learning.py, _attempt_candidates)."""
import json
from types import SimpleNamespace

import pytest

import telohelm.learning as G
from telohelm.runtime.library import Library


def reply(tag):
    return json.dumps({"kind": "procedure", "decision": "heading", "conditions": [],
                       "source": f"def decide(ctx):\n    return {{'heading_deg': {tag}}}\n",
                       "reason": {"code": "missing_tool", "text": "x"}})


class Proposer:
    def __init__(self, texts, budget=3):
        self.texts, self.sent, self.budget = list(texts), [], budget

    def attempt_budget(self, episode):
        return self.budget

    def __call__(self, messages, meta):
        self.sent.append((messages, meta))
        return self.texts.pop(0)


def coordinator(tmp_path, proposer, passing, candidates=3):
    co = G.Coordinator(Library(tmp_path / "lib"), tmp_path, proposer, lambda _: True, lambda *_: None,
                       candidates=candidates, learn_decisions=("heading",))
    co.trial = lambda sha, pred, decision, trial: (co.library.entry(sha)["source"].endswith(f"{passing}}}\n"), {})
    for i in range(1, 6):
        co.completed.append(G.Completed(i, f"e{i}", SimpleNamespace(), False, "join", True))
    co.queue = [{"episode": "e5", "failed": "join"}]
    return co


def test_the_first_passing_candidate_is_adopted_and_the_rest_of_the_calls_are_not_spent(tmp_path):
    p = Proposer(["not json", reply(10), reply(20)])
    co = coordinator(tmp_path, p, passing=20)
    event = co.attempt(co.completed[-1])
    assert event["passed"] and event["candidate_call"] == 3 and len(p.sent) == 3
    assert [e["kind"] for e in co.events if e["kind"] in ("candidate_refused", "attempt_result")] == \
        ["candidate_refused", "attempt_result", "attempt_result"]
    assert [m for m, _ in p.sent][0] == [m for m, _ in p.sent][1] == [m for m, _ in p.sent][2]   # separate samples of one requests
    assert len(co.library.adopted("heading", "procedure")) == 1


def test_an_early_pass_stops_the_attempt(tmp_path):
    p = Proposer([reply(10), reply(20), reply(30)])
    co = coordinator(tmp_path, p, passing=10)
    event = co.attempt(co.completed[-1])
    assert event["passed"] and len(p.sent) == 1


def test_the_allowance_limits_the_candidates_and_a_failed_attempt_lists_what_was_tried(tmp_path):
    p = Proposer([reply(10), reply(20)], budget=2)
    co = coordinator(tmp_path, p, passing=99)
    event = co.attempt(co.completed[-1])
    assert event["kind"] == "attempt_failed" and len(p.sent) == 2 and len(event["tried"]) == 2
    assert not co.library.adopted("heading", "procedure")
    assert [h.get("refused_at") for h in co.history] == ["trial", "trial"]


def test_one_candidate_keeps_the_original_attempt_with_its_format_repair(tmp_path):
    p = Proposer(["not json", reply(10)])
    co = coordinator(tmp_path, p, passing=10, candidates=1)
    event = co.attempt(co.completed[-1])
    assert event["passed"] and len(p.sent) == 2
    assert p.sent[1][0][-1][0] == "user" and "Fix these problems" in p.sent[1][0][-1][1]


def test_the_number_of_candidates_is_checked():
    for bad in (0, 4, 2.0):
        with pytest.raises(ValueError):
            G.Coordinator(Library.__new__(Library), ".", None, None, None, candidates=bad)


def test_each_candidate_call_logs_its_own_example_receipt_and_a_pair_keeps_its_candidate_number(tmp_path):
    # Codex's TURN207 review: example-send receipts for calls 2 and 3, and the candidate ordinal in pair records.
    pair = json.dumps({"kind": "pair", "decision": "heading",
                       "procedure": {"conditions": [], "source": "def decide(ctx):\n    return {'heading_deg': 20}\n"},
                       "check": {"conditions": [], "source": "def check(ctx, out):\n    return 'accept'\n"},
                       "reason": {"code": "missing_tool", "text": "x"}})
    p = Proposer(["not json", pair])
    co = G.Coordinator(Library(tmp_path / "lib"), tmp_path, p, lambda _: True, lambda *_: None, candidates=3)
    for i in range(1, 6):
        co.completed.append(G.Completed(i, f"e{i}", SimpleNamespace(), False, "join", True))
    co.queue = [{"episode": "e5", "failed": "join"}]
    co.examples = True
    calls = []
    co._select_examples = lambda n, reasons: None
    co._log_examples = lambda n, call, fitted: calls.append(call)
    co.check_replay = lambda *a: (True, {})
    co.trial = lambda *a, **k: (True, {})
    event = co.attempt(co.completed[-1])
    assert event["passed"] and calls == [1, 2]
    rows = [json.loads(l) for l in (tmp_path / "pairs.jsonl").read_text().splitlines()]
    assert rows and all(r["candidate"] == 2 for r in rows)
    assert co.library.entry(event["candidate"])["origin"]["candidate"] == 2
