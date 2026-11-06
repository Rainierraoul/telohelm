"""Per-candidate fresh replay sets (study2/learning.py, Coordinator.fresh_set; Codex's TURN207, agreed TURN211).

Each check candidate is replayed on its own fresh wrong-case set, the next unused one for its decision. The
allocation is durable, bound to the candidate's library entry, kept across restarts, and refused when no set is left.
"""

import telohelm.learning_monitor as GM  # replay qualification moved to the monitor
import json
from types import SimpleNamespace

import pytest

import telohelm.learning as G
from telohelm.qualification.store import CaseStore
from telohelm.runtime.library import Library


class Store(CaseStore):
    """A stand-in case store: fresh sets 1..n for each decision, recording which set each replay asked for."""

    def __init__(self, sets=3):
        self.sets, self.asked = sets, []

    def fresh_ordinals(self, stream, decision):
        return list(range(1, self.sets + 1))

    def fresh_set_digest(self, stream, decision, rewrite):
        return f"{getattr(self, 'salt', '')}{stream}-{decision}-{rewrite}"

    def replay_cases(self, stream, decision, rewrite):
        self.asked.append(rewrite)
        return [{"context": {}, "output": {}}], [{"context": {}, "output": {}}] * 100, []


def check_reply(tag):
    return json.dumps({"kind": "check", "decision": "heading", "conditions": [],
                       "source": f"def check(ctx, out):\n    return 'accept' if {tag} else 'reject'\n",
                       "reason": {"code": "missing_tool", "text": "x"}})


class Proposer:
    def __init__(self, texts):
        self.texts = list(texts)

    def attempt_budget(self, episode):
        return 3

    def __call__(self, messages, meta):
        return self.texts.pop(0)


def coordinator(tmp_path, texts, store, passing=(), candidates=3):
    co = G.Coordinator(Library(tmp_path / "lib"), tmp_path, Proposer(texts), lambda _: True, lambda *_: None,
                       corpus=store, stream=4401, candidates=candidates)
    for i in range(1, 6):
        co.completed.append(G.Completed(i, f"e{i}", SimpleNamespace(), False, "join", True))
    co.queue = [{"episode": "e5", "failed": "join"}]
    G_replay = G.replay_test

    def replay(source, wrong, right, old_source=None, **k):
        ok = any(f"if {t} else" in source for t in passing)
        return {"passed": ok, "reasons": [] if ok else ["x"], "wrong_accepted": [] if ok else [0],
                "right_accepted": 100, "old_right_accepted": None, "wrong_cases": 1, "right_cases": 100}
    return co, replay, G_replay


def test_each_candidate_in_an_attempt_gets_its_own_fresh_set(tmp_path, monkeypatch):
    store = Store()
    co, replay, _ = coordinator(tmp_path, ["not json", check_reply(1), check_reply(2)], store, passing=(2,))
    monkeypatch.setattr(GM, "replay_test", replay)
    event = co.attempt(co.completed[-1])
    assert event["passed"] and store.asked == [1, 2]          # the malformed reply took no set
    rows = [json.loads(l) for l in (tmp_path / "fresh-allocations.jsonl").read_text().splitlines()]
    assert [r["fresh_set"] for r in rows] == [1, 2] and len({r["candidate"] for r in rows}) == 2


def test_a_restart_finds_the_candidates_set_again_and_never_reissues_one(tmp_path, monkeypatch):
    store = Store()
    co, replay, _ = coordinator(tmp_path, [check_reply(1)], store, candidates=1)
    monkeypatch.setattr(GM, "replay_test", replay)
    co.attempt(co.completed[-1])
    first = json.loads((tmp_path / "fresh-allocations.jsonl").read_text().splitlines()[0])
    again = G.Coordinator(Library(tmp_path / "lib"), tmp_path, None, lambda _: True, lambda *_: None,
                          corpus=store, stream=4401)
    assert again.fresh_set(first["candidate"], "heading") == first["fresh_set"] == 1
    assert again.fresh_set("another-candidate", "heading") == 2


def test_when_every_set_is_taken_a_check_is_refused_and_counted(tmp_path, monkeypatch):
    store = Store(sets=1)
    co, replay, _ = coordinator(tmp_path, [check_reply(1), check_reply(2)], store, candidates=2)
    monkeypatch.setattr(GM, "replay_test", replay)
    co.attempt(co.completed[-1])
    results = [e for e in co.events if e["kind"] == "attempt_result"]
    assert store.asked == [1] and len(results) == 2
    assert "no unused fresh set" in json.dumps(results[1]["detail"])
    assert [e["kind"] for e in co.events if e["kind"].startswith("fresh_")] == ["fresh_set_allocated",
                                                                              "fresh_sets_exhausted"]


def test_the_allocation_is_on_disk_before_the_replay_runs(tmp_path, monkeypatch):
    store = Store()
    co, _, _ = coordinator(tmp_path, [check_reply(1)], store, candidates=1)

    def stop(*a, **k):
        raise RuntimeError("invented stop during the replay")
    monkeypatch.setattr(GM, "replay_test", stop)
    with pytest.raises(RuntimeError):
        co.attempt(co.completed[-1])
    assert len((tmp_path / "fresh-allocations.jsonl").read_text().splitlines()) == 1


def test_an_allocation_is_refused_under_another_stream_or_changed_fresh_cases(tmp_path):
    store = Store()
    co, _, _ = coordinator(tmp_path, [], store)
    co.fresh_set("invented-a", "heading")
    other, _, _ = coordinator(tmp_path, [], Store())
    other.stream = 4402
    with pytest.raises(ValueError, match="belong to stream"):
        other.fresh_set("invented-b", "heading")
    changed = Store()
    changed.salt = "rebuilt-"
    again, _, _ = coordinator(tmp_path, [], changed)
    with pytest.raises(ValueError, match="has changed since it was allocated"):
        again.fresh_set("invented-a", "heading")
    row = json.loads((tmp_path / "fresh-allocations.jsonl").read_text().splitlines()[0])
    assert row["stream"] == 4401 and row["fresh_cases"] == "4401-heading-1"
