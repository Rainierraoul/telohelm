"""The shared reasoning allowance and the automatic pause (B36, decision D26). Invented requests and rows only."""
from __future__ import annotations

import telohelm.ladder as LD
import telohelm.learning as G


def test_an_episode_has_four_allocations_and_an_attempt_at_most_three(tmp_path):
    a = LD.ReasoningAllowance(tmp_path / "allow.jsonl")
    k = a.key("telohelm", "s", "e1")
    assert a.attempt_budget(k) == 3
    assert [a.allocate(k, f"flight/{i}", "running", "provider") for i in range(3)] == [1, 2, 3]
    assert a.remaining(k) == 1 and a.attempt_budget(k) == 1
    assert a.allocate(k, "build/1", "build", "preflight_refused") == 4      # a refused preflight takes a slot
    assert a.allocate(k, "build/2", "build", "provider") is None and a.attempt_budget(k) == 0
    assert a.remaining(a.key("telohelm", "s", "e2")) == 4                    # another episode is untouched


def test_allocations_are_durable_and_recovery_adds_none(tmp_path):
    path = tmp_path / "allow.jsonl"
    a = LD.ReasoningAllowance(path)
    k = a.key("telohelm", "s", "e1")
    assert a.allocate(k, "flight/1", "running", "provider") == 1
    assert a.allocate(k, "flight/1", "running", "provider") == 1          # the same request again: no new slot
    b = LD.ReasoningAllowance(path)                                         # a restart reads the spent room back
    assert b.remaining(k) == 3 and b.allocate(k, "flight/1", "running", "provider") == 1


def test_a_rewrite_attempt_takes_its_budget_from_what_flight_left(tmp_path, monkeypatch):
    """Flight used 3 of the boundary episode's 4 allocations, so the attempt after it may make one call; the
    proposer's replies here are unusable, so without the shared allowance it would make three."""
    from .test_learning import proposer_with; from .test_learning import coordinator; from .test_learning import finish; from .test_learning import passing
    import telohelm.taskpacks.tanker.world.phase5 as phase5
    p, seen = proposer_with(tmp_path / "p", monkeypatch)
    p.allowance = LD.ReasoningAllowance(tmp_path / "allow.jsonl")
    boundary = phase5.draw(5).world_a.episode_id                # the fifth completed episode triggers the attempt
    key = p.allowance.key(p.condition, p.stream, boundary)
    for i in range(3):
        p.allowance.allocate(key, f"flight/{i}", "running", "provider")
    co, _ = coordinator(tmp_path / "co", p, fly=passing)
    finish(co, 5)
    ends = [e for e in co.events if e["kind"] in ("attempt_failed", "attempt_refused", "attempt_result")]
    assert ends and ends[-1].get("calls") == 1 and len(seen.sent) == 1
    assert p.allowance.remaining(key) == 0


def test_an_attempt_with_no_allowance_left_is_refused_without_a_call(tmp_path, monkeypatch):
    from .test_learning import proposer_with; from .test_learning import coordinator; from .test_learning import finish; from .test_learning import passing
    import telohelm.taskpacks.tanker.world.phase5 as phase5
    p, seen = proposer_with(tmp_path / "p", monkeypatch)
    p.allowance = LD.ReasoningAllowance(tmp_path / "allow.jsonl")
    key = p.allowance.key(p.condition, p.stream, phase5.draw(5).world_a.episode_id)
    for i in range(4):
        p.allowance.allocate(key, f"flight/{i}", "running", "provider")
    co, _ = coordinator(tmp_path / "co", p, fly=passing)
    finish(co, 5)
    assert co.events[-1] == {**co.events[-1], "kind": "attempt_refused", "reason": "reasoning allowance spent"}
    assert seen.sent == []


def test_a_refused_preflight_takes_an_allocation_and_makes_no_request(tmp_path, monkeypatch):
    from .test_learning import proposer_with
    p, seen = proposer_with(tmp_path, monkeypatch, count_tokens=lambda m: 5000)
    p.allowance = LD.ReasoningAllowance(tmp_path / "allow.jsonl")
    assert p([("user", "x")], {"attempt": 1, "call": 1, "episode": "e1"}) == ""
    k = p.allowance.key(p.condition, p.stream, "e1")
    assert seen.sent == [] and p.allowance.allocations(k)[0]["kind"] == "preflight_refused"


def rows(n, bad, episode="e0", prefix="r", **kw):
    # Each row states its purpose and has its own request identity; a row without a purpose is unplaced (PAUSE145).
    return [{"kind": "reasoning_model", "episode": episode, "scope": "running", "request_id": f"{prefix}{episode}-{i}",
             "status": "cut_off" if i < bad else "answered", "recovers": None, **kw} for i in range(n)]


def test_the_pause_is_due_only_above_ten_percent_of_started_calls():
    sched = [f"e{i}" for i in range(25)]
    assert LD.pause_due(rows(20, 3), sched)["due"] is True
    exactly = LD.pause_due(rows(20, 2), sched)
    assert exactly["due"] is False and exactly["fraction"] == 0.1
    none = LD.pause_due([], sched)
    assert none["due"] is False and none["fraction"] is None


def test_the_pause_ignores_recoveries_other_roles_and_episodes_after_the_first_twenty():
    sched = [f"e{i}" for i in range(25)]
    noise = (rows(5, 5, prefix="rec", recovers="r") + [{"kind": "small_model", "episode": "e0", "scope": "running",
                                                         "status": "cut_off", "recovers": None}]
             + rows(5, 5, episode="e22"))
    p = LD.pause_due(rows(10, 1) + noise, sched)
    assert (p["started"], p["bad"], p["due"]) == (10, 1, False)
    unknown = LD.pause_due(rows(10, 0) + [{"kind": "reasoning_model", "episode": "e1", "scope": "running",
                                           "request_id": "odd", "status": "interrupted", "recovers": None}], sched)
    assert unknown["unknown_status"] == 1 and unknown["bad"] == 0


def test_flying_calls_use_study_1s_per_role_output_limits():
    import paper2.experiment2.run_stream as R
    import json
    from pathlib import Path
    study1 = json.loads((next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'dcarr-runtime-0.1/study1/config.cluster.json').read_text())
    assert R.OUTPUT_TOKENS == {"small_model": study1["max_output_tokens"],
                               "reasoning_model": study1["max_output_tokens_by_tier"]["T2"]}
    assert G.PROTOCOL_OUTPUT_TOKENS == R.OUTPUT_TOKENS["reasoning_model"]
