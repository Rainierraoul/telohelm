"""Step 2 of the proof-of-concept plan (paper2/notes/PROOF-OF-CONCEPT-PLAN.md, revision 3): the trusted heading guard
at stream start, the heading-only rules, the attempt cap, the learning input cap and the effort fix."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import telohelm.learning as G; import telohelm.local_model as LM; import paper2.experiment2.run_stream as RS; import telohelm.token_count as T
from telohelm.runtime.library import Library

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file())
GUARD = next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'paper2/telohelm/src/telohelm/taskpacks/tanker/guards/poc-heading-guard.py'


def test_the_cluster_guard_is_the_step1_guard_byte_for_byte():
    assert GUARD.read_bytes() == (next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'paper2/development/poc-step1/interface_guard.py').read_bytes()


@pytest.mark.parametrize("effort", ["high", "maximum", ""])
def test_an_effort_the_pinned_template_lacks_fails_at_start(effort):
    with pytest.raises(ValueError):
        LM.provider("http://127.0.0.1:1", "m", role="reasoning_model", effort=effort)


def test_every_budget_level_uses_an_effort_the_pinned_template_renders():
    msgs = [("system", "s"), ("user", "u")]
    for level in G.BUDGET_LEVELS:
        assert level["effort"] in LM.EFFORTS
        assert T.render(msgs, {"enable_thinking": True, "reasoning_effort": level["effort"]})
    for effort in LM.EFFORTS:
        LM.provider("http://127.0.0.1:1", "m", role="reasoning_model", effort=effort)


def test_the_pinned_template_refuses_high_as_job_4051059_found():
    with pytest.raises(Exception):
        T.render([("system", "s"), ("user", "u")], {"enable_thinking": True, "reasoning_effort": "high"})


def failing(i):
    return SimpleNamespace(episode_id=f"e{i:02d}"), {"goal_met": False, "joined": False, "report_correct": False}


def test_attempts_stop_at_the_cap(tmp_path):
    calls = []

    def propose(messages, meta):
        calls.append(meta)
        return "not json"

    co = G.Coordinator(Library(tmp_path / "library"), tmp_path, propose, lambda _: True, lambda *_: None,
                       max_attempts=4)
    for i in range(1, 31):
        co.after_episode(*failing(i), True)
    assert co.attempts == 4
    past = [e for e in co.events if e["kind"] == "no_attempt" and e.get("reason") == "past the last learning boundary"]
    assert [e["ordinal"] for e in past] == [25, 30]
    assert {m["attempt"] for m in calls} == {1, 2, 3, 4}


def test_no_cap_by_default(tmp_path):
    co = G.Coordinator(Library(tmp_path / "library"), tmp_path, lambda *_: "x", lambda _: True, lambda *_: None)
    for i in range(1, 26):
        co.after_episode(*failing(i), True)
    assert co.attempts == 5


def test_extra_rules_reach_the_proposer_prompt_only_when_set(tmp_path):
    lib = Library(tmp_path / "library")
    plain = G.Coordinator(lib, tmp_path, lambda *_: "", lambda _: True, lambda *_: None)
    assert plain.messages([])[0] == ("system", G.PROPOSER_RULES)
    poc = G.Coordinator(lib, tmp_path, lambda *_: "", lambda _: True, lambda *_: None, extra_rules=RS.POC_RULES)
    assert poc.messages([])[0] == ("system", G.PROPOSER_RULES + "\n" + RS.POC_RULES)


def test_the_poc_rules_state_heading_only_and_the_broker_request_bound():
    assert 'kind "procedure" with decision "heading"' in RS.POC_RULES
    assert "65,536 bytes" in RS.POC_RULES and "x is east and y is north" in RS.POC_RULES


def test_a_stream_can_start_with_the_trusted_guard_adopted(tmp_path):
    RS.run_stream(1, 1, tmp_path / "out", max_steps=3, empty_library=True, trusted_heading_guard=GUARD)
    lib = Library(tmp_path / "out" / "library")
    adopted = lib.adopted("heading", "check")
    assert len(adopted) == 1
    entry = lib.entry(adopted[0])
    assert entry["source"] == GUARD.read_text() and entry["origin"]["trusted_heading_guard"] == GUARD.name
    assert lib.adopted("heading", "procedure") == [] and lib.adopted("report", "check") == []


def test_the_learning_input_cap_is_the_protocols_unless_a_run_sets_another(tmp_path):
    p = G.GatewayProposer("http://127.0.0.1:1", "m", tmp_path, "s", count_tokens=lambda m: 1)
    assert p.input_cap == T.INPUT_TOKEN_CAP == 4096
    q = G.GatewayProposer("http://127.0.0.1:1", "m", tmp_path, "s", count_tokens=lambda m: 1, input_cap=16384)
    assert q.input_cap == 16384


def test_the_command_line_carries_the_step2_settings(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(RS, "run_stream", lambda *a, **k: seen.update(k) or {"summary": {
        "completed": 0, "episodes": 0, "goal_met_of_completed": 0, "joined_of_completed": 0, "model_calls": 0}})
    RS.main(["--out", str(tmp_path / "o"), "--learn-model", "M", "--learn-input-cap", "16384", "--max-attempts", "4",
             "--trusted-heading-guard", str(GUARD), "--learn-effort", "xhigh"])
    assert seen["max_attempts"] == 4 and seen["trusted_heading_guard"] == GUARD
    assert seen["learn"]["input_cap"] == 16384 and seen["learn"]["effort"] == "xhigh"


def outcome(joined, reported):
    return {"goal_met": joined and reported, "joined": joined, "report_correct": reported}


def test_heading_only_learning_ignores_report_only_misses(tmp_path):
    co = G.Coordinator(Library(tmp_path / "library"), tmp_path, lambda *_: "x", lambda _: True, lambda *_: None,
                       learn_decisions=("heading",))
    for i in range(1, 11):
        co.after_episode(SimpleNamespace(episode_id=f"e{i:02d}"), outcome(True, False), True)
    assert co.queue == [] and co.attempts == 0
    assert sum(1 for e in co.events if e["kind"] == "not_queued") == 10
    co.after_episode(SimpleNamespace(episode_id="e11"), outcome(False, False), True)
    assert co.queue == [{"episode": "e11", "failed": "both"}]


def test_heading_only_floor_counts_joins(tmp_path):
    co = G.Coordinator(Library(tmp_path / "library"), tmp_path, lambda *_: "x", lambda _: True, lambda *_: None,
                       learn_decisions=("heading",), attempt_every=1000)
    for i in range(1, 25):
        co.after_episode(SimpleNamespace(episode_id=f"e{i:02d}"), outcome(True, False), True)
    assert not [e for e in co.events if e["kind"] == "floor_breach"]
    both = G.Coordinator(Library(tmp_path / "library2"), tmp_path, lambda *_: "x", lambda _: True, lambda *_: None,
                         attempt_every=1000)
    for i in range(1, 25):
        both.after_episode(SimpleNamespace(episode_id=f"e{i:02d}"), outcome(True, False), True)
    assert [e for e in both.events if e["kind"] == "floor_breach"]        # the default still counts whole goals


def test_a_guarded_stream_learns_heading_only(tmp_path, monkeypatch):
    seen = {}
    class Built(Exception):
        pass

    def spy(*a, **k):
        seen.update(k)
        raise Built                     # stop once the coordinator's settings are known; no episode is flown

    monkeypatch.setattr(G, "Coordinator", spy)
    with pytest.raises(Built):
        RS.run_stream(1, 1, tmp_path / "out", max_steps=2, empty_library=True, trusted_heading_guard=GUARD,
                      learn={"base_url": "http://127.0.0.1:1", "model": "m"})
    assert seen["learn_decisions"] == ("heading",) and seen["extra_rules"] == RS.POC_RULES
    assert seen["max_attempts"] is None


def test_a_skipped_boundary_is_not_made_up_later(tmp_path):
    calls = []
    co = G.Coordinator(Library(tmp_path / "library"), tmp_path, lambda m, meta: calls.append(meta) or "x",
                       lambda _: True, lambda *_: None, max_attempts=4, learn_decisions=("heading",))
    for i in range(1, 41):                 # joins until 15, misses after: attempts only at 20, never at 25 to 40
        co.after_episode(SimpleNamespace(episode_id=f"e{i:02d}"), outcome(i <= 15, False), True)
    assert {m["attempt"] for m in calls} == {1} and co.attempts == 1
    assert [e["ordinal"] for e in co.events if e["kind"] == "attempt_start"] == [20]


def test_a_check_or_pair_is_refused_in_a_heading_only_run_within_the_attempts_calls(tmp_path):
    replies = iter([json.dumps({"kind": "check", "decision": "heading", "conditions": [],
                                "source": "def check(ctx, out):\n    return 'accept'\n",
                                "reason": {"code": "missing_tool", "text": "x"}}),
                    json.dumps({"kind": "pair", "decision": "heading", "reason": {"code": "missing_tool", "text": "x"},
                                "procedure": {"conditions": [], "source": "def decide(ctx):\n    return {}\n"},
                                "check": {"conditions": [], "source": "def check(ctx, out):\n    return 'accept'\n"}}),
                    json.dumps({"kind": "procedure", "decision": "report", "conditions": [],
                                "source": "def decide(ctx):\n    return {}\n",
                                "reason": {"code": "missing_tool", "text": "x"}})])
    lib = Library(tmp_path / "library")
    guard = RS.install_trusted_heading_guard(lib, GUARD)
    co = G.Coordinator(lib, tmp_path, lambda *_: next(replies), lambda _: True, lambda *_: None,
                       learn_decisions=("heading",))
    for i in range(1, 6):
        co.after_episode(SimpleNamespace(episode_id=f"e{i}"), outcome(False, False), True)
    failed = [e for e in co.events if e["kind"] == "attempt_failed"]
    assert len(failed) == 1 and failed[0]["calls"] == 3
    assert lib.adopted("heading", "check") == [guard] and lib.adopted("report", "procedure") == []
    assert set(lib.projection()["entries"]) == {guard}         # nothing was added to the library
