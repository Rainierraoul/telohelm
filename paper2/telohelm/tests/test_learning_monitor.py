"""The monitor's public boundary, using invented outcomes and scripted proposals only."""
from dataclasses import asdict
import ast
import json
from pathlib import Path

import pytest

import telohelm.learning as G
from paper2.experiment2.tests.test_learning import coordinator; from paper2.experiment2.tests.test_learning import Scripted; from paper2.experiment2.tests.test_learning import reply; from paper2.experiment2.tests.test_learning import finish


def test_record_keeps_the_scenario_and_reference_label_out_of_the_public_episode(tmp_path):
    co, _ = coordinator(tmp_path, Scripted(), solvable=lambda scenario: None)
    from types import SimpleNamespace
    scenario = SimpleNamespace(episode_id="invented")
    public = co.monitor.record(scenario, {"goal_met": False, "joined": False, "report_correct": False}, True)
    assert set(asdict(public)) == {"ordinal", "episode", "goal_met", "failed", "helped"}
    assert public.episode == "invented" and public.failed == "both"
    assert co.monitor.completed[0].scenario is scenario and co.monitor.completed[0].solvable is None
    with pytest.raises(AttributeError):
        public.goal_met = True


def test_a_verdict_has_only_allowed_feedback_and_the_host_keeps_the_detailed_result(tmp_path):
    co, _ = coordinator(tmp_path, Scripted())
    private = {"replay": {"passed": False, "wrong_accepted_any": True, "right_short": False,
                           "wrong_cases": 734, "secret_case": "private-sentinel"}}
    co.monitor.check_replay = lambda *args: (False, private)
    boundary = G.PublicEpisode(5, "invented", False, "both", False)
    verdict = co.monitor.submit(1, 1, json.loads(reply(kind="check", source="def check(ctx, out):\n    return 'accept'\n")),
                                boundary, [])
    assert not verdict.passed
    assert set(verdict.feedback) == {"attempt", "kind", "decision", "adopted", "stage", "why"}
    assert verdict.feedback["why"] == ("the check accepted at least one stored wrong answer",)
    assert "private-sentinel" not in repr(verdict.feedback) and "734" not in repr(verdict.feedback)
    with pytest.raises(TypeError):
        verdict.feedback["stage"] = "changed"
    assert co.monitor.host_event(verdict.event_index)["detail"] == private


def test_the_next_request_uses_the_same_public_feedback_without_private_details(tmp_path):
    proposer = Scripted(reply(kind="check", source="def check(ctx, out):\n    return 'accept'\n"))
    co, _ = coordinator(tmp_path, proposer)
    co.monitor.check_replay = lambda *args: (False, {"replay": {"passed": False, "wrong_accepted_any": True,
                                                                "secret": "private-sentinel"}})
    finish(co, 5)
    messages = co.messages([])
    assert messages[0] == ("system", G.PROPOSER_RULES)
    body = json.loads(messages[1][1])
    assert body["previous_attempts"][-1]["why"] == ["the check accepted at least one stored wrong answer"]
    assert "private-sentinel" not in json.dumps(messages)
    assert co.monitor.events[-1]["detail"]["replay"]["secret"] == "private-sentinel"


def test_proposal_methods_do_not_read_cases_scenarios_or_the_library_writer():
    module = ast.parse((Path(G.__file__)).read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "Coordinator")
    proposal = {"messages", "_submit", "_attempt_candidates", "_remember_verdict", "_scope_errors"}
    forbidden = {"corpus", "scenario", "solvable", "library", "completed", "_attempt_trial", "_regression_set"}
    for method in cls.body:
        if isinstance(method, ast.FunctionDef) and method.name in proposal:
            assert not {n.attr for n in ast.walk(method) if isinstance(n, ast.Attribute)} & forbidden
