"""Simulator and protected-trial interfaces with an authored two-step task and stand-in action selection.

No model, generated procedure or tanker reference case is executed by these controls.
"""
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

import telohelm.learning as G
from telohelm.contracts.decisions import DecisionRoles
from telohelm.contracts.goal_parts import GoalPart; from telohelm.contracts.goal_parts import GoalParts
from telohelm.taskpacks.tanker.pack import TANKER_PACK; from telohelm.taskpacks.tanker.pack import TANKER_REDUCED_PACK
import telohelm.runtime.episode_loop as L
from telohelm.runtime.environment import Environment; from telohelm.runtime.environment import environment_for
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.library import Library
from telohelm.runtime.trusted_log import TrustedLog
from telohelm.taskpacks.tanker.world.tanker_environment import TANKER_ENVIRONMENT


class Builder:
    def __init__(self, briefing):
        assert briefing == {"base": 5}
    def update(self, bundle):
        return {"t": bundle["t"], "goal_version": 1, "value": bundle["reading"] + 5}


class TwoSteps:
    def __init__(self, scenario):
        self.t = 0
        self.record = SimpleNamespace(actions=[])
    def reset(self):
        return {"briefing": {"base": 5}, "bundle": {"t": 0, "reading": 0}}
    def step(self, action):
        self.record.actions.append({"actions": dict(action), "accepted": True})
        self.t += 1
        return None if self.t == 2 else {"t": self.t, "reading": self.t}
    def outcome(self):
        return {"goal_met": self.t == 2, "inside": self.t == 2, "settled": True,
                "end_reason": "done" if self.t == 2 else None, "ended_at": self.t,
                "action_steps": len(self.record.actions), "alternatives": [], "help_requests": []}


def judge(decision, sha, outcome, records):
    used = [p for p in records if p.get("decision") == decision and p.get("method") == sha
            and p.get("goal_version") == 1 and p.get("part") == {"step": 1}]
    return {"passed": bool(outcome["goal_met"] and outcome["end_reason"] and used), "used_seconds": len(used)}


def task(environment=True):
    public = replace(TANKER_PACK.public, pack_id="two-step-fixture", decisions=("move",), pair_decisions=(),
                     decision_fields={"move": {"step"}}, context_fields={"t", "goal_version", "value"},
                     output_fields={"step"})
    env = Environment(public.pack_id, public.schema_version, TwoSteps, judge, 2) if environment else None
    return replace(TANKER_PACK, public=public, goal=GoalParts((GoalPart("inside", "outside", {"move"}),), "all"),
                   roles=DecisionRoles(("move",), "move", (), {"move": ("value",)}),
                   context_builder=Builder, part_error=lambda part: None, environment=env)


def test_legacy_and_reduced_tasks_keep_the_same_tanker_environment():
    assert environment_for(TANKER_PACK) is environment_for(TANKER_REDUCED_PACK) is TANKER_ENVIRONMENT
    assert TANKER_ENVIRONMENT.max_steps == 3601 and TANKER_ENVIRONMENT.tools is None
    assert G.component_trial is TANKER_ENVIRONMENT.trial


@pytest.mark.parametrize("change", ({"task_id": "other"}, {"schema_version": 2}, {"tools": None}))
def test_environment_identity_and_explicit_new_task_tools_are_required(change):
    pack = task()
    with pytest.raises(ValueError):
        environment_for(replace(pack, environment=replace(pack.environment, **change)))
    with pytest.raises(ValueError, match="supply an environment"):
        environment_for(task(environment=False))


@pytest.mark.parametrize("change", ({"max_steps": 0}, {"max_steps": True}, {"create": None}, {"trial": None},
                                    {"schema_version": True}, {"tools": {"bad": object()}}))
def test_bad_host_environment_configuration_is_refused(change):
    with pytest.raises(ValueError):
        replace(task().environment, **change)


def test_the_tool_table_is_copied_and_immutable():
    registry = {}
    env = replace(task().environment, tools=registry)
    registry["later"] = object()
    assert dict(env.tools) == {}
    with pytest.raises(TypeError):
        env.tools["later"] = object()


def test_model_request_construction_neither_calls_nor_exposes_environment_handles(tmp_path):
    pack = task()
    def forbidden(*args):
        pytest.fail("request construction must not call private environment callbacks")
    pack = replace(pack, environment=replace(pack.environment, create=forbidden, trial=forbidden))
    lib = Library(tmp_path / "library", pack=pack.public)
    co = G.Coordinator(lib, tmp_path / "work", None, None, None, flight_pack=pack)
    messages = co.messages([{"episode": "two-step", "failed": "outside"}])
    text = json.dumps(messages)
    assert "test_pack_environment" not in text and "forbidden" not in text and "TwoSteps" not in text
    assert "environment" not in json.loads(messages[-1][1])


def test_another_simulator_runs_through_the_same_loop_without_constructing_a_tanker(tmp_path, monkeypatch):
    pack = task()
    def forbidden(*args):
        pytest.fail("the custom environment must not create a tanker world")
    monkeypatch.setattr(L.E, "Episode", forbidden)
    lib = Library(tmp_path / "library", pack=pack.public)
    log = TrustedLog(tmp_path / "episode.log", tmp_path / "episode.heads", "fixture")
    result = L.run_episode(SimpleNamespace(episode_id="two-step"), lib, L.NoModels(), log, Ledger(), pack=pack)
    assert result["outcome"]["goal_met"] and result["outcome"]["end_reason"] == "done"
    assert result["outcome"]["action_steps"] == 2
    records = [json.loads(line)["payload"] for line in (tmp_path / "episode.log").read_text().splitlines()]
    starts = [p for p in records if p["kind"] == "episode_start"]
    assert len(starts) == 1 and starts[0]["environment"] == pack.environment.record()
    assert len([p for p in records if p["kind"] == "decision" and p["decision"] == "move"]) == 2


@pytest.mark.parametrize("step_limit, expected", ((None, True), (1, False)))
def test_private_trials_use_the_same_environment_tools_judge_bound_and_durable_evidence(tmp_path, monkeypatch,
                                                                                    step_limit, expected):
    pack = task()
    lib = Library(tmp_path / "library", pack=pack.public)
    source = 'def decide(ctx):\n    return {"step": 1}\n'
    sha = lib.add({"kind": "procedure", "decision": "move", "conditions": [], "source": source, "origin": {}})
    seen = []
    def selected(decision, context, shadow, **kwargs):
        assert context["value"] in (5, 6)
        assert shadow.pack is pack.public and not kwargs["broker"]._tools
        seen.append(context["t"])
        return {"use": sha, "output": {"step": 1}, "tried": [], "checks": []}
    monkeypatch.setattr(L, "choose", selected)
    co = G.Coordinator(lib, tmp_path / "work", None, lambda scenario: True, None,
                       flight_pack=pack, shadow_max_steps=step_limit)
    co.workdir.mkdir()
    completed = G.Completed(1, "two-step", SimpleNamespace(episode_id="two-step"), False, "outside", True)
    passed, detail = co.monitor.trial(sha, None, "move", [completed])
    assert passed is expected and seen == ([0, 1] if expected else [0])
    result = detail["trial"][0]
    assert result["component"]["passed"] is expected
    assert result["candidate_source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    from pathlib import Path
    evidence = Path(result["evidence_dir"])
    assert json.loads((evidence / "result.json").read_text()) == result
    settings = json.loads((evidence / "settings.json").read_text())
    assert settings["max_steps"] == (2 if step_limit is None else 1)
    assert settings["environment"] == pack.environment.record()
    assert lib.adopted("move", "procedure") == []  # trials do not themselves adopt
