"""The learner's task and its private trial task are one configuration.

These controls exercise configuration and saved-request construction without executing a model or a trial.
"""
import json
from dataclasses import replace

import pytest

import telohelm.learning as G
from telohelm.contracts.goal_parts import GoalPart; from telohelm.contracts.goal_parts import GoalParts
from telohelm.taskpacks.tanker.pack import TANKER_PACK; from telohelm.taskpacks.tanker.pack import TANKER_REDUCED_PACK
from telohelm.runtime.library import Library


def coordinator(tmp_path, **kwargs):
    def no_call(*args):
        pytest.fail("construction must not make a model or tool call")
    return G.Coordinator(Library(tmp_path / "library"), tmp_path / "work", no_call, no_call, no_call, **kwargs)


def test_omitted_pack_arguments_preserve_the_default_rules_and_monitor_task(tmp_path):
    co = coordinator(tmp_path)
    assert co.rules == G.PROPOSER_RULES
    assert co.pack is co.monitor.pack is TANKER_PACK.public
    assert co.goal is co.monitor.goal is TANKER_PACK.goal
    assert co.flight_pack is co.monitor.flight_pack is TANKER_PACK
    assert co.learn_decisions == TANKER_PACK.public.decisions


def test_equal_legacy_arguments_resolve_to_the_selected_task_objects(tmp_path):
    public, goal = replace(TANKER_PACK.public), replace(TANKER_PACK.goal)
    assert public is not TANKER_PACK.public and goal is not TANKER_PACK.goal
    co = coordinator(tmp_path, pack=public, goal=goal)
    assert co.pack is co.monitor.pack is TANKER_PACK.public
    assert co.goal is co.monitor.goal is TANKER_PACK.goal
    assert co.rules == G.PROPOSER_RULES


def test_selecting_a_reduced_task_alone_also_selects_its_public_request_fields(tmp_path):
    co = coordinator(tmp_path, flight_pack=TANKER_REDUCED_PACK, procedures_only=True)
    body = json.loads(co.messages([{"episode": "e", "failed": "join"}])[-1][1])
    assert set(body["context_fields"]) == TANKER_REDUCED_PACK.public.context_fields
    assert co.pack is co.monitor.pack is TANKER_REDUCED_PACK.public
    assert co.goal is co.monitor.goal is TANKER_REDUCED_PACK.goal
    assert co.flight_pack is co.monitor.flight_pack is TANKER_REDUCED_PACK


def test_selecting_a_reduced_task_alone_keeps_the_learned_check_refusal(tmp_path):
    with pytest.raises(ValueError, match="procedure-only"):
        coordinator(tmp_path, flight_pack=TANKER_REDUCED_PACK)
    assert not (tmp_path / "work").exists()


def test_a_selected_goal_is_shared_by_feedback_and_private_monitoring(tmp_path):
    goal = GoalParts(parts=(GoalPart("joined", "not close", {"heading"}),), all_failed="all")
    task = replace(TANKER_PACK, goal=goal)
    co = coordinator(tmp_path, flight_pack=task)
    assert co.goal is co.monitor.goal is goal
    assert co.goal.failed({"joined": False}) == co.monitor.goal.failed({"joined": False}) == "not close"
    assert co.goal.concerns("not close") == {"heading"}


@pytest.mark.parametrize("mismatch", ("contract", "goal", "reduced_contract"))
def test_mismatches_fail_before_rendering_writing_or_calls(tmp_path, monkeypatch, mismatch):
    def forbidden_render(*args):
        pytest.fail("a mismatched configuration must fail before rules are rendered")
    monkeypatch.setattr(G, "render_learner_rules", forbidden_render)
    options = {
        "contract": {"pack": replace(TANKER_PACK.public, pack_id="different-task")},
        "goal": {"goal": replace(TANKER_PACK.goal, all_failed="different-failure")},
        "reduced_contract": {"flight_pack": TANKER_REDUCED_PACK, "pack": TANKER_PACK.public,
                             "procedures_only": True},
    }[mismatch]
    with pytest.raises(ValueError, match="must match the selected flight pack"):
        G.Coordinator(object(), tmp_path / "work", None, None, None, **options)
    assert list(tmp_path.iterdir()) == []
