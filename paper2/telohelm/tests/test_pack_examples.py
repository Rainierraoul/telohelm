"""Task example policy controls using invented authenticated sensor records. No model, world or learned task."""
from dataclasses import replace
import json

import pytest

import telohelm.examples as X; import telohelm.learning as G
from telohelm.contracts.decisions import DecisionRoles
from telohelm.contracts.example_view import ExamplePolicy
from telohelm.contracts.goal_parts import GoalPart; from telohelm.contracts.goal_parts import GoalParts
from telohelm.taskpacks.tanker.examples import TANKER_EXAMPLES
from telohelm.taskpacks.tanker.pack import TANKER_PACK
from telohelm.runtime.library import Library
from telohelm.runtime.trusted_log import TrustedLog


class SensorBuilder:
    def __init__(self, briefing):
        assert briefing == {"base": 7}
    def update(self, bundle):
        return {"t": bundle["t"], "reading": bundle["meter"] + 7}


POLICY = ExamplePolicy(
    rules="Examples contain your recorded sensor reading and submitted move, with its acceptance.",
    points=lambda decisions, failed, outcome, goal: ([max(p["t"] for p in decisions)], []),
    project=lambda ctx: (dict(ctx), {"readings_shown": 1}),
    details=lambda t, outcome, actions, sources, steps, changes: {"move_applied": steps[t]["accepted"]})


def sensor_records(tmp_path):
    root = tmp_path / "sensor-run"
    root.mkdir()
    broker = TrustedLog(root / "broker.log", root / "broker.heads", "invented")
    episode = TrustedLog(root / "episode.log", root / "episode.heads", "invented")
    broker.append({"kind": "briefing", "briefing": {"base": 7}})
    for t in range(2):
        ctx = {"t": t, "reading": t + 7}
        broker.append({"kind": "bundle", "t": t, "bundle": {"t": t, "meter": t}})
        broker.append({"kind": "decision", "t": t, "decision": "move", "source": "procedure"})
        broker.append({"kind": "action", "t": t, "action": {"step": 1}, "field_writers": {"step": "move"}})
        broker.append({"kind": "world_step", "t": t, "accepted": True})
        episode.append({"kind": "decision", "t": t, "decision": "move", "context_sha256": X._digest(ctx)})
    (root / "summary.json").write_text(json.dumps({"outcome": {"private_sentinel": "not public"}}))
    return root


def sensor_task(policy=POLICY):
    public = replace(TANKER_PACK.public, pack_id="sensor-fixture", decisions=("move",), pair_decisions=(),
                     decision_fields={"move": {"step"}}, context_fields={"t", "reading"}, output_fields={"step"},
                     learner_purpose="keep the sensor reading within its goal", learner_actions='move: {"step": 1}',
                     learner_goal="The reading stays in its requested interval.", learner_tools=lambda limits: "",
                     learner_trial_rule="The goal is met.", learner_attribute_example="ctx.reading or out.step",
                     learner_decision_choices='"move"')
    goal = GoalParts((GoalPart("inside", "outside", {"move"}),), "all")
    roles = DecisionRoles(("move",), "move", frozenset(), {"move": ()})
    return replace(TANKER_PACK, public=public, goal=goal, roles=roles, context_builder=SensorBuilder,
                   example_policy=policy)


def test_a_task_without_radar_or_radio_can_project_its_own_verified_examples(tmp_path):
    sensor_records(tmp_path)
    done = [G.Completed(1, "sensor-run", None, False, "outside", True)]
    shown, receipts, omitted = X.select(tmp_path, done, [{"episode": "sensor-run"}], goal=sensor_task().goal,
                                        context_builder=SensorBuilder, policy=POLICY)
    assert not omitted and len(shown) == len(receipts) == 1
    assert shown[0]["context"] == {"t": 1, "reading": 8}
    assert shown[0]["submitted"] == {"step": {"value": 1, "by": "stored procedure"}}
    assert shown[0]["move_applied"] is True and "report_transmitted" not in shown[0]
    assert receipts[0]["readings_shown"] == 1
    assert "private_sentinel" not in json.dumps(shown)


def test_the_learner_gets_the_selected_tasks_exact_example_rules(tmp_path):
    co = G.Coordinator(Library(tmp_path / "library"), tmp_path, None, lambda _: True, None,
                       examples=True, flight_pack=sensor_task())
    system = co.messages([])[0][1]
    assert system.endswith("\n" + POLICY.rules) and X.RULES not in system
    assert not any(word in system for word in ("tanker", "radar", "radio", "heading_deg"))


def test_missing_task_policy_is_refused_when_examples_are_requested(tmp_path):
    pack = sensor_task(None)
    with pytest.raises(ValueError, match="must supply an example policy"):
        G.Coordinator(object(), tmp_path / "absent", None, None, None, examples=True, flight_pack=pack)
    assert not (tmp_path / "absent").exists()
    co = G.Coordinator(Library(tmp_path / "library"), tmp_path, None, lambda _: True, None, flight_pack=pack)
    with pytest.raises(ValueError, match="no example policy"):
        co.monitor.examples(1, [])


@pytest.mark.parametrize("target", ("context", "receipt"))
def test_task_annotations_cannot_overwrite_core_evidence(tmp_path, target):
    root = sensor_records(tmp_path)
    policy = (replace(POLICY, details=lambda *args: {"context": {"invented": True}}) if target == "context" else
              replace(POLICY, project=lambda ctx: (dict(ctx), {"full_context_sha256": "forged"})))
    with pytest.raises(X.ExampleError, match="may not overwrite"):
        X.build(root, "sensor-run", [1], {"goal_met": False}, {}, context_builder=SensorBuilder, policy=policy)


def test_a_task_may_select_only_seconds_that_were_recorded(tmp_path):
    root = sensor_records(tmp_path)
    policy = replace(POLICY, points=lambda *args: ([999], []))
    with pytest.raises(X.ExampleError, match="unrecorded second"):
        X.points(root, "outside", {}, sensor_task().goal, policy=policy)


def test_policy_setup_is_frozen_and_refuses_missing_callbacks():
    with pytest.raises(Exception):
        POLICY.rules = "changed"
    with pytest.raises(ValueError):
        replace(POLICY, rules="")
    with pytest.raises(TypeError):
        replace(POLICY, project=None)
    assert TANKER_PACK.example_policy is TANKER_EXAMPLES and X.RULES == TANKER_EXAMPLES.rules
