"""Slice 1 of the task pack: the tanker's public contract (study2/packs/; COMPONENT-MAP-COMPARISON.md).

What Telohelm is told must be byte for byte what it was before the move (D42), the pack must hold no handle to the
world, the monitor, the reference pilot or the prior, and the core's text must work for another pack.
"""
import ast
from dataclasses import replace
from pathlib import Path

import telohelm.learning as G
import telohelm.local_model as L
import telohelm.qualification.store as H
import telohelm.taskpacks.tanker.public as T
from telohelm.taskpacks.tanker.public import TANKER
import telohelm.runtime.delivery as D
import telohelm.runtime.procedures as P

PACKS = tuple(next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / name for name in ['paper2/telohelm/src/telohelm/contracts/__init__.py', 'paper2/telohelm/src/telohelm/contracts/decisions.py', 'paper2/telohelm/src/telohelm/contracts/evidence.py', 'paper2/telohelm/src/telohelm/contracts/example_view.py', 'paper2/telohelm/src/telohelm/contracts/goal_parts.py', 'paper2/telohelm/src/telohelm/taskpacks/tanker/public.py', 'paper2/telohelm/src/telohelm/taskpacks/tanker/context.py', 'paper2/telohelm/src/telohelm/taskpacks/tanker/dynamics.py', 'paper2/telohelm/src/telohelm/taskpacks/tanker/examples.py', 'paper2/telohelm/src/telohelm/taskpacks/tanker/execution.py', 'paper2/telohelm/src/telohelm/taskpacks/tanker/pack.py', 'paper2/telohelm/src/telohelm/taskpacks/tanker/prediction.py', 'paper2/telohelm/src/telohelm/taskpacks/tanker/tools.py', 'paper2/telohelm/src/telohelm/contracts/tools.py'])
FORBIDDEN = ('telohelm.taskpacks.tanker.world', 'paper2.experiment2.outcomecases', 'paper2.experiment2.devcases', 'telohelm.qualification', "world", "outcomecases",
             "devcases", "hidden_set", "pilot", "phase5", "streams", "episode")


def test_the_core_and_the_pack_together_say_exactly_what_was_said_before():
    assert G.render_learner_rules(TANKER, G.learner_limits()) == G.PROPOSER_RULES
    assert L.SYSTEM == TANKER.flight_system and L.ASK == dict(TANKER.flight_ask)
    assert P.CONTEXT_FIELDS == TANKER.context_fields and P.OUTPUT_FIELDS == TANKER.output_fields
    assert P.DECISIONS == frozenset(TANKER.decisions) | {"help"} and P.PAIR_DECISIONS == TANKER.pair_decisions
    assert D.DECISION_FIELDS == dict(TANKER.decision_fields) and H.DECISIONS == TANKER.decisions


def test_the_pack_states_the_worlds_own_tolerances():
    import telohelm.taskpacks.tanker.world.episode as E
    assert T.JOIN_TOLERANCE_NM == E.JOIN_TOLERANCE and T.REPORT_TOLERANCE_NM == E.REPORT_TOLERANCE


def test_the_public_contract_imports_nothing_from_the_world_monitor_pilot_or_prior():
    for path in PACKS:
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [a.name for a in node.names]
            for name in names:
                assert not any(part in FORBIDDEN for part in name.split(".")), f"{path.name} imports {name}"


def test_the_cores_text_carries_another_packs_task_and_none_of_the_tankers():
    toy = replace(TANKER, pack_id="toy-tracking", decisions=("move", "signal"),
                  learner_purpose="keep a cart near a moving target and signal once when it is close",
                  learner_actions='- decision "move": {"speed": <number>}.\n- decision "signal": {"signal": true} or {}.\n',
                  learner_goal="The goal is met when the cart stays within 1 unit of the target and signals once.\n",
                  learner_tools=lambda limits: "",
                  learner_trial_rule="move procedure passes if the cart stays close; a signal procedure passes if it "
                                     "signals once at the right time.",
                  learner_attribute_example="ctx.position or out.speed",
                  learner_decision_choices='"move" or "signal"')
    text = G.render_learner_rules(toy, G.learner_limits())
    assert "keep a cart near a moving target" in text and '"decision": "move" or "signal"' in text
    assert not any(word in text for word in ("tanker", "Wedgetail", "heading_deg", "receiver aircraft", "predict"))
    assert TANKER.fingerprint(G.learner_limits()) != toy.fingerprint(G.learner_limits())


def test_the_contract_cannot_be_changed_after_it_is_made_nor_through_what_it_was_given():
    import pytest
    from telohelm.contracts import PublicContract
    with pytest.raises(TypeError):
        TANKER.decision_fields["heading"] = frozenset({"anything"})
    with pytest.raises(TypeError):
        TANKER.flight_ask["heading"] = "changed"
    with pytest.raises(AttributeError):
        TANKER.decision_fields["heading"].add("anything")
    fields = {"heading": {"heading_deg"}}
    ask = {"heading": "a question"}
    made = replace(TANKER, decision_fields=fields, flight_ask=ask, decisions=["heading"])
    fields["heading"].add("leak")
    fields["report"] = {"x"}
    ask["heading"] = "changed"
    assert dict(made.decision_fields) == {"heading": frozenset({"heading_deg"})}
    assert dict(made.flight_ask) == {"heading": "a question"} and made.decisions == ("heading",)
    with pytest.raises(TypeError):
        PublicContract(**{**{k: getattr(TANKER, k) for k in TANKER.__dataclass_fields__}, "learner_goal": None})
