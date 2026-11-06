"""Slice 2(b): how the decisions behave each second comes from the task pack (study2/packs/decisions.py, TANKER_ROLES).

The tanker's values must equal the inline ones they replaced in runtime/episode_loop.py and runtime/chooser.py (as at
feb18eb), and the core must follow the roles it is given.
"""
import math

import pytest

from telohelm.contracts.decisions import DecisionRoles
from telohelm.taskpacks.tanker.public import TANKER
from telohelm.taskpacks.tanker.execution import TANKER_ROLES
import telohelm.runtime.chooser as C
from telohelm.runtime.episode_loop import NoModels; from telohelm.runtime.episode_loop import run_episode
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.library import Library
from telohelm.runtime.trusted_log import TrustedLog
import telohelm.taskpacks.tanker.world.episode as E
import telohelm.taskpacks.tanker.world.physics as P

INLINE_NEEDED = {"heading": ("report_x", "report_y"), "report": ("report_x", "report_y", "observed_range_nm"),
                 "help": ()}


def test_the_tanker_roles_are_the_inline_values_they_replaced():
    assert dict(TANKER_ROLES.needs) == INLINE_NEEDED and dict(C.NEEDED) == INLINE_NEEDED
    assert TANKER_ROLES.asked() == ("heading", "report") and TANKER_ROLES.asked({"report"}) == ("heading",)
    assert TANKER_ROLES.flying == "heading" and TANKER_ROLES.order == TANKER.decisions


def test_roles_refuse_to_be_inconsistent_or_changed():
    with pytest.raises(ValueError):
        DecisionRoles(order=("move",), flying="steer", once_per_goal=(), needs={"move": ()})
    with pytest.raises(ValueError):
        DecisionRoles(order=("move",), flying="move", once_per_goal={"signal"}, needs={"move": ()})
    with pytest.raises(ValueError):
        DecisionRoles(order=("move", "signal"), flying="move", once_per_goal=(), needs={"move": ()})
    with pytest.raises(TypeError):
        TANKER_ROLES.needs["heading"] = ()


def test_the_chooser_asks_for_the_fields_the_roles_say_a_decision_needs(tmp_path):
    lib = Library(tmp_path / "empty")
    ctx = {"t": 0, "report_x": None, "report_y": None, "observed_range_nm": None}
    assert C.choose("heading", ctx, lib)["climb"] == "missing_or_stale_evidence"
    assert C.choose("heading", ctx, lib, needs={"heading": ()})["climb"] == "missing_tool"


def scenario():
    rcv = P.ReceiverState(0.0, 0.0, 0.0, math.radians(90), math.radians(90), P.kt(450.0), P.turn_rate(450.0))
    tankers = (E.Tanker("SHELL 1", P.Orbit(40.0, 10.0, 8.0, P.kt(280.0), math.radians(200))),
               E.Tanker("SHELL 2", P.Orbit(40.0, 60.0, 8.0, P.kt(280.0), 0.0)))
    return E.Scenario(episode_id="roles-test", receiver=rcv, wind=(0.0, 0.0), tankers=tankers,
                      goals=((0, E.Goal(1, "SHELL 1", 20.0)),))


def test_the_episode_loop_asks_only_the_decisions_the_roles_name(tmp_path):
    (tmp_path / "protected").mkdir()
    log = TrustedLog(tmp_path / "runner.log", tmp_path / "protected" / "heads.jsonl", "roles-run")
    heading_only = DecisionRoles(order=("heading",), flying="heading", once_per_goal=(),
                                 needs={"heading": ("report_x", "report_y"), "help": ()})
    from dataclasses import replace
    from telohelm.taskpacks.tanker.pack import TANKER_PACK
    run_episode(scenario(), Library(tmp_path / "empty"), NoModels(), log, Ledger(), max_steps=4,
                pack=replace(TANKER_PACK, roles=heading_only))
    import json
    asked = [json.loads(line)["payload"]["decision"] for line in (tmp_path / "runner.log").read_text().splitlines()
             if json.loads(line)["payload"].get("kind") == "decision"]
    assert asked == ["heading"] * 4


def test_the_ladder_asks_for_help_on_the_stale_evidence_the_task_names():
    from telohelm.contracts.decisions import HelpEvidence
    from telohelm.runtime.escalation import FlightLadder; from telohelm.runtime.escalation import part_error
    battery = HelpEvidence(age_field="battery_age_s", stale_after_s=20.0, text="the battery reading is old")
    ladder = FlightLadder(None, help_evidence=battery)
    assert ladder.help({"t": 0, "deadline": None, "battery_age_s": 5}, {}, False) is None
    assert ladder.help({"t": 0, "deadline": None, "battery_age_s": 50}, {}, False) == {
        "reason": "missing_or_stale_evidence", "text": "the battery reading is old"}
    tanker = FlightLadder(None)
    assert tanker.help({"t": 0, "deadline": None, "report_age_s": 11}, {}, False)["text"] == (
        "the goal tanker's newest report is missing or over 10 s old")
    assert part_error("heading", {"heading_deg": 400}) == "heading out of range"
    assert part_error("move", {"speed": -1}, task_rules=lambda p: "too slow" if p["speed"] < 0 else None) == "too slow"
    assert part_error("move", {"speed": 1, "help": "x"}, task_rules=lambda p: None).startswith("help must")
