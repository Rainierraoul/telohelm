"""Once-per-goal issuance belongs to the selected task. Short scripted execution fixtures, no model calls."""
from dataclasses import replace
import json

import pytest

from telohelm.contracts.decisions import DecisionRoles
from telohelm.taskpacks.tanker.pack import TANKER_PACK
import telohelm.runtime.episode_loop as loop
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.library import Library
from telohelm.runtime.trusted_log import TrustedLog
from .test_pack_decisions import scenario
import telohelm.taskpacks.tanker.world.episode as E


def task():
    public = replace(TANKER_PACK.public, decisions=("heading", "signal"), pair_decisions=(),
                     decision_fields={"heading": {"heading_deg", "help"},
                                      "signal": {"report", "report_trigger_nm"}})
    roles = DecisionRoles(order=("heading", "signal"), flying="heading", once_per_goal={"signal"},
                          needs={"heading": (), "signal": ()}, completion_fields={"signal": {"report"}})
    return replace(TANKER_PACK, public=public, roles=roles)


def fly(tmp_path, monkeypatch, *, first=None, refuse_first=False, changed_goal=False):
    sc = scenario()
    if changed_goal:
        sc = replace(sc, goals=(*sc.goals, (2, E.Goal(2, "SHELL 2", 20.0))))
    seen = []
    def choose(decision, ctx, library, **kwargs):
        seen.append((decision, ctx["t"], ctx["goal_version"]))
        if decision == "heading":
            part = {"heading_deg": 400 if refuse_first and ctx["t"] == 0 else 90}
        elif first is not None and ctx["t"] == 0:
            part = dict(first)
        else:
            part = {"report": "Wedgetail, aircraft 2, 20 NM from " + ctx["goal_tanker"]}
        return {"use": "scripted-fixture", "output": part, "tried": [], "checks": []}
    monkeypatch.setattr(loop, "choose", choose)
    log = TrustedLog(tmp_path / "episode.log", tmp_path / "episode.heads", "once-fixture")
    result = loop.run_episode(sc, Library(tmp_path / "library"), loop.NoModels(), log, Ledger(),
                              max_steps=4, pack=task())
    rows = [json.loads(line)["payload"] for line in (tmp_path / "episode.log").read_text().splitlines()]
    return [s for s in seen if s[0] == "signal"], rows, result


def test_a_renamed_once_per_goal_decision_is_not_asked_after_issuing_its_action(tmp_path, monkeypatch):
    asked, _, result = fly(tmp_path, monkeypatch)
    assert asked == [("signal", 0, 1)]
    assert len(result["outcome"]["reports"]) == 1


@pytest.mark.parametrize("waiting", ({}, {"report_trigger_nm": 1.0}, {"help": {"text": "outside signal"}}))
def test_waiting_setup_and_foreign_fields_do_not_complete_the_decision(tmp_path, monkeypatch, waiting):
    asked, rows, _ = fly(tmp_path, monkeypatch, first=waiting)
    assert asked == [("signal", 0, 1), ("signal", 1, 1)]
    signal = [r for r in rows if r.get("kind") == "decision" and r.get("decision") == "signal"]
    assert "report" not in signal[0]["part"] and "report" in signal[1]["part"]


def test_whole_action_refusal_allows_the_once_decision_to_retry(tmp_path, monkeypatch):
    asked, _, result = fly(tmp_path, monkeypatch, refuse_first=True)
    assert asked == [("signal", 0, 1), ("signal", 1, 1)]
    assert len(result["outcome"]["reports"]) == 1


def test_a_changed_goal_allows_one_new_action(tmp_path, monkeypatch):
    asked, _, result = fly(tmp_path, monkeypatch, changed_goal=True)
    assert asked == [("signal", 0, 1), ("signal", 2, 2)]
    assert [r["goal_version"] for r in result["outcome"]["reports"]] == [1, 2]


@pytest.mark.parametrize("completion", ({}, {"signal": set()}, {"signal": "report"},
                                       {"heading": {"heading_deg"}}, {"signal": {""}}))
def test_roles_refuse_missing_empty_or_misassigned_completion_fields(completion):
    with pytest.raises(ValueError):
        replace(task().roles, completion_fields=completion)


def test_completion_fields_are_copied_and_immutable():
    fields = {"signal": {"report"}}
    roles = replace(task().roles, completion_fields=fields)
    fields["signal"].add("report_trigger_nm")
    assert roles.completion_fields["signal"] == {"report"}
    with pytest.raises(TypeError):
        roles.completion_fields["signal"] = frozenset()
    with pytest.raises(AttributeError):
        roles.completion_fields["signal"].add("other")


def test_completion_fields_outside_decision_ownership_fail_before_the_world_starts(tmp_path, monkeypatch):
    def forbidden_world(*args):
        pytest.fail("bad task configuration must fail before constructing the world")
    monkeypatch.setattr(E, "Episode", forbidden_world)
    pack = task()
    pack = replace(pack, roles=replace(pack.roles, completion_fields={"signal": {"heading_deg"}}))
    with pytest.raises(ValueError, match="must belong"):
        loop.run_episode(scenario(), None, loop.NoModels(), None, Ledger(), pack=pack)
