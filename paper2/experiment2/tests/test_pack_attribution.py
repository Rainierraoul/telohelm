"""Task-aware control attribution from invented authenticated records, without a world or model run."""
from dataclasses import replace
import json

import pytest

from telohelm.contracts.decisions import DecisionRoles
from telohelm.taskpacks.tanker.pack import TANKER_PACK
import telohelm.runtime.attribution as A
from telohelm.runtime.trusted_log import TrustedLog
from paper2.telohelm.tests.test_attribution import signed


def task():
    public = replace(TANKER_PACK.public, decisions=("drive", "signal"), pair_decisions=(),
                     decision_fields={"drive": {"throttle", "help"}, "signal": {"lamp"}})
    roles = DecisionRoles(order=("drive", "signal"), flying="drive", once_per_goal=(),
                          needs={"drive": (), "signal": ()})
    return replace(TANKER_PACK, public=public, roles=roles)


def renamed(rows):
    rows["reply"].update(decision="drive", part={"throttle": 0.5})
    rows["heading"].update(decision="drive", part={"throttle": 0.5})
    rows["report"].update(decision="signal")
    rows["action"].update(action={"throttle": 0.5, "goal_version": 1},
                          field_writers={"throttle": "drive"}, pending={"drive": False})


def test_another_tasks_model_control_is_attributed_with_its_delivery_provenance(tmp_path):
    paths = signed(tmp_path, renamed)
    result = A.derive(*paths, "inv", pack=task())
    assert result["counts"]["small_model"] == result["K"] == 1 and result["complete"]
    assert result["steps"][0]["set"]["issued"]["reply_sha256"] == "c" * 64
    assert result["steps"][0]["request_id"] == "rq"
    with pytest.raises(A.AttributionError):
        A.derive(*paths, "inv")


def test_explicit_tanker_selection_equals_the_legacy_default(tmp_path):
    paths = signed(tmp_path)
    assert A.derive(*paths, "inv") == A.derive(*paths, "inv", pack=TANKER_PACK)


@pytest.mark.parametrize("fault", ("foreign_field", "wrong_writer", "stale_reply", "wrong_reply_part",
                                   "missing_pending", "wrong_goal_type"))
def test_a_different_task_does_not_relax_provenance_checks(tmp_path, fault):
    def mutate(rows):
        renamed(rows)
        if fault == "foreign_field":
            rows["action"]["action"]["heading_deg"] = 90
            rows["action"]["field_writers"]["heading_deg"] = "drive"
        elif fault == "wrong_writer":
            rows["action"]["field_writers"]["throttle"] = "signal"
        elif fault == "stale_reply":
            rows["reply"]["stale"] = True
        elif fault == "wrong_reply_part":
            rows["reply"]["part"]["throttle"] = 0.6
        elif fault == "missing_pending":
            rows["action"]["pending"] = {"heading": False}
        else:
            rows["heading"]["goal_version"] = True
    with pytest.raises(A.AttributionError):
        A.derive(*signed(tmp_path, mutate), "inv", pack=task())


def test_refused_actions_do_not_gain_control_credit(tmp_path):
    def mutate(rows):
        renamed(rows)
        rows["world"]["accepted"] = False
    result = A.derive(*signed(tmp_path, mutate), "inv", pack=task())
    assert result["counts"]["small_model"] == 0
    assert result["steps"][0]["category"] == "holding_none"
    assert result["steps"][0]["held"] == {"source": "initial"}


def test_a_held_second_keeps_the_selected_control_decisions_original_source(tmp_path):
    def first(rows):
        renamed(rows)
        rows["world"]["ended"] = False
    original, _ = signed(tmp_path / "original", first, endings=())
    events = [json.loads(line)["payload"] for line in original.read_text().splitlines()[1:]]
    log, heads = tmp_path / "two-steps", tmp_path / "heads"
    writer = TrustedLog(log, heads, "two-steps")
    for row in events:
        writer.append(row)
    ctx = "d" * 64
    second = [
        {"kind": "bundle", "bundle": {"t": 1}},
        {"kind": "decision", "decision": "drive", "goal_version": 2, "method": "climb:missing_tool",
         "part": None, "source": "none", "request_id": None, "context_sha256": ctx},
        {"kind": "decision", "decision": "signal", "goal_version": 2, "method": "invented-procedure",
         "part": {}, "source": "procedure", "request_id": None, "context_sha256": ctx},
        {"kind": "action", "action": {}, "field_writers": {}, "pending": {"drive": True}},
        {"kind": "world_step", "accepted": True, "ended": True, "action_steps": 2},
        {"kind": "ending", "ending": "nominal", "end_reason": "join", "ended_at": 2, "action_steps": 2},
    ]
    for row in second:
        writer.append({"episode": "inv", "t": 1, "action_step": 1, **row})
    result = A.derive(log, heads, "inv", pack=task())
    assert [r["category"] for r in result["steps"]] == ["small_model", "holding_pending"]
    held = result["steps"][1]["held"]
    assert held == result["steps"][0]["set"] and held["goal_version"] == 1
    prefix = A.derive(log, heads, "inv", prefix=1, pack=task())
    assert prefix["K"] == 1 and not prefix["complete"] and prefix["steps_after_cut"] == 1


def test_control_fields_cannot_be_credited_to_another_decision_even_if_the_tables_overlap(tmp_path):
    def mutate(rows):
        renamed(rows)
        rows["report"]["part"] = {"throttle": 0.5}
        rows["action"]["field_writers"]["throttle"] = "signal"
    pack = task()
    pack = replace(pack, public=replace(pack.public,
                   decision_fields={"drive": {"throttle", "help"}, "signal": {"lamp", "throttle"}}))
    with pytest.raises(A.AttributionError, match="control fields were written by another"):
        A.derive(*signed(tmp_path, mutate), "inv", pack=pack)


@pytest.mark.parametrize("reduced", (False, True))
def test_the_host_uses_the_exact_episode_task_for_its_attribution(tmp_path, monkeypatch, reduced):
    from types import SimpleNamespace
    import paper2.experiment2.run_stream as host
    from telohelm.taskpacks.tanker.pack import TANKER_REDUCED_PACK
    from telohelm.runtime.library import Library
    seen = {}
    def episode(*args, **kwargs):
        seen["episode"] = kwargs["pack"]
        return {"outcome": {"goal_met": False, "joined": False, "report_correct": False,
                            "end_reason": "horizon", "ended_at": 1, "action_steps": 1}, "climbs": 0}
    def attribution(*args, pack):
        seen["attribution"] = pack
        return {"shares": {}, "K": 1, "complete": True}
    monkeypatch.setattr(host.L, "run_episode", episode)
    monkeypatch.setattr(host.A, "derive", attribution)
    host.run_episode(SimpleNamespace(episode_id="invented"), Library(tmp_path / "library"),
                     tmp_path / "run", "invented", 1, {}, flight={"reduced": reduced, "no_ladder": True})
    expected = TANKER_REDUCED_PACK if reduced else TANKER_PACK
    assert seen["episode"] is seen["attribution"] is expected
