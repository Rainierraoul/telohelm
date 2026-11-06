"""Slice 2(a) of the task pack: the goal's parts (study2/packs/goal_parts.py, tanker_execution.py).

Wired in on 23 October 2026. The pack's rules must give what the inline rules they replaced in learning.py,
runtime/episode_loop.py, runtime/broker.py and examples.py gave, on every input they can meet; those rules are copied
here as they stood at d8d164b.
"""
from itertools import product
from dataclasses import replace

import pytest

from telohelm.contracts.goal_parts import GoalPart; from telohelm.contracts.goal_parts import GoalParts
from telohelm.taskpacks.tanker.public import TANKER
from telohelm.taskpacks.tanker.execution import TANKER_GOAL

LEARNED = (("heading",), ("heading", "report"))     # the only sets run_stream.py builds a coordinator with


def inline_failed(outcome):                          # learning.Coordinator.after_episode
    return {(False, True): "join", (True, False): "report", (False, False): "both"}.get(
        (bool(outcome["joined"]), bool(outcome["report_correct"])))


def inline_failed_part(outcome):                     # runtime/episode_loop.py, feedback at episode end
    failed = [part for part, ok in (("join", outcome["joined"]), ("report", outcome["report_correct"])) if not ok]
    return {0: None, 1: failed[0] if failed else None, 2: "both"}[len(failed)]


def inline_concerns(failed):                         # learning.Coordinator.after_episode
    return {"join": {"heading"}, "report": {"report"}, "both": {"heading", "report"}}.get(failed, set())


def inline_floor_met(goal_met, failed, helped, learned):   # learning.Coordinator._floor
    if "report" in learned:
        return goal_met and not helped
    return (goal_met or failed == "report") and not helped


def test_the_pack_gives_the_inline_rules_on_every_outcome_they_can_meet():
    for joined, report_correct, goal_met, helped in product((False, True), repeat=4):
        outcome = {"joined": joined, "report_correct": report_correct}
        failed = inline_failed(outcome)
        assert TANKER_GOAL.failed(outcome) == failed == inline_failed_part(outcome)
        assert TANKER_GOAL.concerns(failed) == inline_concerns(failed)
        for learned in LEARNED:
            assert (TANKER_GOAL.met_for(goal_met, failed, learned) and not helped) \
                == inline_floor_met(goal_met, failed, helped, learned)


def test_truthy_fields_are_read_as_the_inline_rules_read_them():
    for joined, report_correct in product((0, 1, None, "", "yes"), repeat=2):
        outcome = {"joined": joined, "report_correct": report_correct}
        assert TANKER_GOAL.failed(outcome) == inline_failed(outcome) == inline_failed_part(outcome)


def test_the_floor_label_and_none_point_to_no_decision():
    assert TANKER_GOAL.concerns("floor") == frozenset() == TANKER_GOAL.concerns(None)


def test_the_tankers_parts_name_only_its_own_decisions():
    assert set().union(*(p.decisions for p in TANKER_GOAL.parts)) == set(TANKER.decisions)


def test_a_one_part_goal_from_another_task_needs_no_tanker_words():
    toy = GoalParts(parts=(GoalPart("kept_close", "close", {"move"}),), all_failed="all")
    assert toy.failed({"kept_close": False}) == "close" and toy.failed({"kept_close": True}) is None
    assert toy.concerns("close") == {"move"} == toy.concerns("all")
    assert toy.met_for(False, "close", ("signal",)) and not toy.met_for(False, "close", ("move",))


def test_a_three_part_goal_refuses_a_combination_it_has_no_label_for():
    three = GoalParts(parts=(GoalPart("a", "a", {"x"}), GoalPart("b", "b", {"y"}), GoalPart("c", "c", {"z"})),
                      all_failed="all")
    assert three.failed({"a": False, "b": True, "c": True}) == "a"
    assert three.failed({"a": False, "b": False, "c": False}) == "all"
    with pytest.raises(ValueError):
        three.failed({"a": False, "b": False, "c": True})


def test_the_parts_cannot_be_changed_or_made_ambiguous():
    decisions = {"heading"}
    part = GoalPart("joined", "join", decisions)
    decisions.add("leak")
    assert part.decisions == {"heading"}
    with pytest.raises(AttributeError):
        part.decisions.add("anything")
    with pytest.raises(Exception):
        TANKER_GOAL.all_failed = "changed"
    with pytest.raises(ValueError):
        GoalParts(parts=(GoalPart("joined", "join", {"heading"}), GoalPart("other", "join", {"report"})),
                  all_failed="both")
    with pytest.raises(ValueError):
        GoalParts(parts=(GoalPart("joined", "join", {"heading"}),), all_failed="join")
    with pytest.raises(TypeError):
        GoalParts(parts=(), all_failed="both")
    with pytest.raises(TypeError):
        GoalPart("joined", None, {"heading"})


# -- wired in (slice 2(a), 23 October 2026) ------------------------------------------------------------------------

def test_the_learning_core_takes_its_labels_and_their_decisions_from_the_goal_it_is_given(tmp_path):
    import telohelm.learning as G
    import paper2.experiment2.mvp_seed as seed
    from telohelm.runtime.library import Library
    import telohelm.taskpacks.tanker.world.phase5 as phase5
    from telohelm.taskpacks.tanker.pack import TANKER_PACK
    toy = GoalParts(parts=(GoalPart("joined", "missed the join", {"heading"}),
                           GoalPart("report_correct", "bad call", {"report"})), all_failed="everything")
    for learned, queued in ((("heading", "report"), ["missed the join", "bad call", "everything"]),
                            (("heading",), ["missed the join", "everything"])):
        lib = Library(tmp_path / f"library-{len(learned)}")
        seed.stock(lib)
        co = G.Coordinator(lib, tmp_path / f"w{len(learned)}", lambda m, meta: "no json", lambda s: True,
                           seed.stand_in_tool, attempt_every=1000, learn_decisions=learned,
                           flight_pack=replace(TANKER_PACK, goal=toy))
        for i, (joined, report) in enumerate(((False, True), (True, False), (False, False)), 1):
            co.after_episode(phase5.draw(i).world_a, {"goal_met": False, "joined": joined, "report_correct": report},
                             complete=True)
        assert [c.failed for c in co.completed] == ["missed the join", "bad call", "everything"]
        assert [q["failed"] for q in co.queue] == queued


def test_the_broker_accepts_exactly_the_labels_the_tanker_goal_gives():
    import telohelm.runtime.broker as B
    assert set(B.FEEDBACK["episode_end"]["failed_part"]) == {"join", "report", "both", None}


def test_a_broker_given_a_goal_accepts_that_goals_labels_and_no_others(tmp_path):
    import telohelm.runtime.broker as B
    from telohelm.runtime.ledger import Ledger
    from telohelm.runtime.trusted_log import TrustedLog
    toy = GoalParts(parts=(GoalPart("kept_close", "drifted", {"move"}),), all_failed="all")
    (tmp_path / "h").mkdir()
    br = B.Broker("e", TrustedLog(tmp_path / "log", tmp_path / "h" / "heads", "s"), Ledger(), lambda a: {}, goal=toy)
    br.end_episode()
    br.feedback("episode_end", goal_met=False, failed_part="drifted")
    import pytest
    with pytest.raises(B.BrokerError):
        br.feedback("episode_end", goal_met=False, failed_part="join")
    assert TANKER_GOAL.labels() == ("join", "report", "both", None)
