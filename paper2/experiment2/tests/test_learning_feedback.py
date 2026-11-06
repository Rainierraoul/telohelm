"""Learning feedback from job 4028249 (paper2/notes/LEARNING-FEEDBACK-PROPOSAL.md): the proposer is told the stage that
refused each of its last attempts, with fixed phrases and no case, count or context from the monitor's store; the pair
validator names the top-level reason; the prompt says what stored wrong answers are. Invented proposers, no model."""
from __future__ import annotations

import json

import telohelm.learning as G
from telohelm.runtime.library import Library
from telohelm.runtime.procedures import pair_errors

PROC = 'def decide(ctx):\n    return {"heading_deg": 90.0}\n'
CHECK = "def check(ctx, out):\n    return 'accept'\n"


def pair(text="invented", inner_reason=False):
    half = {"conditions": [], "source": PROC}
    chk = {"conditions": [], "source": CHECK}
    if inner_reason:
        half = {**half, "reason": {"code": "missing_tool", "text": "x"}}
        chk = {**chk, "reason": {"code": "missing_tool", "text": "x"}}
    return {"kind": "pair", "decision": "heading", "procedure": half, "check": chk,
            "reason": {"code": "missing_tool", "text": text}}


def test_the_pair_validator_names_the_top_level_reason_and_its_length():
    errs = pair_errors(pair("x" * 340))
    assert errs == ["the pair's top-level reason must give one of the 4 codes and text of at most 300 characters "
                    "(yours has 340)"]
    errs = pair_errors(pair(inner_reason=True))
    assert all("the reason belongs at the top level" in e for e in errs) and len(errs) == 2
    assert pair_errors(pair()) == []


def test_the_prompt_says_what_stored_wrong_answers_are():
    assert "well-formed actions that do the wrong thing" in G.PROPOSER_RULES
    assert "previous_attempts" in G.PROPOSER_RULES


def coordinator(tmp_path, answers, replay=None, trial=None):
    lib = Library(tmp_path / "library")
    it = iter(answers)
    co = G.Coordinator(lib, tmp_path, lambda messages, meta: next(it), lambda s: True, lambda a: {})
    co.completed = [G.Completed(5, "s4401-p1-e04", None, False, "both", True)]
    if replay is not None:
        co.check_replay = lambda *a, **k: replay
    if trial is not None:
        co.trial = lambda *a, **k: trial
    return co, lib


def prompt(co):
    return json.loads(co.messages([])[1][1])


def test_a_check_refused_at_the_replay_test_is_reported_by_stage_only(tmp_path):
    facts = {"replay": {"passed": False, "right_accepted": 100, "old_right_accepted": None, "wrong_cases": 140,
                        "right_cases": 100, "wrong_accepted_any": True, "right_short": False, "fewer_than_old": False}}
    co, lib = coordinator(tmp_path, [json.dumps(pair())], replay=(False, facts))
    co.attempt(co.completed[-1])
    previous = prompt(co)["previous_attempts"]
    assert previous == [{"attempt": 1, "kind": "pair", "decision": "heading", "refused_at": "replay_test",
                         "why": ["the check accepted at least one stored wrong answer"],
                         "source": {"procedure": PROC, "check": CHECK}}]
    text = json.dumps(previous)
    assert "140" not in text and "wrong_cases" not in text and "100" not in text   # no counts from the store


def test_a_procedure_refused_at_trial_names_its_own_episode(tmp_path):
    ok_check = (True, {"replay": {"passed": True, "right_accepted": 100, "old_right_accepted": None,
                                  "wrong_cases": 140, "right_cases": 100, "wrong_accepted_any": False,
                                  "right_short": False, "fewer_than_old": False}})
    failed = (False, {"trial": [{"episode": "s4401-p1-e04", "goal_met": False,
                                 "component": {"passed": False}}]})
    co, lib = coordinator(tmp_path, [json.dumps(pair())], replay=ok_check, trial=failed)
    co.attempt(co.completed[-1])
    last = prompt(co)["previous_attempts"][-1]
    assert last["refused_at"] == "trial"
    assert last["why"] == ["the procedure did not do its part of the goal in trial episode s4401-p1-e04"]


def test_format_and_no_answer_attempts_and_only_the_latest_source(tmp_path):
    long_reason = json.dumps(pair("x" * 340))
    co, lib = coordinator(tmp_path, [long_reason, long_reason, long_reason, "", "", ""])
    co.attempt(co.completed[-1])
    co.attempt(co.completed[-1])
    previous = prompt(co)["previous_attempts"]
    assert [p["refused_at"] for p in previous] == ["format", "no_answer"]
    assert "top-level reason" in previous[0]["why"][0]
    assert all("source" not in p for p in previous)
    facts = {"replay": {"passed": False, "right_accepted": 90, "old_right_accepted": None, "wrong_cases": 1,
                        "right_cases": 100, "wrong_accepted_any": False, "right_short": True,
                        "fewer_than_old": False}}
    co2, _ = coordinator(tmp_path / "b", [json.dumps(pair()), json.dumps(pair())], replay=(False, facts))
    co2.attempt(co2.completed[-1])
    co2.attempt(co2.completed[-1])
    previous = prompt(co2)["previous_attempts"]
    assert "source" not in previous[0] and "source" in previous[1]
    assert previous[1]["why"] == ["the check accepted fewer than 95 of the 100 right answers"]


def test_a_long_refused_source_is_dropped_first_and_its_omission_stated(tmp_path):
    big = pair()
    big["procedure"]["source"] = "def decide(ctx):\n" + "    x = 1\n" * 690 + "    return {}\n"   # under 8,000
    big["check"]["source"] = "def check(ctx, out):\n" + "    y = 2\n" * 690 + "    return 'accept'\n"
    facts = {"replay": {"passed": False, "wrong_accepted_any": True, "right_short": False, "fewer_than_old": False}}
    co, _ = coordinator(tmp_path, [json.dumps(big)], replay=(False, facts))
    co.attempt(co.completed[-1])
    last = prompt(co)["previous_attempts"][-1]
    assert last.get("source_omitted") is True and "source" not in last


def test_replay_details_carry_only_the_two_facts():
    r = {"passed": False, "wrong_accepted": [3, 7], "right_accepted": 96, "old_right_accepted": 98}
    assert G._two_facts(r) == {"wrong_accepted_any": True, "right_short": False, "fewer_than_old": True}


def test_development_budget_options_reach_the_proposer_and_the_template():
    import telohelm.local_model as LM
    p = G.GatewayProposer("http://127.0.0.1:1", "Qwen/Qwen3.8-27B", __import__("pathlib").Path("/tmp"), "dev",
                          max_output_tokens=24576, timeout_s=1800.0, effort="xhigh", count_tokens=lambda m: 1)
    assert p.max_output_tokens == 24576 and p.provider.timeout_s_by_role["reasoning_model"] == 1800.0
    assert p.provider._extra_body["chat_template_kwargs"]["reasoning_effort"] == "xhigh"
    assert p._config()["effort"] == "xhigh" and p._config()["timeout_s"] == 1800.0
    plain = LM.provider("http://127.0.0.1:1", "m", role="reasoning_model")
    assert plain._extra_body["chat_template_kwargs"]["reasoning_effort"] == "medium"     # the protocol's default


def test_small_model_rationing_spaces_calls_per_decision():
    from paper2.experiment2.run_stream import SmallRation
    r = SmallRation(10)
    assert r.may_issue("small_model", "heading", 0) and not r.may_issue("small_model", "heading", 5)
    assert r.may_issue("small_model", "report", 5)                                        # per decision
    assert r.may_issue("small_model", "heading", 10) and r.may_issue("reasoning_model", "heading", 11)
    assert SmallRation(0).may_issue("small_model", "heading", 1)                          # protocol rule kept


def test_rationing_without_the_ladder_builds_no_ladder(tmp_path, monkeypatch):
    from . import test_flight_ladder as F
    import paper2.experiment2.run_stream as RS
    monkeypatch.setattr(RS.seed, "stand_in_model", F.answers(None, None))
    out = tmp_path / "r"; out.mkdir()
    done = RS.run_episode(F.scenario(), Library(tmp_path / "lib"), out, "dev-test", 120,
                          {"small_model": 2.0, "reasoning_model": 60.0}, None,
                          {"allowance": None, "reasoning": None, "no_ladder": True, "small_interval_s": 20})
    assert done["record"]["ladder"] is None
    assert done["record"]["model_calls"] <= 12                     # 120 s, two decisions, one call per 20 s each
