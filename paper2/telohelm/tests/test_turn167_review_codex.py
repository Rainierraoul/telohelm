"""Exact help repair controls on a fixed invented world, never a stream draw."""
import pytest
import paper2.experiment2.tests.test_flight_ladder as F
from .test_episode_loop import scenario
from telohelm.runtime.library import Library
from telohelm.runtime.escalation import part_error


def test_sender_help_once_and_malformed_control_on_fixed_episode(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "scenario", scenario)
    F.test_a_model_help_request_is_the_episodes_only_one_and_malformed_help_escalates(tmp_path, monkeypatch)


@pytest.mark.parametrize("reason", [None, True, [], {}, 1, "because"])
def test_bad_help_reason_is_a_public_refusal(reason):
    assert part_error("heading", {"help": {"reason": reason, "text": "fixed"}}) is not None


def test_world_refusal_does_not_consume_a_models_unaccepted_help(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "scenario", scenario)
    lib = Library(tmp_path / "library-atomic")
    proc = lib.add({"kind": "procedure", "decision": "report", "conditions": [],
                    "source": 'def decide(ctx):\n    return {"report_trigger_nm":1001} if ctx.t == 2 else {}\n',
                    "origin": {"invented": True}})
    check = lib.add({"kind": "check", "decision": "report", "conditions": [],
                     "source": "def check(ctx,out):\n    return 'accept'\n", "origin": {"invented":True}})
    for sha in (proc,check):lib.set_status(sha,"adopted","fixed unit")
    help_reply='{"help":{"reason":"missing_tool","text":"fixed"}}'
    rec,rows,_=F.fly(tmp_path,monkeypatch,small=help_reply,reasoning=help_reply,library=lib,steps=66,name="atomic")
    assert any(e["kind"] == "escalated" and e["why"] == "world_refused" for e in rec["ladder"])
    # The help in second2 never acted. A later current reasoning reply must
    # still be able to submit the episode's one real request.
    assert rec["help_requests"] == 1
