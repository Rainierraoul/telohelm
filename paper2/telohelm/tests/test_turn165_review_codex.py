"""Exact TURN165 fixed invented controls; no generated episode or provider."""
import pytest
from paper2.experiment2.tests.test_turn163_review_codex import fly


def test_atomic_world_refusal_link_on_fixed_invented_episode(tmp_path, monkeypatch):
    import paper2.experiment2.tests.test_flight_ladder as F
    from .test_episode_loop import scenario
    monkeypatch.setattr(F, "scenario", scenario)
    F.test_a_world_refusal_escalates_the_small_reply_it_swallowed(tmp_path, monkeypatch)


def test_current_help_only_model_reply_is_delivered_without_heading_attribution(tmp_path, monkeypatch):
    result = fly(tmp_path, monkeypatch, {"help": {"reason": "missing_tool", "text": "Need a tool"}})
    assert result['record']['help_requests'] == 1
    assert result['record']['share_of_seconds']['small_model'] == 0
