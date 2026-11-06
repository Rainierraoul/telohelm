"""The adviser comparison (analysis_adviser.py) on invented records: no world, model or flight."""
import json

import pytest

import paper2.experiment2.analysis_adviser as B
from paper2.experiment2.tests import test_analysis as T


def _with_adviser(tmp_path, summaries=True):
    study = T._study(tmp_path)
    root = study / str(T.STREAM)
    path = root / "schedule.json"
    envelope = json.loads(path.read_text())
    picked = {T._ep(p, i) for p in (1, 2, 3, 4, 5) for i in (0, 6, 11, 16, 19)}
    for row in envelope["schedule"]["rows"]:
        row["adviser"] = row["episode"] in picked
    path.write_text(json.dumps(envelope))
    if summaries:
        records = {}
        for e in sorted(picked):
            summary = T._summary(goal_met=e.endswith("00"), calls=4)
            summary["method_seconds"] = {"procedure": "not applicable", "small_model": "not applicable",
                                         "reasoning_model": 4, "holding_pending": 240, "holding_none": 156}
            records[e] = summary
        T._write_arm(root, "adviser", records)
        for e in picked:
            (root / "arms/adviser" / e / "ledger.jsonl").write_text(
                ''.join(json.dumps({"tokens_in": 10, "tokens_out": 1}) + '\n' for _ in range(4)))
    return study, root, picked


def test_adviser_and_learning_are_counted_on_the_same_episodes(tmp_path):
    _, root, picked = _with_adviser(tmp_path)
    out = B.stream(root)
    assert out["matched_episodes"] == 25
    adviser, learning = out["table"]["adviser"], out["table"]["L"]
    assert adviser["pooled"]["positions"] == 25 and adviser["pooled"]["cost"]["model_calls"] == 100
    assert adviser["pooled"]["outcomes"] == {"met_alone": 5, "missed": 20}
    assert learning["pooled"]["positions"] == 25 and learning["pooled"]["outcomes"]["not_attempted"] == 1
    assert adviser["by_phase"][1]["positions"] == 5
    assert set(learning["d1_learning_share_on_matched"]) == {"model_calls", "tokens_in", "tokens_out"}
    assert "total" in learning["full_learning_cost"]


def test_a_stream_the_adviser_has_not_flown_is_reported_as_such(tmp_path):
    _, root, _ = _with_adviser(tmp_path, summaries=False)
    assert B.stream(root)["adviser"] == "not flown"


def test_flown_inputs_that_differ_are_refused(tmp_path):
    _, root, _ = _with_adviser(tmp_path)
    state = json.loads((root / "adviser-state.json").read_text())
    state["positions"][0]["input_matches_schedule"] = False
    (root / "adviser-state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="differ from the schedule"):
        B.stream(root)


def test_the_frozen_analysis_is_only_read(tmp_path):
    study, _, _ = _with_adviser(tmp_path)
    result = B.analyse(study)
    assert result["analysis_sha256"] and result["this_module_sha256"]


def test_timeout_keeps_answered_call_usage_and_actual_unknown_counts(tmp_path):
    import hashlib
    _, root, picked = _with_adviser(tmp_path)
    episode = sorted(picked)[0]
    folder = root / "arms/adviser" / episode
    calls = [{"status": "answered", "tokens_in": 10, "tokens_out": 1},
             {"status": "timed_out", "request_id": "request-2", "tokens_in": None, "tokens_out": None},
             {"status": "answered", "tokens_in": 30, "tokens_out": 3},
             {"status": "cut_off", "request_id": "request-4", "tokens_in": None, "tokens_out": None}]
    (folder / "ledger.jsonl").write_text(''.join(json.dumps(c) + '\n' for c in calls))
    path = folder / "summary.json"
    summary = json.loads(path.read_text())
    summary.update(tokens_in=None, tokens_out=None)
    path.write_text(json.dumps(summary))
    state_path = root / "adviser-state.json"
    state = json.loads(state_path.read_text())
    next(p for p in state["positions"] if p["episode"] == episode)["summary_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    state_path.write_text(json.dumps(state))
    result = B.stream(root)
    table = result["table"]["adviser"]["pooled"]
    assert table["cost"] == {"model_calls": 100, "tokens_in": None, "tokens_out": None}
    assert table["known_subtotal"]["tokens_in"] == 24 * 40 + 40
    assert table["known_subtotal"]["tokens_out"] == 24 * 4 + 4
    assert table["unknown_calls"] == {"tokens_in": 2, "tokens_out": 2}
    assert table["calls_with_unknown_usage"] == 2
    assert json.loads(path.read_text()) == summary
    assert [(c["episode"], c["phase"], c["call_position"], c["status"])
            for c in result["recorded_timeout_or_cutoff_calls"]] == [
                (episode, 1, 2, "timed_out"), (episode, 1, 4, "cut_off")]
    assert result["call_records_complete"] is True


def test_adviser_seconds_use_recorded_counts_with_inapplicable_methods(tmp_path):
    _, root, _ = _with_adviser(tmp_path)
    seconds = B.stream(root)["table"]["adviser"]["pooled"]["seconds"]
    assert seconds == {"procedure": "not applicable", "small_model": "not applicable",
                       "reasoning_model": 100, "holding_pending": 6000, "holding_none": 3900}


def test_secondary_comparison_uses_only_the_original_draw_intersection(tmp_path):
    _, root, _ = _with_adviser(tmp_path)
    common = B.stream(root)["common_with_C"]
    assert common["episode_ids"] == [T._ep(p, 0) for p in (1, 2, 3, 4)]
    assert common["matched_episodes"] == 4
    assert {arm: t["positions"] for arm, t in common["table"].items()} == {
        "adviser": 4, "L": 4, "C": 4, "C-full": 4}
    assert common["table"]["adviser"]["cost"]["model_calls"] == 16
    assert common["table"]["C"]["cost"]["model_calls"] == 12


def test_the_recorded_counterfactual_stop_is_read_without_changing_its_effect(tmp_path):
    study, _, _ = _with_adviser(tmp_path)
    path = study / "counterfactual-stops/adviser.json"
    path.parent.mkdir()
    observed = {"due": True, "enforced": False, "bad": 1, "started": 4, "detected_in_stream": T.STREAM}
    path.write_text(json.dumps(observed))
    result = B.analyse(study)
    assert result["adviser_counterfactual_stop"] == observed
    assert not (study / "stops/adviser.json").exists()
