"""The pre-registered analysis (analysis.py) on invented records: no world, model or flight."""
import hashlib
import json

import pytest

import paper2.experiment2.analysis as A

STREAM = 4999


def _ep(phase, index):
    return f"s{STREAM}-p{phase}-e{index:02d}"


def _summary(goal_met=False, complete=True, settled=True, help_reasons=(), calls=0):
    return {"goal_met": goal_met, "complete": complete, "settled": settled, "help_requests": len(help_reasons),
            "outcome": {"help_requests": [{"t": 1, "reason": r} for r in help_reasons]},
            "model_calls": calls, "tokens_in": 10 * calls, "tokens_out": calls, "calls_with_unknown_usage": 0,
            "action_steps": 100, "share_of_seconds": {"procedure": 0.5, "small_model": 0.5}}


def _write_arm(root, arm, rows):
    """rows: episode -> summary or None (not attempted)."""
    positions = []
    for episode, summary in rows.items():
        if summary is None:
            positions.append({"episode": episode, "status": "not_attempted"})
            continue
        rel = f"arms/{arm}/{episode}/summary.json"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(json.dumps(summary))
        positions.append({"episode": episode, "status": "complete" if summary["complete"] else "censored",
                          "summary": rel, "summary_sha256": hashlib.sha256((root / rel).read_bytes()).hexdigest()})
    (root / f"{arm}-state.json").write_text(json.dumps({"status": "returned", "positions": positions}))


def _study(tmp_path, attempt_at_second_boundary=True):
    study = tmp_path / "study"
    root = study / str(STREAM)
    root.mkdir(parents=True)
    (study / "study.json").write_text(json.dumps({"streams": [STREAM]}))
    schedule = [_ep(p, i) for p in (1, 2, 3, 4, 5) for i in range(20)]
    c = {_ep(p, i) for p in (1, 2, 3, 4) for i in (0, 1, 2, 3, 4)}
    f = {_ep(3, i) for i in (*range(5, 10), *range(15, 20))}
    (root / "schedule.json").write_text(json.dumps({"schedule": {"stream": STREAM, "mode": "development", "rows": [
        {"episode": e, "c": e in c, "f": e in f} for e in schedule]}}))
    _write_arm(root, "L", {e: _summary(goal_met=True, calls=1) for e in schedule[:-1]} | {schedule[-1]: None})
    _write_arm(root, "C", {e: _summary(goal_met=False, calls=3) for e in sorted(c)})
    _write_arm(root, "C-full", {e: _summary(goal_met=True, help_reasons=("missing_tool",), calls=30)
                                for e in sorted(c)})
    _write_arm(root, "F", {e: _summary(goal_met=e.endswith(("05", "06")), complete=not e.endswith("07"))
                           for e in sorted(f)})
    events = [{"kind": "completed", "episode": e, "ordinal": i + 1} for i, e in enumerate(schedule[:-1])]
    events += [{"kind": "attempt_start", "attempt": 1, "ordinal": 45},
               {"kind": "attempt_result", "attempt": 1, "decision": "heading", "entry_kind": "procedure",
                "passed": True}]
    if attempt_at_second_boundary:
        events += [{"kind": "attempt_start", "attempt": 2, "ordinal": 55},
                   {"kind": "attempt_result", "attempt": 2, "decision": "heading", "passed": False}]
    (root / "arms/L/learning-events.json").write_text(json.dumps(events))
    ledger = [{"scope": "build", "episode": _ep(3, 4), "tokens_in": 900, "tokens_out": 90},
              {"scope": "running", "episode": _ep(3, 4), "tokens_in": 5, "tokens_out": 5}]
    (root / "arms/L/learning-ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ledger))
    return study


def test_outcomes_are_met_alone_with_help_missed_or_kept_apart():
    assert A.outcome(_summary(goal_met=True), "complete") == "met_alone"
    assert A.outcome(_summary(goal_met=True, help_reasons=("missing_tool",)), "complete") == "met_with_help"
    assert A.outcome(_summary(goal_met=False), "complete") == "missed"
    assert A.outcome(_summary(goal_met=True, complete=False), "censored") == "censored"
    assert A.outcome(_summary(goal_met=True, settled=False), "complete") == "unsettled"
    assert A.outcome(None, "not_attempted") == "not_attempted"


def test_rq1_compares_l_c_and_c_full_on_cs_matched_positions_with_d1_learning_share(tmp_path):
    result = A.analyse(_study(tmp_path))["per_stream"][0]
    rq1 = result["rq1"]
    assert rq1["L"]["pooled"]["positions"] == rq1["C"]["pooled"]["positions"] == rq1["C-full"]["pooled"]["positions"] == 20
    assert rq1["L"]["pooled"]["outcomes"] == {"met_alone": 20}
    assert rq1["C"]["pooled"]["outcomes"] == {"missed": 20} and rq1["C"]["pooled"]["model_calls_per_goal_met"] is None
    assert rq1["C-full"]["pooled"]["outcomes"] == {"met_with_help": 20}
    assert rq1["C-full"]["pooled"]["cost"]["model_calls"] == 600
    assert rq1["L"]["pooled"]["seconds"] == {"procedure": 1000, "small_model": 1000, "reasoning_model": 0,
                                             "holding_pending": 0, "holding_none": 0}
    # One build at position 44 (p3-e04): 900 tokens in over the 56 positions from 44 to 99, forward only. Of C's
    # matched positions, p3-e00 to e04 are before or at it (only e04 shares) and all five of phase 4 are after it.
    share = rq1["L"]["d1_learning_share_on_matched"]
    assert share["tokens_in"] == pytest.approx(900 / 56 * 6) and share["model_calls"] == pytest.approx(6 / 56)
    assert result["learning"]["build_cost_total"] == {"model_calls": 1, "tokens_in": 900, "tokens_out": 90}
    assert result["statuses"]["L"] == {"complete": 99, "not_attempted": 1}


def test_rq2_reports_each_block_beside_its_attempts_and_marks_a_block_without_one(tmp_path):
    rq2 = A.analyse(_study(tmp_path, attempt_at_second_boundary=False))["per_stream"][0]["rq2"]
    first, second = rq2
    assert first["boundary"] == _ep(3, 4) and first["attempts"] == 1 and first["adoptions"] == 1
    assert first["informative"] is True
    assert first["F_after"]["outcomes"] == {"met_alone": 2, "censored": 1, "missed": 2}
    assert first["L_after"]["outcomes"] == first["L_before"]["outcomes"] == {"met_alone": 5}
    assert second["boundary"] == _ep(3, 14) and second["attempts"] == 0 and second["informative"] is False
    both = A.analyse(_study(tmp_path / "b"))["across"]
    assert both["RQ2 block 2 informative streams"] == {"per_stream": [1], "range": [1, 1]}


def test_help_is_counted_by_condition_phase_and_reason(tmp_path):
    help_table = A.analyse(_study(tmp_path))["per_stream"][0]["help"]
    assert help_table == {f"C-full phase {p} missing_tool": 5 for p in (1, 2, 3, 4)}


def test_a_summary_that_differs_from_its_state_record_is_refused(tmp_path):
    study = _study(tmp_path)
    path = study / str(STREAM) / "arms/C" / _ep(1, 0) / "summary.json"
    path.write_text(json.dumps(_summary(goal_met=True)))
    with pytest.raises(ValueError, match="differs from its state record"):
        A.analyse(study)


def test_a_declared_stream_that_never_ran_is_not_analysed_and_keeps_its_disposition(tmp_path):
    """Codex's TURN295 witness: a prepared stream the budget rule never started is not zero observations."""
    study = _study(tmp_path)
    declared = json.loads((study / "study.json").read_text())
    declared["streams"] += [4998, 4997]
    (study / "study.json").write_text(json.dumps(declared))
    for s in (4998, 4997):
        (study / str(s)).mkdir()
        (study / str(s) / "schedule.json").write_text((study / f"{STREAM}/schedule.json").read_text())
    (study / "4998" / "launch-disposition.json").write_text(json.dumps({"status": "not run for budget"}))
    out = A.analyse(study)
    assert out["declared_streams"] == [STREAM, 4998, 4997] and out["analysed_streams"] == [STREAM]
    assert out["dispositions"] == {str(STREAM): {"status": "ran"}, "4998": {"status": "not run for budget"},
                                   "4997": {"status": "not started"}}
    assert out["across"]["RQ1 L goals met alone on matched positions"] == {"per_stream": [20], "range": [20, 20]}
    # Inside a stream that ran, an unattempted position stays in the denominator.
    assert out["per_stream"][0]["statuses"]["L"]["not_attempted"] == 1


def test_unknown_token_usage_makes_totals_and_d1_shares_unavailable(tmp_path):
    """Codex's TURN295 witness: unknown usage is never zero; known subtotals and unknown counts are kept."""
    study = _study(tmp_path)
    root = study / str(STREAM)
    ledger = root / "arms/L/learning-ledger.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    rows[0].update(tokens_in=None)
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
    episode = _ep(1, 0)
    unknown = {**_summary(goal_met=False, calls=3), "tokens_in": None, "tokens_in_known_subtotal": 20,
               "tokens_in_unknown_calls": 1, "calls_with_unknown_usage": 1}
    _write_arm(root, "C", {e: (unknown if e == episode else _summary(goal_met=False, calls=3))
                           for e in sorted(json.loads((root / "C-state.json").read_text())["positions"][i]["episode"]
                                           for i in range(20))})
    result = A.analyse(study)["per_stream"][0]
    learning = result["learning"]
    assert learning["build_cost_total"]["tokens_in"] is None and learning["build_cost_total"]["tokens_out"] == 90
    assert learning["build_known_subtotal"]["tokens_in"] == 0 and learning["build_unknown_calls"]["tokens_in"] == 1
    share = result["rq1"]["L"]["d1_learning_share_on_matched"]
    assert share["tokens_in"] is None and share["tokens_out"] == pytest.approx(90 / 56 * 6)
    c = result["rq1"]["C"]["pooled"]
    assert c["cost"]["tokens_in"] is None and c["cost"]["tokens_out"] == 60 and c["cost"]["model_calls"] == 60
    assert c["known_subtotal"]["tokens_in"] == 19 * 30 + 20 and c["unknown_calls"]["tokens_in"] == 1
    assert c["calls_with_unknown_usage"] == 1


# Cut-off records (Codex's TURN303 review): a dispatcher stopped at the host's time limit can leave its first, running
# state; its positions are then read from the host's receipt rebuilt from the records, and learning from the journal.

def _cut_off_l(study, flown, open_at=None):
    """Leave L's first, running state and write the host's rebuilt receipt for it: the first `flown` positions
    complete with their summaries bound, the position `open_at` with its learning boundary open, the rest not
    started. Returns the receipt's path."""
    root = study / str(STREAM)
    schedule = [r["episode"] for r in json.loads((root / "schedule.json").read_text())["schedule"]["rows"]]
    (root / "L-state.json").write_text(json.dumps({"status": "running", "positions": [
        {"episode": e, "status": "scheduled"} for e in schedule]}))
    positions = []
    for i, e in enumerate(schedule):
        if i < flown:
            rel = f"arms/L/{e}/summary.json"
            positions.append({"episode": e, "status": "boundary_open" if e == open_at else "complete",
                              "boundary": "open" if e == open_at else "closed", "summary": rel,
                              "summary_sha256": hashlib.sha256((root / rel).read_bytes()).hexdigest()})
        else:
            positions.append({"episode": e, "status": "not_started", "boundary": "none"})
    path = study.parent / "host-receipts.json"
    path.write_text(json.dumps({"reason": "deadline", "dispatch": [
        {"name": f"L:{STREAM}", "arm": "L", "receipts_from": "records", "dispatch_status": "cut_off",
         "positions": positions}]}))
    return path


def _journal(folder, events, session="s1"):
    (folder / "learning-events.jsonl").write_text("".join(
        json.dumps({"session": session, "event_number": i + 1, "event": e}, sort_keys=True) + "\n"
        for i, e in enumerate(events)))


def test_a_cut_off_condition_is_read_from_the_host_receipt_and_refused_without_it(tmp_path):
    study = _study(tmp_path)
    receipt = _cut_off_l(study, flown=99)
    with pytest.raises(ValueError, match="not final"):
        A.analyse(study)
    result = A.analyse(study, [receipt])
    l = result["per_stream"][0]["rq1"]["L"]["pooled"]
    assert l["outcomes"] == {"met_alone": 20} and l["cost"]["model_calls"] == 20
    assert result["per_stream"][0]["statuses"]["L"] == {"complete": 99, "not_started": 1}
    assert list(result["host_receipts"]) == [str(receipt)]
    body = json.loads(receipt.read_text())
    body["dispatch"][0]["receipts_from"] = "dispatcher"
    receipt.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="read from a final one"):
        A.analyse(study, [receipt])


def test_a_receipt_summary_that_differs_from_its_digest_is_refused(tmp_path):
    study = _study(tmp_path)
    receipt = _cut_off_l(study, flown=99)
    (study / str(STREAM) / "arms/L" / _ep(1, 0) / "summary.json").write_text(json.dumps(_summary(goal_met=False)))
    with pytest.raises(ValueError, match="differs from its state record"):
        A.analyse(study, [receipt])


def test_an_interrupted_position_counts_apart_and_makes_its_costs_unavailable(tmp_path):
    study = _study(tmp_path)
    path = study / str(STREAM) / "C-state.json"
    state = json.loads(path.read_text())
    state["positions"][0].update(status="interrupted", interrupted=True)   # its saved summary is kept as evidence
    path.write_text(json.dumps(state))
    c = A.analyse(study)["per_stream"][0]["rq1"]["C"]["pooled"]
    assert c["outcomes"] == {"interrupted": 1, "missed": 19}
    assert c["cost"] == {"model_calls": None, "tokens_in": None, "tokens_out": None} and c["seconds"] is None
    assert c["known_subtotal"]["model_calls"] == 19 * 3 and c["positions_with_unknown_cost"] == 1
    assert c["model_calls_per_goal_met"] is None


def test_learning_comes_from_the_journal_when_the_final_record_is_missing_or_stale(tmp_path):
    study = _study(tmp_path)
    folder = study / str(STREAM) / "arms/L"
    events = json.loads((folder / "learning-events.json").read_text())
    _journal(folder, events)
    assert len(A.analyse(study)["per_stream"][0]["learning"]["attempts"]) == 2          # final record agrees
    (folder / "learning-events.json").write_text("[]")
    with pytest.raises(ValueError, match="differs from every session"):
        A.analyse(study)
    (folder / "learning-events.json").unlink()
    with (folder / "learning-events.jsonl").open("a") as f:
        f.write('{"session": "s1", "event_number": 1')                     # a write the stop cut short
    learn = A.analyse(study)["per_stream"][0]["learning"]
    assert [a["attempt"] for a in learn["attempts"]] == [1, 2] and learn["open_attempts"] == 0
    assert learn["journal"] == {"source": "learning-events.jsonl", "sessions": 1, "cut_last_line": True}


def test_an_attempt_cut_off_at_an_open_boundary_has_unavailable_build_cost(tmp_path):
    study = _study(tmp_path)
    folder = study / str(STREAM) / "arms/L"
    boundary = _ep(3, 4)                                                    # schedule position 44, ordinal 45
    events = [{"kind": "completed", "episode": _ep(p, i), "ordinal": 20 * (p - 1) + i + 1}
              for p in (1, 2, 3) for i in range(20) if 20 * (p - 1) + i < 45]
    _journal(folder, events + [{"kind": "attempt_start", "attempt": 1, "ordinal": 45}])
    (folder / "learning-events.json").unlink()
    receipt = _cut_off_l(study, flown=45, open_at=boundary)
    result = A.analyse(study, [receipt])["per_stream"][0]
    learn = result["learning"]
    assert learn["attempts"] == [{"attempt": 1, "boundary": boundary, "results": [], "open": True}]
    assert learn["open_attempts"] == 1
    assert learn["build_cost_total"] == {"model_calls": None, "tokens_in": None, "tokens_out": None}
    assert learn["build_known_subtotal"] == {"model_calls": 1, "tokens_in": 900, "tokens_out": 90}
    assert all(v is None for v in result["rq1"]["L"]["d1_learning_share_on_matched"].values())
    first = result["rq2"][0]
    assert first["boundary"] == boundary and first["attempts"] == 1 and first["adoptions"] == 0
    assert result["statuses"]["L"] == {"complete": 44, "boundary_open": 1, "not_started": 55}
    # The flight before the open boundary finished, so its goal counts.
    assert result["rq2"][0]["L_before"]["outcomes"] == {"met_alone": 5}


def test_flown_inputs_that_differ_from_the_schedule_are_refused(tmp_path):
    study = _study(tmp_path)
    path = study / str(STREAM) / "C-state.json"
    state = json.loads(path.read_text())
    state["positions"][0]["input_matches_schedule"] = False
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="differ from the schedule"):
        A.analyse(study)


def test_pending_f_positions_are_kept_from_the_host_receipt_before_its_dispatcher_starts(tmp_path):
    """F may wait for L's snapshot until the deadline, leaving neither an F state nor an arm folder."""
    import shutil
    study = _study(tmp_path)
    root = study / str(STREAM)
    (root / "F-state.json").unlink()
    shutil.rmtree(root / "arms/F")
    eps = [_ep(3, i) for i in (*range(5, 10), *range(15, 20))]
    receipt = study.parent / "pending-F-receipts.json"
    receipt.write_text(json.dumps({"reason": "deadline", "dispatch": [
        {"name": f"F:{STREAM}", "arm": "F", "stream": STREAM, "receipts_from": "records",
         "dispatch_status": "not_started", "positions": [{"episode": ep, "status": "not_started"} for ep in eps]}]}))
    result = A.analyse(study, [receipt])["per_stream"][0]
    assert result["statuses"]["F"] == {"not_started": 10}
    for block in result["rq2"]:
        assert block["F_after"]["positions"] == 5
        assert block["F_after"]["outcomes"] == {"not_started": 5}
        assert block["F_after"]["cost"] == {"model_calls": 0, "tokens_in": 0, "tokens_out": 0}
