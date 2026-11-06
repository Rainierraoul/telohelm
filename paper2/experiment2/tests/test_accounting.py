"""Combined method-seconds and costs (D24, Codex's C131 corrections). Invented records, plus readers on records
written in the test; nothing here is a result."""
from __future__ import annotations

import pytest

import paper2.experiment2.accounting as A


def row(id, step, process="policy", purpose="running", **kw):
    measured = kw.pop("measured", ("cpu_s", "wall_s", "tokens_in", "tokens_out"))
    return A.Row(id, "telohelm", "e", purpose, process, step, measured=measured, **kw)


def test_not_applicable_categories_are_null_and_counts_sum_to_k():
    m = A.method_seconds("calculator", {"calculator": 3, "holding_none": 7})
    assert m["K"] == 10 and m["counts"]["calculator"] == 3 and m["counts"]["procedure"] is None
    assert m["fractions"]["holding_none"] == 0.7 and m["fractions"]["small_model"] is None
    with pytest.raises(ValueError):
        A.method_seconds("calculator", {"procedure": 1})          # never guessed into another category


def test_zero_seconds_give_undefined_fractions_and_pooling_adds_counts():
    z = A.method_seconds("adaptive_controller", {})
    assert z["K"] == 0 and z["fractions"]["adaptive_controller"] is None
    a = A.method_seconds("adaptive_controller", {"adaptive_controller": 1, "holding_none": 1})
    p = A.pool_method_seconds([z, a])
    assert p["K"] == 2 and p["counts"]["adaptive_controller"] == 1 and p["fractions"]["holding_none"] == 0.5
    with pytest.raises(ValueError):
        A.pool_method_seconds([a, A.method_seconds("calculator", {})])


def test_d1_suffix_allocation_matches_codexs_control_and_conserves_the_total():
    """C131.D1: running [1, 2, 3, 4], builds 12 at position 1 and 8 at position 3 give [1, 6, 7, 16], total 30."""
    r = A.allocate_learning([1.0, 2.0, 3.0, 4.0], {1: 12.0, 3: 8.0})
    assert r["amortised"] == [1.0, 6.0, 7.0, 16.0] and sum(r["amortised"]) == r["raw_total"] == 30.0
    assert r["actual"] == [1.0, 14.0, 3.0, 12.0] and r["cumulative_build"] == [0.0, 12.0, 12.0, 20.0]
    stopped = A.allocate_learning([1.0, 2.0, 3.0, 4.0], {1: 12.0}, attempted=2)
    assert stopped["observed"] == [True, True, False, False]


def test_a_matched_subset_selects_existing_shares_and_redistributes_nothing():
    shares = A.allocate_learning([1.0, 2.0, 3.0, 4.0], {1: 12.0, 3: 8.0})["amortised"]
    m = A.matched_subset(shares, [0, 3])
    assert m["values"] == [1.0, 16.0] and m["total"] == 17.0


def test_the_cut_counts_calls_issued_before_k_in_full_and_lists_straddling_and_unknown():
    """C131.CUT control: counted 2, excluded 1, straddling 1, unknown 1; the original's 5 s and 11 input tokens once,
    the zero-token recovery row a carrier only."""
    rows = [row("before", 1, "model:small_model", answered_step=2, wall_s=5.0, tokens_in=11, tokens_out=3,
                status="answered"),
            row("late", 4, "model:small_model", answered_step=None, wall_s=None, tokens_in=None, tokens_out=None,
                status="timed_out"),
            row("after", 6, "model:small_model", answered_step=7, wall_s=1.0, tokens_in=9, tokens_out=1),
            row("recovery", 1, "model:small_model", answered_step=2, wall_s=0.001, tokens_in=0, tokens_out=0,
                recovers="before")]
    c = A.cut(rows, 5)
    assert [r.id for r in c["counted"]] == ["before", "late"] and [r.id for r in c["excluded"]] == ["after"]
    assert [r.id for r in c["unknown_placement"]] == ["late"] and [r.id for r in c["carriers"]] == ["recovery"]
    assert [r.id for r in A.cut(rows, 2)["straddling"]] == ["before"]
    tin = A.total(c["counted"], "tokens_in")
    assert (tin["total"], tin["known_subtotal"], tin["unknown"]) == (None, 11, 1)
    assert A.total(c["counted"] + c["carriers"], "wall_s")["known_subtotal"] == 5.0


def test_work_before_the_first_action_counts_in_every_ending_even_k_zero():
    rows = [row("reset", None, cpu_s=0.2, wall_s=0.3), row("act0", 0, cpu_s=0.1, wall_s=0.1)]
    assert [r.id for r in A.cut(rows, 0)["counted"]] == ["reset"]


def test_covered_resources_are_not_counted_twice_and_purposes_are_views():
    proc = row("proc", 3, "procedure", cpu_s=0.5, wall_s=1.0, measured=A.TIME)
    pred = row("pred", 3, "tool", cpu_s=0.2, wall_s=0.4, covered=("wall_s",), parent="proc", measured=A.TIME)
    trial = row("trial", None, "world", purpose="trial", cpu_s=2.0, wall_s=2.0, measured=A.TIME)
    v = A.resource_vector([proc, pred, trial])
    assert v["cpu_s"]["total"] == pytest.approx(2.7)            # disjoint child and parent processor time add
    assert v["wall_s_diagnostic"]["total"] == pytest.approx(3.0)  # the prediction's wait is inside the procedure's
    views = A.by_purpose([proc, pred, trial])
    assert set(views) == {"running", "trial"} and views["trial"]["cpu_s"]["total"] == 2.0


def test_model_vectors_keep_roles_failures_and_unknowns_apart():
    rows = [row("a", 0, "model:reasoning_model", tokens_in=500, tokens_out=8192, status="cut_off",
                measured=("wall_s", "tokens_in", "tokens_out"), wall_s=385.0),
            row("b", 0, "model:small_model", tokens_in=600, tokens_out=None, status="answered",
                measured=("wall_s", "tokens_in", "tokens_out"), wall_s=0.2),
            row("gpu", None, "allocation", purpose="setup", gpu_h=4.5, measured=("gpu_h",))]
    v = A.resource_vector(rows)
    assert v["models"]["reasoning_model"]["not_answered"] == 1
    assert v["models"]["small_model"]["tokens_out"] == {"total": None, "known_subtotal": 0, "unknown": 1}
    assert v["gpu_h"]["total"] == 4.5 and v["cpu_s"]["total"] == 0


def test_denominators_keep_failed_censored_missing_declined_and_unsettled_apart():
    sched = ["e0", "e1", "e2", "e3", "e4", "e5"]
    records = {
        "e0": {"status": "ok", "complete": True, "outcome": {"goal_met": True, "settled": True,
                                                              "goal_met_options": [True]}},
        "e1": {"status": "ok", "complete": True, "condition_outcome": {"declined": {"t": 0, "reason": "wind"}},
               "outcome": {"goal_met": False, "settled": True, "goal_met_options": [False]}},
        "e2": {"status": "ok", "complete": True, "outcome": {"goal_met": True, "settled": False,
                                                              "goal_met_options": [False, True]}},
        "e3": {"status": "ok", "complete": False, "outcome": {"goal_met": False}},
        "e4": {"status": "policy_failed", "complete": False, "outcome": None},
    }
    d = A.denominators(sched, records)
    assert d["counts"] == {"scheduled": 6, "attempted": 5, "natural_complete": 3, "policy_failed": 1, "censored": 1,
                           "not_attempted": 1, "declined_complete": 1}
    assert d["goals"] == {"settled_met": 1, "settled_not_met": 1, "unsettled": 1, "nominal_met": 2, "unavailable": 0}
    assert d["views"]["settled_met_over_scheduled"] == pytest.approx(1 / 6)
    assert A.cost_per_success(12.0, 0) is None and A.cost_per_success(None, 2) is None


def test_the_fixed_reader_gives_rows_and_combined_method_seconds(tmp_path):
    import paper2.experiment2.comparators as K
    from .test_comparators import scenario
    K.run_pilot_episode(scenario(), K.conditions()["calculator"], tmp_path / "f", "s")
    r = A.read_fixed(tmp_path / "f")
    m = r["method_seconds"]
    assert m["K"] == 12 and m["counts"]["calculator"] + m["counts"]["holding_none"] == 12
    assert m["counts"]["procedure"] is None
    reset = [x for x in r["rows"] if x.step is None]
    assert len(reset) == 1 and reset[0].process == "policy"
    assert len([x for x in r["rows"] if x.process == "world"]) == 12
    assert A.resource_vector(r["rows"])["cpu_s"]["total"] is not None


def test_the_telohelm_reader_uses_the_saved_ledger_and_the_attribution(tmp_path):
    import paper2.experiment2.run_stream as R
    R.run_stream(1, 1, tmp_path / "s", max_steps=30)
    ep = next(p for p in (tmp_path / "s").iterdir() if p.is_dir() and (p / "ledger.jsonl").exists())
    r = A.read_telohelm(ep)
    assert r["method_seconds"]["K"] == 30 and "world stepping" in r["unmeasured"]
    procs = [x for x in r["rows"] if x.process in ("procedure", "check")]
    assert procs and all(x.cpu_s is not None and x.wall_s is not None for x in procs)   # timings come from the
    assert A.resource_vector(r["rows"])["cpu_s"]["total"] is not None                     # episode log



def test_learning_rows_stay_out_of_flight_cuts_and_are_allocated_at_their_completed_episode(tmp_path):
    import json as _json
    build = row("rewrite-1", 0, "model:reasoning_model", purpose="build", tokens_in=600, tokens_out=9000,
                status="answered", measured=("wall_s", "tokens_in", "tokens_out"), wall_s=300.0)
    flight = row("fly-0", 0, "model:small_model", tokens_in=10, tokens_out=2, status="answered",
                 measured=("wall_s", "tokens_in", "tokens_out"), wall_s=0.1)
    c = A.cut([build, flight], 5)
    assert [r.id for r in c["counted"]] == ["fly-0"] and [r.id for r in c["other_purpose"]] == ["rewrite-1"]
    rewrite = A.Row("rewrite-1", "telohelm", "e1", "build", "model:reasoning_model", 0, tokens_out=9000,
                    measured=("tokens_out",))
    alloc = A.stream_allocation(["e0", "e1", "e2", "e3"], {"e0": 2.0, "e1": 2.0, "e2": 2.0, "e3": 2.0}, [rewrite],
                                "tokens_out")
    assert alloc["available"] and alloc["amortised"] == [2.0, 3002.0, 3002.0, 3002.0]
    unknown = A.Row("rewrite-2", "telohelm", "e1", "build", "model:reasoning_model", 0, tokens_out=None,
                    measured=("tokens_out",))
    assert A.stream_allocation(["e0", "e1"], {"e0": 1.0, "e1": 1.0}, [unknown], "tokens_out")["available"] is False
    (tmp_path / "learning-ledger.jsonl").write_text(_json.dumps({
        "request_id": "r", "kind": "reasoning_model", "episode": "e1", "scope": "build", "action_step": 0,
        "status": "cut_off", "wall_s": 385.0, "tokens_in": 500, "tokens_out": 8192}) + "\n")
    learned = A.read_learning(tmp_path)
    assert learned["rows"][0].purpose == "build" and "replay tests" in learned["unmeasured"]
    assert "proposer calls (no learning-ledger.jsonl)" in A.read_learning(tmp_path / "missing")["unmeasured"]


def test_an_old_summary_without_the_whole_outcome_is_unavailable_not_settled():
    d = A.denominators(["e"], {"e": {"status": "ok", "complete": True, "goal_met": True}})
    assert d["goals"]["unavailable"] == 1 and d["goals"]["settled_met"] == 0
