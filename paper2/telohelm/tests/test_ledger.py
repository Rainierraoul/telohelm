"""Tests for the cost ledger (backlog note B10), with hand-worked numbers."""
from __future__ import annotations

import pytest

import telohelm.runtime.ledger as G


def call(rid, start, wall, kind="reasoning_model", step=0, episode="e1", gpus=(0,), allocation="job1"):
    return G.Call(rid, kind, episode, step, start, wall, 100, 50, "answered", allocation=allocation, gpus=gpus)


def test_codex_example_busy_and_idle_reconcile():
    # A = 10 h on one GPU (36,000 s); 2 overlapping calls 2:1 in time, active 8 h in all, so I = 2 h.
    lg = G.Ledger()
    lg.add(call("a", 0.0, 8 * 3600.0))                      # 0 to 8 h
    lg.add(call("b", 0.0, 4 * 3600.0))                      # 0 to 4 h, overlapping a
    out = lg.gpu_shares(10.0, [("job1", 0.0, 36000.0, 1)])
    assert out["idle_h"] == pytest.approx(2.0)
    assert out["shares_h"]["a"] == pytest.approx(16 / 3) and out["shares_h"]["b"] == pytest.approx(8 / 3)
    assert sum(out["shares_h"].values()) + out["idle_h"] == pytest.approx(10.0)


def test_idle_counts_each_gpu_by_placement():
    one = G.Ledger()
    one.add(call("a", 0.0, 1800.0, gpus=(0,)))               # 2 GPUs for 1 h; the call uses GPU 0 for half an hour
    out = one.gpu_shares(2.0, [("job1", 0.0, 3600.0, 2)])
    assert out["idle_h"] == pytest.approx(1.5) and out["busy_h"] == pytest.approx(0.5)
    both = G.Ledger()
    both.add(call("a", 0.0, 1800.0, gpus=(0, 1)))
    out = both.gpu_shares(2.0, [("job1", 0.0, 3600.0, 2)])
    assert out["idle_h"] == pytest.approx(1.0) and out["busy_h"] == pytest.approx(1.0)


def test_codex_placement_derivation():
    """R22.6: 2 GPUs for 1 h and one 1-hour request: B = 1, I = 1 on one GPU; B = 2, I = 0 on both."""
    for gpus, busy in (((0,), 1.0), ((0, 1), 2.0)):
        lg = G.Ledger()
        lg.add(call("r", 0.0, 3600.0, gpus=gpus))
        out = lg.gpu_shares(2.0, [("job1", 0.0, 3600.0, 2)])
        assert out["busy_h"] == pytest.approx(busy) and out["idle_h"] == pytest.approx(2.0 - busy)


def test_unknown_placement_withholds_busy_idle_and_shares():
    lg = G.Ledger()
    lg.add(call("a", 0.0, 100.0, gpus=None))
    out = lg.gpu_shares(1.0, [("job1", 0.0, 3600.0, 1)])
    assert out["busy_h"] is None and out["idle_h"] is None and out["shares_h"] is None


def test_allocation_must_match_its_windows_and_be_valid():
    with pytest.raises(ValueError):
        G.Ledger().gpu_shares(5.0, [("job1", 0.0, 3600.0, 1)])
    with pytest.raises(ValueError):
        G.Ledger().gpu_shares(1.0, [("job1", 0.0, float("nan"), 1)])
    with pytest.raises(ValueError):
        G.Call("x", "reasoning_model", "e1", 0, -1.0, 1.0, 1, 1, "answered")


def test_unknown_time_withholds_shares_and_keeps_raw_null():
    lg = G.Ledger()
    lg.add(call("a", 0.0, 100.0))
    lg.add(G.Call("b", "reasoning_model", "e1", 1, 50.0, None, None, None, "interrupted", allocation="job1", gpus=(0,)))
    out = lg.gpu_shares(1.0, [("job1", 0.0, 3600.0, 1)])
    assert out["shares_h"] is None and out["calls_unknown_time_or_placement"] == ["b"] and out["idle_h"] is None
    assert lg.calls[1].wall_s is None


def test_prediction_calls_take_no_gpu_share():
    lg = G.Ledger()
    lg.add(call("a", 0.0, 100.0))
    lg.add(call("p", 10.0, 0.001, kind="prediction_tool"))
    assert set(lg.gpu_shares(1.0, [("job1", 0.0, 3600.0, 1)])["shares_h"]) == {"a"}


def test_ending_cut_off_before_at_and_after():
    lg = G.Ledger()
    for rid, step in (("before", 4), ("at", 5), ("after", 6)):
        lg.add(call(rid, 0.0, 1.0, step=step))
    # TURN 56: the cut also lists straddling calls; these records have no answer step, so it is unknown.
    # TURN 71: the cut also gives the charge-carrying rows (MG70.CUT.CHARGE); here the counted row alone.
    assert lg.ending_calls("e1", 5) == {"counted": ["before"], "charge_ids": ["before"], "excluded": ["at", "after"],
                                        "straddling": [], "straddling_unknown": ["before"]}


def test_amortised_view_reconciles_to_the_raw_total():
    running = [1.0, 1.0, 1.0, 1.0]
    charges = G.amortised({1: 3.0}, running)
    assert charges == [1.0, 2.0, 2.0, 2.0]
    assert sum(charges) == pytest.approx(sum(running) + 3.0)
    with pytest.raises(ValueError):
        G.amortised({4: 1.0}, running)


def test_cost_per_goal_met():
    assert G.per_goal_met(12.0, 3) == 4.0 and G.per_goal_met(12.0, 0) is None


def test_bad_records_are_refused():
    with pytest.raises(ValueError):
        G.Call("x", "frontier", "e1", 0, 0.0, 1.0, 1, 1, "answered")
    lg = G.Ledger()
    lg.add(call("a", 0.0, 1.0))
    with pytest.raises(ValueError):
        lg.add(call("a", 0.0, 1.0))


def test_no_active_requests_gives_zero_busy_and_all_idle():
    out = G.Ledger().gpu_shares(3.0, [("job1", 0.0, 3600.0, 3)])
    assert out["busy_h"] == pytest.approx(0.0) and out["idle_h"] == pytest.approx(3.0) and out["shares_h"] is None


def test_codex_unknown_end_counterexample_leaves_busy_and_idle_null():
    lg = G.Ledger()
    lg.add(G.Call("r", "reasoning_model", "e1", 0, 2 * 3600.0, None, 10, 10, "interrupted", allocation="job1", gpus=(0,)))
    out = lg.gpu_shares(10.0, [("job1", 0.0, 36000.0, 1)])
    assert out["busy_h"] is None and out["idle_h"] is None and out["shares_h"] is None


def test_l24_1_shares_follow_wall_time_whatever_the_device_count():
    """Two overlapping 1-hour calls, on one and on two GPUs, in a 2-GPU 1-hour allocation: B = 2, I = 0, and
    under B10's wall-time weights each call gets 1 GPU-hour (Codex's L24.1)."""
    lg = G.Ledger()
    lg.add(call("one", 0.0, 3600.0, gpus=(0,)))
    lg.add(call("two", 0.0, 3600.0, gpus=(0, 1)))
    out = lg.gpu_shares(2.0, [("job1", 0.0, 3600.0, 2)])
    assert out["busy_h"] == pytest.approx(2.0) and out["idle_h"] == pytest.approx(0.0)
    assert out["shares_h"]["one"] == pytest.approx(1.0) and out["shares_h"]["two"] == pytest.approx(1.0)


@pytest.mark.parametrize("gpus", [(-1,), (0, 0), (True,), ("0",), ()])
def test_l24_2_bad_device_ids_are_refused(gpus):
    with pytest.raises(ValueError):
        G.Call("x", "reasoning_model", "e1", 0, 0.0, 1.0, 1, 1, "answered", allocation="job1", gpus=gpus)


def test_l24_2_devices_and_allocations_must_exist():
    lg = G.Ledger()
    lg.add(call("a", 0.0, 10.0, gpus=(3,)))
    with pytest.raises(ValueError, match="outside its allocation"):
        lg.gpu_shares(2.0, [("job1", 0.0, 3600.0, 2)])
    other = G.Ledger()
    other.add(call("a", 0.0, 10.0, allocation="job9"))
    with pytest.raises(ValueError, match="not in the windows"):
        other.gpu_shares(2.0, [("job1", 0.0, 3600.0, 2)])


def test_two_allocations_with_the_same_device_numbers_are_kept_apart():
    lg = G.Ledger()
    lg.add(call("a", 0.0, 3600.0, gpus=(0,), allocation="job1"))
    out = lg.gpu_shares(2.0, [("job1", 0.0, 3600.0, 1), ("job2", 0.0, 3600.0, 1)])
    assert out["busy_h"] == pytest.approx(1.0) and out["idle_h"] == pytest.approx(1.0)


def test_l26_1_unhashable_device_is_refused_not_raised():
    with pytest.raises(ValueError):
        G.Call("x", "reasoning_model", "e1", 0, 0.0, 1.0, 1, 1, "answered", allocation="job1", gpus=([0],))


def test_l26_2_overlapping_windows_of_one_allocation_are_refused():
    lg = G.Ledger()
    lg.add(call("a", 0.0, 3600.0))
    with pytest.raises(ValueError, match="overlap"):
        lg.gpu_shares(2.0, [("job1", 0.0, 3600.0, 1), ("job1", 0.0, 3600.0, 1)])
    out = lg.gpu_shares(2.0, [("job1", 0.0, 3600.0, 1), ("job2", 0.0, 3600.0, 1)])   # different jobs may overlap
    assert out["busy_h"] == pytest.approx(1.0)
