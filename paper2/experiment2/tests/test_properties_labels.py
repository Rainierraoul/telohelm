"""Property tests of the outcome label rule's pure parts (study2/outcomecases.py), with Hypothesis.

Development tests only. The labelling itself flies the world and is tested by example in test_outcome_rule_v3.py;
these state rules over every input for the parts that decide and count. Runs are derandomised so they repeat.
"""
import json

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st          # noqa: E402

import paper2.experiment2.outcomecases as O                               # noqa: E402

SETTINGS = settings(derandomize=True, max_examples=500, deadline=None)
RANK = {"wrong": 0, "marginal": 1, "right": 2}
EDGES = [O.RIGHT_DELAY_S - 1, O.RIGHT_DELAY_S, O.RIGHT_DELAY_S + 1, O.WRONG_DELAY_S - 1, O.WRONG_DELAY_S,
         O.WRONG_DELAY_S + 1, 0, -1]
bases = st.integers(0, 3000)
delays = st.one_of(st.none(), st.sampled_from(EDGES), st.integers(-200, 400))     # the band's edges, and around


@st.composite
def base_and_joins(draw, n):
    base = draw(bases)
    return base, [None if d is None else base + d for d in (draw(delays) for _ in range(n))]


@SETTINGS
@given(base_and_joins(2))
def test_a_later_join_never_gets_a_better_label(bj):
    base, (a, b) = bj
    later, earlier = (a, b) if (a is None or (b is not None and a >= b)) else (b, a)
    assert RANK[O._disposition(base, later)] <= RANK[O._disposition(base, earlier)]


@SETTINGS
@given(base_and_joins(1))
def test_the_label_follows_the_band_exactly(bj):
    base, (join,) = bj
    disp = O._disposition(base, join)
    if join is None or join - base > O.WRONG_DELAY_S:
        assert disp == "wrong"
    elif join - base <= O.RIGHT_DELAY_S:
        assert disp == "right"
    else:
        assert disp == "marginal"


row = st.builds(lambda disp, x, phase, decision: {"disposition": disp, "decision": decision,
                                                  "output": {"heading_deg": 90.0},
                                                  "moment": {"ctx": {"x": x}, "phase": phase}},
                st.sampled_from(["right", "wrong", "marginal", "unknown"]), st.integers(0, 6), st.integers(1, 4),
                st.sampled_from(["heading", "report"]))


@SETTINGS
@given(st.lists(row, max_size=40))
def test_every_row_is_kept_or_counted_once_and_no_kept_pair_disagrees(rows):
    kept, counts = O._keep(rows)
    assert all(r["disposition"] in ("right", "wrong") for r in kept)
    assert len(kept) + counts["unknown"] + counts["marginal"] + counts["conflicting"] == counts["labelled"] == len(rows)
    assert sum(counts["by_phase"].values()) == len(rows)
    labels: dict[str, set] = {}
    for r in kept:
        labels.setdefault(json.dumps([r["moment"]["ctx"], r["output"]], sort_keys=True), set()).add(r["disposition"])
    assert all(len(v) == 1 for v in labels.values())
