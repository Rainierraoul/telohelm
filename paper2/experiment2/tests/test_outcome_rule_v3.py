"""dev.outcome.v3: labels settled in quiet continuations, the same for the same context whichever episode it comes
from; headings judged against the reference pilot's own heading held for the same time, with a band of marginal
delays left out; content-negative reports; a fixed source partition (paper2/notes/OUTCOME-RULE-V3.md)."""
import copy
import json
from pathlib import Path

import pytest

import paper2.experiment2.devcases as DC
import paper2.experiment2.outcomecases as O


def test_a_heading_is_right_within_15_seconds_wrong_beyond_60_and_marginal_between():
    assert O._disposition(100, 50) == "right"           # better than the baseline
    assert O._disposition(100, 115) == "right"
    assert O._disposition(100, 116) == "marginal"
    assert O._disposition(100, 160) == "marginal"
    assert O._disposition(100, 161) == "wrong"
    assert O._disposition(100, None) == "wrong"          # the join is lost


def _row(disposition, x, heading=90.0):
    return {"disposition": disposition, "decision": "heading", "output": {"heading_deg": heading},
            "moment": {"ctx": {"x": x}, "phase": 1}}


def test_marginal_and_unknown_rows_are_left_out_and_counted():
    rows = [_row("right", 1), _row("wrong", 2), _row("marginal", 3), _row("unknown", 4), _row("marginal", 5)]
    kept, counts = O._keep(rows)
    assert [r["disposition"] for r in kept] == ["right", "wrong"]
    assert counts == {"labelled": 5, "unknown": 1, "marginal": 2, "conflicting": 0,
                      "by_phase": {"p1|heading|marginal": 2, "p1|heading|right": 1, "p1|heading|unknown": 1,
                                   "p1|heading|wrong": 1}}


def test_opposite_labels_on_the_same_context_and_output_are_still_both_left_out():
    kept, counts = O._keep([_row("right", 1), _row("wrong", 1), _row("right", 2)])
    assert [r["moment"]["ctx"]["x"] for r in kept] == [2] and counts["conflicting"] == 2


@pytest.fixture(scope="module")
def moments():
    return O.moments(4400, 1, phases=(1,))


def test_the_baseline_is_the_reference_heading_held_for_the_same_time(moments):
    m = moments[2]
    p = O._prefix(m["source"], m["phase"], m["index"], m["t"])
    assert O._reference_heading(p) == m["reference_heading"]
    ep, pilot, _, bundle = (copy.deepcopy(x) for x in p)
    held = O._joined_at(O._continue(ep, pilot, bundle, heading=m["reference_heading"], hold=O.HOLD_S))
    assert O._base_from(p) == held is not None


def test_the_reference_heading_is_right_at_every_moment_it_can_be_judged(moments):
    for m in moments[:6]:
        for r in O._label_moment(m):
            if r["decision"] == "heading" and r["evidence"]["construction"] == "reference":
                assert r["disposition"] == "right", (m["episode"], m["t"], r["evidence"])
                assert r["evidence"]["join_s"] == r["evidence"]["base_join_s"]


def test_a_case_labelled_under_an_earlier_rule_does_not_verify(moments):
    rows = [r for r in O._label_moment(moments[2]) if r["disposition"] in ("right", "wrong")]
    case = O._case(4401, "stored_right" if rows[0]["disposition"] == "right" else "stored_wrong", rows[0], 0)
    assert case["label"]["rule"] == O.RULE == "dev.outcome.v3" and O.verify(case) is True
    for old_rule in ("dev.outcome.v1", "dev.outcome.v2"):
        old = json.loads(json.dumps(case))
        old["label"]["rule"] = old_rule
        assert O.verify(old) is False


def test_the_quiet_continuation_removes_only_what_is_still_to_come():
    sc = O._scenario(4400, 4, 0)
    assert [g for g in sc.goals if g[0] > 30], "a phase 4 episode has a goal change after 30 s"
    p = O._prefix_of(sc, 30)
    family = O._family(sc, 30, O._context(p))
    assert family and all(O._context(w) == O._context(p) for w in family)
    first = family[0][0].s
    assert all(g[0] <= 30 for g in first.goals) and first.delay_s == sc.delay_s


def test_an_eventful_episode_and_its_quiet_version_label_the_same_second_alike():
    # Codex's TURN207 witness: source 4400, phase 4, episode 0, second 30 gives the same context with and without the
    # later goal change; under v2 an early report was left out in one and kept as wrong in the other.
    sc = O._scenario(4400, 4, 0)
    quiet = O._quiet(sc, 30)
    assert O._context(O._prefix_of(sc, 30)) == O._context(O._prefix_of(quiet, 30))
    d = sc.goals[0][1].report_at_nm
    tanker = sc.goals[0][1].tanker
    cands = [("report", {"report": DC.report_text(sc.control_callsign, d, tanker)}), ("report", {}),
             ("report", {"report_trigger_nm": d}), ("heading", {"heading_deg": 90.0})]
    assert [x[0] for x in O._labels(sc, 30, cands)] == [x[0] for x in O._labels(quiet, 30, cands)]


def test_the_radar_delays_compared_are_those_the_context_cannot_rule_out():
    import paper2.experiment2.streams as streams
    sc = O._scenario(4400, 1, 0)
    assert sc.delay_s == 0
    at_0 = O._family(sc, 0, O._context(O._prefix_of(sc, 0)))
    assert [w[0].s.delay_s for w in at_0] == [0] + list(streams.DELAYS)     # nothing has arrived yet
    at_30 = O._family(sc, 30, O._context(O._prefix_of(sc, 30)))
    assert [w[0].s.delay_s for w in at_30] == [0]                            # reports have arrived with no delay


def test_a_trigger_at_second_0_is_unknown_because_the_delay_cannot_yet_be_seen():
    sc = O._scenario(4400, 1, 0)
    d = sc.goals[0][1].report_at_nm
    disp, ev = O._labels(sc, 0, [("report", {"report_trigger_nm": d})])[0]
    assert disp == "unknown" and "disagree" in ev["reason"] and len(ev["delays_compared"]) == 4
    assert O._labels(sc, 30, [("report", {"report_trigger_nm": d})])[0][0] == "right"


def test_at_the_reference_reporting_second_only_the_content_decides_a_report():
    ms = [m for m in O.moments(4400, 1, phases=(1,)) if m["reporting"]]
    assert ms, "the reference's reporting second is a moment"
    rows = {(r["evidence"]["construction"], r["evidence"].get("field")): r["disposition"]
            for r in O._label_moment(ms[0]) if r["decision"] == "report"}
    assert rows[("report_now", None)] == "right"
    assert rows[("wrong_content", "controller")] == rows[("wrong_content", "distance")] == "wrong"
    assert rows[("wrong_content", "tanker")] == "wrong"


def test_the_source_partition_is_disjoint_and_keeps_out_what_designed_the_rule():
    part = O.partition()
    assert set(part) == set(O.RUN_STREAMS)
    used = [x for roles in part.values() for block in roles.values() for x in block]
    assert len(used) == len(set(used)) and not set(used) & set(O.C0_DEVELOPMENT)
    assert all(len(part[s]["replay"]) == 5 for s in part)
    with pytest.raises(ValueError):
        O.build(4499, Path("/nonexistent"))
    with pytest.raises(ValueError):
        O.build(4301, Path("/nonexistent"))


def test_duplicates_are_found_across_roles():
    a = {"context": {"t": 1}, "output": {}}
    b = {"context": {"t": 2}, "output": {}}
    assert O.duplicates([a, b], [b]) == [json.dumps([b["context"], b["output"]], sort_keys=True)]
    assert O.duplicates([a], [b]) == []


def test_doing_nothing_at_the_second_the_reference_would_report_lets_it_report_a_second_later():
    # Found by the reference report check (rewrite 20 of the stream 4401 store): source 4404, phase 1, index 0, at
    # 510 s the reference pilot reports in its own flight; the candidate 'nothing' takes that report away. The pilot
    # must be able to report again, as the rule says, and not end the episode with no report because it had marked
    # one sent. A second later the range is 9.68 NM against an ordered 10, so doing nothing there is still wrong.
    sc = O._scenario(4404, 1, 0)
    p = O._prefix_of(sc, 510)
    _, pilot, _, bundle = (copy.deepcopy(x) for x in p)
    assert "report" in pilot.act(bundle)                  # the reference would report at this very second
    ep, pilot, _, bundle = (copy.deepcopy(x) for x in p)
    o = O._continue(ep, pilot, bundle, report_action={})
    assert o["reports_under_goal"] == 1 and o["reports"][0]["t"] == 511 and o["report_correct"] is False
