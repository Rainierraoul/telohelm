"""Hand-worked prospective B8 fixtures; no empirical bounds or world draws."""
from dataclasses import FrozenInstanceError, replace
import json
import pytest
from paper2.experiment2.monitor_labels import Original; from paper2.experiment2.monitor_labels import Bounds; from paper2.experiment2.monitor_labels import propose_heading

O = Original("invented-episode", 4, 180, *(["a" * 64] * 5))


def test_elapsed_example_requires_original_origin_not_absolute_finish():
    # Feasible original finishes at 210 (30 elapsed); after lower at 225 (45).
    p = propose_heading(O, Bounds(180, lower_after=45, upper_best=30))
    assert (p.disposition, p.conditional_label) == ("proposed", "wrong")
    assert not p.admissible and p.qualification == "unverified_bounds_and_witnesses"
    assert propose_heading(O, Bounds(0, 225, 210)).disposition == "invalid"


def test_near_optimal_whole_goal_witness_with_missing_wrong_branch():
    p = propose_heading(O, Bounds(180, upper_after=33, lower_best=30))
    assert p.conditional_label == "right" and not p.admissible
    assert propose_heading(O, Bounds(180)).disposition == "unknown"


@pytest.mark.parametrize("gap", [9, 10, 11])
def test_wrong_cutoff_band_is_unknown(gap):
    assert propose_heading(O, Bounds(180, lower_after=30+gap, upper_best=30)).disposition == "unknown"


@pytest.mark.parametrize("gap", [4, 5, 6])
def test_right_cutoff_band_is_unknown(gap):
    assert propose_heading(O, Bounds(180, upper_after=30+gap, lower_best=30)).disposition == "unknown"


@pytest.mark.parametrize("change", [dict(goal_changed=True), dict(unsettled=True), dict(complete_at_issue=True), dict(numerical_flags=("range near 1e-5 NM allowance",))])
def test_unsettled_physics_or_goal_never_becomes_a_label(change):
    assert propose_heading(O, Bounds(180, lower_after=50, upper_best=30, **change)).disposition == "unknown"


@pytest.mark.parametrize("value,status", [(True,"invalid"), (10**400,"invalid"), (-1,"invalid"), (float("nan"),"invalid"), (float("inf"),"unknown"), ("30","invalid")])
def test_unavailable_or_malformed_evidence_retains_auditable_disposition(value, status):
    p = propose_heading(O, Bounds(180, lower_after=value, upper_best=30))
    assert p.disposition == status and not p.admissible
    json.loads(p.evidence_json)


@pytest.mark.parametrize("b", [Bounds(180, 50, 30, 40), Bounds(180, upper_best=30, lower_best=40), Bounds(180, upper_best=40, upper_after=30), Bounds(180, upper_after=20, lower_best=30), Bounds(180, commitment_seconds=30), Bounds(180, units="absolute_seconds"), Bounds(180, goal_changed=1)])
def test_contradiction_units_and_longer_sequence_refuse(b):
    assert propose_heading(O, b).disposition == "invalid"


def test_originals_immutable_distinct_and_content_hash_not_authority():
    b = Bounds(180, lower_after=45, upper_best=30)
    a = propose_heading(O,b)
    other = propose_heading(replace(O, goal_version=5),b)
    assert a.original.goal_version == 4 and other.fingerprint != a.fingerprint
    with pytest.raises(FrozenInstanceError):
        a.original.second = 181
    assert not a.admissible and not other.admissible
    assert propose_heading(replace(O, input_sha256="pretend"), b).disposition == "invalid"
    assert propose_heading(replace(O, second=True), b).disposition == "invalid"
    assert propose_heading(replace(O, decision="report"), b).disposition == "invalid"


def test_nonfinite_original_second_is_invalid_with_record_retained():
    p = propose_heading(replace(O, second=float("nan")), Bounds(180))
    assert p.disposition == "invalid" and not p.admissible
