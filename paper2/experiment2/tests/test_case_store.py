"""The goal monitor's case store (note B8, D29): structure, trusted admission, audit, recovery, seeds, schedule and
routing to the replay test. Invented cases and an invented verifier; no world, label solver or model."""
from __future__ import annotations

import telohelm.learning_monitor as GM  # replay qualification moved to the monitor

import copy

import pytest

import telohelm.learning as G
import telohelm.qualification.store as H
from telohelm.runtime.library import Library


def case(split="stored_right", decision="heading", kind=None, ordinal=0, rewrite=None, heading=90.0,
         stream=4301, disposition=None):
    disposition = disposition or ("right" if split in ("stored_right",) else "wrong")
    if split in H.WRONG_SPLITS and kind is None:
        kind = "geometry"
    output = {"heading_deg": heading} if decision == "heading" else {"report_trigger_nm": 25.0}
    origin = {"episode": "dev-e00", "second": 12, "generator": f"invented-{split}", "seed": 1, "ordinal": ordinal,
              "attempt": 0}
    if rewrite is not None:
        origin["rewrite"] = rewrite
    return {"schema": H.SCHEMA, "stream": stream, "split": split, "decision": decision, "kind": kind,
            "interface": "action", "goal": {"version": 1, "tanker": "SHELL 1", "report_nm": 25},
            "context": {"t": 12, "goal_version": 1, "probe": ordinal}, "output": output, "origin": origin,
            "label": {"disposition": disposition, "reason": "invented", "rule": "fixture",
                      "evidence": {"invented": True}}}


def trusting(tmp_path):
    return H.CaseStore(tmp_path / "store", verifier=lambda c: True, verifier_name="fixture")


def test_structure_is_checked_before_any_verifier(tmp_path):
    assert H.case_errors(case()) == []
    bad = []
    c = case(); del c["goal"]; bad.append(c)
    bad.append(case(split="stored_right", disposition="wrong"))
    bad.append(case(split="stored_wrong", disposition="right"))
    bad.append(case(disposition="unknown"))
    bad.append(case(heading=360.0))
    c = case(); c["output"] = {"heading_deg": True}; bad.append(c)
    c = case(decision="report"); c["output"] = {"report_trigger_nm": 0}; bad.append(c)
    c = case(); c["context"]["goal_version"] = 2; bad.append(c)
    bad.append(case(split="fresh_wrong"))                                   # no rewrite ordinal
    c = case(); c["interface"] = "sequence"; bad.append(c)
    c = case(); c["stream"] = 4401; bad.append(c)
    c = case(split="stored_wrong"); c["kind"] = "made-up"; bad.append(c)
    for c in bad:
        assert H.case_errors(c), c


def test_admission_needs_a_trusted_verifier_and_refuses_conflicts(tmp_path):
    with pytest.raises(H.StoreError):
        H.CaseStore(tmp_path / "a").admit(case())                            # no verifier at all
    with pytest.raises(H.StoreError):
        H.CaseStore(tmp_path / "b", verifier=lambda c: "yes").admit(case())  # only True authorises
    s = trusting(tmp_path)
    cid = s.admit(case())
    assert s.admit(case()) == cid and len(s.cases()) == 1                  # the same case again is a no-op
    other = case(heading=91.0)                                               # same typed identity, other content
    with pytest.raises(H.StoreError):
        s.admit(other)


def test_every_attempt_is_audited_with_its_cost(tmp_path):
    s = trusting(tmp_path)
    row = {"stream": 4301, "split": "stored_right", "decision": "heading", "kind": None, "ordinal": 0}
    s.attempt(row, "unknown", "bounds did not settle", cost={"cpu_s": 0.4})
    s.attempt(row, "admitted", "settled", cost={"cpu_s": 0.3}, case=s.admit(case()))
    cov = s.coverage(4301)
    assert cov["attempts"] == {"unknown": 1, "admitted": 1}
    assert cov["admitted"] == {"stored_right|heading|None": 1}
    with pytest.raises(H.StoreError):
        s.attempt(row, "maybe", "")


def test_a_torn_record_is_removed_and_a_damaged_one_stops_writing(tmp_path):
    s = trusting(tmp_path)
    s.admit(case())
    with s.log_path.open("ab") as f:
        f.write(b"12 abc {\"half")
    again = H.CaseStore(tmp_path / "store", verifier=lambda c: True)
    with pytest.raises(H.StoreError):
        again.admit(case(ordinal=1))
    assert again.recover() > 0 and len(again.cases()) == 1
    again.admit(case(ordinal=1))
    data = again.log_path.read_bytes().replace(b"invented", b"inventeX", 1)
    again.log_path.write_bytes(data)
    damaged = H.CaseStore(tmp_path / "store", verifier=lambda c: True)
    with pytest.raises(H.StoreError):
        damaged.admit(case(ordinal=2))


def test_seeds_and_schedule_are_fixed_before_any_draw():
    assert H.base_seed(4301, "stored_right") == 5001 and H.base_seed(4304, "eligibility") == 7004
    with pytest.raises(H.StoreError):
        H.base_seed(4401, "stored_right")
    a = H.draw_seed(4301, "fresh_wrong", "heading", "stale", 3, 0, rewrite=2)
    assert a == H.draw_seed(4301, "fresh_wrong", "heading", "stale", 3, 0, rewrite=2)
    assert a != H.draw_seed(4301, "fresh_wrong", "heading", "stale", 3, 0, rewrite=3)
    assert a != H.draw_seed(4302, "fresh_wrong", "heading", "stale", 3, 0, rewrite=2)
    rows = H.schedule(4301)
    by = lambda split: [r for r in rows if r["split"] == split]
    assert len(by("stored_wrong")) == 200 and len(by("final")) == 200
    assert all(sum(r["kind"] == k for r in by("stored_wrong")) == 40 for k in H.KINDS)
    assert len(by("stored_right")) == 200 and len(by("eligibility")) == 200
    assert {r["cap"] for r in rows} == {H.ATTEMPTS_PER_CASE}


def stocked(tmp_path, rights=100, fresh_for=1):
    s = trusting(tmp_path)
    for i in range(rights):
        s.admit(case("stored_right", "heading", ordinal=i))
        s.admit(case("stored_right", "report", ordinal=i))
    s.admit(case("stored_wrong", "heading", ordinal=0, heading=270.0))
    s.admit(case("stored_wrong", "report", kind="identity", ordinal=1))
    for i in range(100):
        s.admit(case("fresh_wrong", "heading", kind="stale", ordinal=i, rewrite=fresh_for, heading=180.0))
    s.admit(case("eligibility", "heading", ordinal=0, disposition="right"))
    s.admit(case("final", "heading", ordinal=0, heading=10.0))
    return s


def test_replay_routes_only_the_checks_own_decision(tmp_path):
    s = stocked(tmp_path)
    wrong, right, missing = s.replay_cases(4301, "heading", 1)
    assert missing == [] and len(right) == 100 and len(wrong) == 101
    assert all(set(c) == {"context", "output"} for c in wrong + right)      # payload only, no label or split
    assert all("heading_deg" in c["output"] for c in wrong + right)          # no report case reaches it
    assert {"heading_deg": 10.0} not in [c["output"] for c in wrong]          # final cases never replayed
    _, _, missing = s.replay_cases(4301, "report", 1)
    assert missing == ["0 fresh wrong cases for rewrite 1, not 100"]
    _, _, missing = s.replay_cases(4301, "heading", 2)                       # fresh cases belong to rewrite 1
    assert missing == ["0 fresh wrong cases for rewrite 2, not 100"]


def test_too_few_right_cases_or_no_wrong_case_closes_the_gate(tmp_path):
    s = stocked(tmp_path, rights=99)
    _, _, missing = s.replay_cases(4301, "heading", 1)
    assert missing == ["99 right cases for heading, not 100"]
    empty = trusting(tmp_path / "e")
    _, _, missing = empty.replay_cases(4301, "heading", None)
    assert "no settled wrong case for heading" in missing


def coordinator_with(tmp_path, store):
    lib = Library(tmp_path / "library")
    return G.Coordinator(lib, tmp_path, None, lambda s: True, lambda a: {}, corpus=store, stream=4301), lib


def test_check_replay_uses_the_store_for_its_decision_and_rewrite(tmp_path, monkeypatch):
    co, lib = coordinator_with(tmp_path, stocked(tmp_path))
    seen = {}
    def fake(source, wrong, right, old_source=None, **kw):
        seen.update(wrong=len(wrong), right=len(right))
        return {"passed": True, "right_accepted": 100, "old_right_accepted": None, "wrong_accepted": [],
                "wrong_cases": len(wrong), "right_cases": len(right)}
    monkeypatch.setattr(GM, "replay_test", fake)
    ok, detail = co.check_replay("def check(ctx, out):\n    return 'accept'\n", None, "heading", 1)
    assert ok and seen == {"wrong": 101, "right": 100}
    ok, detail = co.check_replay("def check(ctx, out):\n    return 'accept'\n", None, "report", 1)
    assert not ok and "fresh" in detail["refused"]


def test_a_real_replay_rejects_a_check_that_accepts_everything(tmp_path):
    co, lib = coordinator_with(tmp_path, stocked(tmp_path))
    ok, detail = co.check_replay("def check(ctx, out):\n    return 'accept'\n", None, "heading", 1)
    assert not ok


def test_confirm_wrong_admits_the_case_and_rechecks_only_its_decision(tmp_path, monkeypatch):
    store = stocked(tmp_path)
    co, lib = coordinator_with(tmp_path, store)
    hcheck = lib.add({"kind": "check", "decision": "heading", "conditions": [],
                      "source": "def check(ctx, out):\n    return 'accept'\n", "origin": {"predecessor": None}})
    lib.set_status(hcheck, "adopted", "invented")
    seen = []
    monkeypatch.setattr(GM, "replay_test", lambda src, wrong, right, **kw: seen.append(
        {o for c in wrong for o in c["output"]}) or {"wrong_accepted": [0]})
    co.confirm_wrong(case("confirmed_wrong", "heading", kind="stale", ordinal=7, heading=200.0))
    assert len(store.cases(4301, "confirmed_wrong")) == 1
    assert seen == [{"heading_deg"}]
    assert lib.projection()["entries"][hcheck]["status"] == "retired"


def proposal(lower_after=30.0, upper_best=12.0, upper_after=40.0, lower_best=10.0):
    from paper2.experiment2.monitor_labels import Bounds; from paper2.experiment2.monitor_labels import Original; from paper2.experiment2.monitor_labels import propose_heading
    h = "a" * 64
    return propose_heading(Original("dev-e00", 1, 12, h, h, h, h, h),
                           Bounds(12, lower_after=lower_after, upper_best=upper_best, upper_after=upper_after,
                                  lower_best=lower_best))


def test_unverified_label_proposals_are_never_admitted(tmp_path):
    """D30: Codex's label arithmetic proposes; it does not verify. A trusting verifier cannot change that."""
    from dataclasses import asdict
    p = proposal()
    assert p.disposition == "proposed" and p.conditional_label == "wrong" and p.admissible is False
    s = trusting(tmp_path)
    with pytest.raises(H.StoreError, match="not a case"):
        s.admit(p)                                                            # the proposal object itself
    carried = case("stored_wrong", kind="geometry")
    carried["label"]["evidence"] = {"proposal": asdict(p)}                   # a proposal passed off as evidence
    named = case("stored_wrong", kind="geometry", ordinal=1)
    named["label"]["rule"] = "B8.heading.action.v1"                          # the scaffold's rule named directly
    for c in (carried, named):
        assert any("unverified proposal" in e for e in H.case_errors(c))
        with pytest.raises(H.StoreError, match="unverified proposal"):
            s.admit(c)
    assert s.cases() == [] and s.records == []                               # nothing written



def test_audit_rows_and_re_admission_are_independent_of_the_callers_objects(tmp_path):
    """B8.STORE.IMMUTABLE, the paths Codex's witnesses do not cover: an audit row and its cost, and a caller who
    changes a case after admitting it and offers it again."""
    s = trusting(tmp_path)
    row = {"stream": 4301, "split": "stored_right", "decision": "heading", "kind": None, "ordinal": 0}
    cost = {"cpu_s": 0.4}
    s.attempt(row, "unknown", "bounds did not settle", cost=cost)
    row["split"], cost["cpu_s"] = "final", 99.0
    assert s.coverage(4301)["attempts"] == {"unknown": 1}
    assert [r for r in s.records if r["record"] == "attempt"][0]["cost"] == {"cpu_s": 0.4}
    c = case()
    cid = s.admit(c)
    c["output"]["heading_deg"] = 91.0                                        # the caller changes its own copy
    with pytest.raises(H.StoreError, match="conflicting"):
        s.admit(c)                                                           # a different case, same identity
    assert s.admit(case()) == cid                                            # the original again is a no-op
    again = H.CaseStore(s.root, verifier=lambda c: True)
    assert again.cases() == s.cases() and again.cases()[0]["output"] == {"heading_deg": 90.0}
    with pytest.raises(H.StoreError, match="not a JSON record"):
        s.admit({**case(ordinal=5), "context": {"t": float("nan")}})


def test_same_stream_confirmation_still_undoes_and_survives_restart(tmp_path, monkeypatch):
    """B8.CONFIRM.SCOPE must not block the ordinary path: a confirmed wrong answer of the coordinator's own stream
    is admitted, re-checks only its decision's adopted check, and is read back unchanged after a restart."""
    store = stocked(tmp_path)
    co, lib = coordinator_with(tmp_path, store)
    hcheck = lib.add({"kind": "check", "decision": "heading", "conditions": [],
                      "source": "def check(ctx, out):\n    return 'accept'\n", "origin": {"predecessor": None}})
    rcheck = lib.add({"kind": "check", "decision": "report", "conditions": [],
                      "source": "def check(ctx, out):\n    return 'accept'\n", "origin": {"predecessor": None}})
    lib.set_status(hcheck, "adopted", "invented")
    lib.set_status(rcheck, "adopted", "invented")
    # Each check accepts everything, so a replay "accepts" a wrong case only if the new confirmed one is among them.
    monkeypatch.setattr(GM, "replay_test", lambda src, wrong, right, **kw: {
        "wrong_accepted": [i for i, c in enumerate(wrong) if c["output"] == {"heading_deg": 200.0}]})
    confirmed = case("confirmed_wrong", "heading", kind="stale", ordinal=8, heading=200.0)
    co.confirm_wrong(confirmed)
    states = {k: v["status"] for k, v in lib.projection()["entries"].items()}
    assert states[hcheck] == "retired" and states[rcheck] == "adopted"     # only the heading check is re-checked
    reread = H.CaseStore(store.root).cases(4301, "confirmed_wrong")
    assert reread == [confirmed]


def test_conditional_contact_evidence_is_never_admitted(tmp_path):
    """D31: Codex's contact lower bound is conditional and inadmissible; carried in a label, it is refused."""
    from dataclasses import asdict
    from paper2.experiment2.contact_bounds import contact_lower
    from paper2.experiment2.monitor_labels import Original
    h = "a" * 64
    ev = contact_lower(Original("dev-e00", 1, 12, h, h, h, h, h), lambda s: (30.0, 0.0), lambda s: (0.0, 0.0),
                       0.15, 0.08, 400.0)
    assert ev.admissible is False and ev.qualification.startswith("conditional")
    c = case("stored_wrong", kind="geometry")
    c["label"]["evidence"] = {"contact": {k: v for k, v in asdict(ev).items() if k != "admissible"}}
    s = trusting(tmp_path)
    with pytest.raises(H.StoreError, match="unverified proposal"):
        s.admit(c)
    assert s.cases() == []
