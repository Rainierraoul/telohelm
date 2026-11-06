"""Development cases (Simon, 14 October): the development store never mixes with a scored one, the development
verifier re-derives each label, and a real sandboxed replay on the built set accepts a sound check and refuses a
careless one. Built from development streams only; no scored stream, model or cluster."""
from __future__ import annotations

import copy

import pytest

import paper2.experiment2.devcases as D
import telohelm.learning as G
import telohelm.qualification.store as H
from telohelm.runtime.library import Library
from .test_case_store import case

LEAD_CHECK = '''
import math
def check(ctx, out):
    if ctx.report_x is None:
        return "reject"
    cx, cy, r, v = ctx.tanker_centre_x, ctx.tanker_centre_y, ctx.tanker_radius_nm, ctx.tanker_speed_nm_s
    th = math.atan2(ctx.report_x - cx, ctx.report_y - cy)
    tau = ctx.observed_range_nm / max(ctx.ground_speed_nm_s, 1e-6) + ctx.report_age_s
    for _ in range(5):
        a = th + v / r * tau
        px, py = cx + r * math.sin(a), cy + r * math.cos(a)
        tau = math.hypot(px - ctx.x, py - ctx.y) / ctx.tas_nm_s + ctx.report_age_s
    lead = math.degrees(math.atan2(px - ctx.x - ctx.wind_x_nm_s * tau, py - ctx.y - ctx.wind_y_nm_s * tau))
    d = abs((out.heading_deg - lead) % 360)
    return "accept" if min(d, 360 - d) <= 10 else "reject"
'''


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("dev4401")
    D.build(4401, root, rewrites=1, per_phase=3, log=lambda *a: None)
    return root


def test_development_and_scored_stores_never_mix(tmp_path, built):
    dev = H.CaseStore(built, development=True)
    assert dev.cases(4401) and {c["label"]["rule"] for c in dev.cases(4401)} == {D.RULE}
    with pytest.raises(H.StoreError, match="development store"):
        H.CaseStore(built)                                                   # a scored store cannot open it
    scored = H.CaseStore(tmp_path / "scored", verifier=lambda c: True)
    scored.admit(case())
    with pytest.raises(H.StoreError, match="not made as a development store"):
        H.CaseStore(tmp_path / "scored", development=True)
    one = dev.cases(4401, "stored_right", "heading")[0]
    with pytest.raises(H.StoreError, match="does not belong in a scored store"):
        scored.admit({**one, "stream": 4301})                                # a dev rule in a scored store
    fresh_dev = H.CaseStore(tmp_path / "dev", verifier=lambda c: True, development=True)
    with pytest.raises(H.StoreError, match="not a development stream"):
        fresh_dev.admit(case())                                              # a scored stream in a development store
    other_rule = copy.deepcopy(one)
    other_rule["label"]["rule"] = "B8.heading.action.v2"
    with pytest.raises(H.StoreError, match="does not belong in a development store"):
        fresh_dev.admit(other_rule)


def test_the_development_verifier_re_derives_each_label(built):
    dev = H.CaseStore(built, development=True)
    cases = dev.cases(4401)
    assert all(D.verify(c) for c in cases)
    turned = copy.deepcopy(next(c for c in cases if c["label"]["evidence"]["construction"] == "reference_turned"))
    turned["output"]["heading_deg"] = turned["label"]["evidence"]["reference_heading"]
    assert not D.verify(turned)                                              # a right heading labelled wrong
    right = copy.deepcopy(next(c for c in cases if c["label"]["evidence"]["construction"] == "reference"))
    right["output"]["heading_deg"] = (right["output"]["heading_deg"] + 40) % 360
    assert not D.verify(right)
    trig = copy.deepcopy(next(c for c in cases if c["label"]["evidence"]["construction"] == "trigger_at_order"))
    trig["output"]["report_trigger_nm"] += 1
    assert not D.verify(trig)
    assert not D.verify({**right, "label": {**right["label"], "rule": "dev.other"}})


def test_counts_and_sources(built):
    dev = H.CaseStore(built, development=True)
    for decision in H.DECISIONS:
        assert len(dev.cases(4401, "stored_right", decision)) == H.RIGHT_PER_DECISION
        assert 0 < len(dev.cases(4401, "stored_wrong", decision)) <= D.STORED_WRONG_PER_DECISION
        assert len(dev.cases(4401, "fresh_wrong", decision, 1)) == H.FRESH_PER_REWRITE
    sources = {c["origin"]["episode"][:5] for c in dev.cases(4401)}
    assert "s4401" not in sources                                            # never the stream being flown


def test_a_real_replay_on_the_development_set(tmp_path, built):
    dev = H.CaseStore(built, verifier=D.verify, verifier_name=D.VERIFIER_NAME, development=True)
    lib = Library(tmp_path / "library")
    co = G.Coordinator(lib, tmp_path, None, lambda s: True, lambda a: {}, corpus=dev, stream=4401)
    ok, detail = co.check_replay(LEAD_CHECK, None, "heading", 1)
    assert ok, detail
    ok, detail = co.check_replay("def check(ctx, out):\n    return 'accept'\n", None, "heading", 1)
    assert not ok
    ok, detail = co.check_replay(LEAD_CHECK, None, "heading", 2)             # no fresh cases built for rewrite 2
    assert not ok and "fresh" in detail["refused"]


def test_a_torn_development_marker_is_restored_and_a_scored_log_is_never_adopted(tmp_path):
    """DEV161.MARKER beyond Codex's witness: a torn marker prefix refuses a scored opening; recovery restores the
    mark before any case; a scored log with cases is never adopted as development."""
    full = H._marker_line()
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "cases.log").write_bytes(full[:20])
    with pytest.raises(H.StoreError, match="torn development store marker"):
        H.CaseStore(tmp_path / "a")
    dev = H.CaseStore(tmp_path / "a", verifier=lambda c: True, development=True)
    with pytest.raises(H.StoreError):
        dev.admit(case(stream=4401))                                         # torn: recover first
    dev.recover()
    assert dev._marked() and [r["record"] for r in dev.records] == ["recovery", "store"]
    scored = H.CaseStore(tmp_path / "b", verifier=lambda c: True)
    scored.admit(case())
    with open(scored.log_path, "ab") as f:
        f.write(b"12 abc")
    with pytest.raises(H.StoreError, match="not made as a development store"):
        H.CaseStore(tmp_path / "b", development=True)
