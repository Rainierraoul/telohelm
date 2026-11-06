"""End-to-end test of Telohelm's loop with stored procedures only and the model layer stubbed."""
from __future__ import annotations

import math

import pytest

import telohelm.runtime.sandbox as S
from telohelm.runtime.context import ContextBuilder
from telohelm.runtime.episode_loop import NoModels; from telohelm.runtime.episode_loop import run_episode
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.library import Library
from telohelm.runtime.trusted_log import TrustedLog; from telohelm.runtime.trusted_log import verify
import telohelm.taskpacks.tanker.world.episode as E
import telohelm.taskpacks.tanker.world.physics as P

AIM = '''import math
def decide(ctx):
    # Lead pursuit on the briefed orbit from the newest report, with wind allowed for.
    a = math.atan2(ctx.report_x - ctx.tanker_centre_x, ctx.report_y - ctx.tanker_centre_y)
    tau = 0.0
    for _ in range(25):
        b = a + ctx.tanker_speed_nm_s * (ctx.t - ctx.report_measured_at + tau) / ctx.tanker_radius_nm
        px = ctx.tanker_centre_x + ctx.tanker_radius_nm * math.sin(b) - ctx.wind_x_nm_s * tau
        py = ctx.tanker_centre_y + ctx.tanker_radius_nm * math.cos(b) - ctx.wind_y_nm_s * tau
        tau = math.hypot(px - ctx.x, py - ctx.y) / ctx.tas_nm_s
    return {"heading_deg": math.degrees(math.atan2(px - ctx.x, py - ctx.y)) % 360}
'''
HEADING_CHECK = "def check(ctx, out):\n    return 'accept' if ctx.report_age_s <= 10 and 0 <= out.heading_deg < 360 else 'reject'\n"
REPORT = '''def decide(ctx):
    rate = ctx.observed_range_rate_nm_s
    if rate is not None and rate < 0 and abs(ctx.observed_range_nm - ctx.report_at_nm) < 0.15:
        return {"report": "Wedgetail, aircraft 2, %g NM from %s" % (ctx.report_at_nm, ctx.goal_tanker)}
    return {}
'''
REPORT_CHECK = "def check(ctx, out):\n    return 'accept' if ctx.report_age_s <= 2 else 'reject'\n"


def scenario(wind=(0.0, 0.0)):
    rcv = P.ReceiverState(0.0, 0.0, 0.0, math.radians(90), math.radians(90), P.kt(450.0), P.turn_rate(450.0))
    tankers = (E.Tanker("SHELL 1", P.Orbit(40.0, 10.0, 8.0, P.kt(280.0), math.radians(200))),
               E.Tanker("SHELL 2", P.Orbit(40.0, 60.0, 8.0, P.kt(280.0), 0.0)))
    return E.Scenario(episode_id="loop-test", receiver=rcv, wind=wind, tankers=tankers,
                      goals=((0, E.Goal(1, "SHELL 1", 20.0)),))


def stocked(tmp_path) -> Library:
    lib = Library(tmp_path / "library")
    for kind, decision, source in (("procedure", "heading", AIM), ("check", "heading", HEADING_CHECK),
                                   ("procedure", "report", REPORT), ("check", "report", REPORT_CHECK)):
        sha = lib.add({"kind": kind, "decision": decision, "conditions": [], "source": source,
                       "origin": {"calls": []}})
        lib.set_status(sha, "adopted", "test")
    return lib


def test_context_has_every_b7_field():
    from telohelm.runtime.procedures import CONTEXT_FIELDS
    sc = scenario()
    ep = E.Episode(sc)
    start = ep.reset()
    ctx = ContextBuilder(start["briefing"]).update(start["bundle"])
    assert set(ctx) == CONTEXT_FIELDS
    assert ctx["report_age_s"] == 0 and ctx["observed_range_rate_nm_s"] is None


@pytest.mark.skipif(not S.available(), reason="macOS sandbox route only")
def test_an_episode_flown_by_stored_procedures_alone(tmp_path):
    (tmp_path / "protected").mkdir()
    log = TrustedLog(tmp_path / "runner.log", tmp_path / "protected" / "heads.jsonl", "loop-run")
    ledger = Ledger()
    result = run_episode(scenario(), stocked(tmp_path), NoModels(), log, ledger)
    out = result["outcome"]
    assert out["goal_met"] and out["settled"] and out["end_reason"] == "join"
    assert ledger.calls == []                                     # no model was needed
    assert verify(tmp_path / "runner.log", tmp_path / "protected" / "heads.jsonl")["ok"]


@pytest.mark.skipif(not S.available(), reason="macOS sandbox route only")
def test_an_empty_library_climbs_and_holds(tmp_path):
    (tmp_path / "protected").mkdir()
    log = TrustedLog(tmp_path / "runner.log", tmp_path / "protected" / "heads.jsonl", "loop-run")
    result = run_episode(scenario(), Library(tmp_path / "empty"), NoModels(), log, Ledger(), max_steps=5)
    assert result["climbs"] == 10 and not result["outcome"]["goal_met"]
