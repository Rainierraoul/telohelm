"""The broker inside run_episode (TURN 56): action steps from the world's own record, event prefixes kept per step,
ending cuts before, at and after a step, straddling lists, links and episode-end feedback. An invented scenario and
library, a stub prediction tool and the model layer stubbed; a few seconds only."""
from __future__ import annotations

import json

import pytest

import telohelm.runtime.broker as B
import telohelm.runtime.sandbox as S
from telohelm.runtime.episode_loop import NoModels; from telohelm.runtime.episode_loop import run_episode
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.library import Library
from telohelm.runtime.trusted_log import TrustedLog; from telohelm.runtime.trusted_log import verify
from .test_episode_loop import HEADING_CHECK; from .test_episode_loop import REPORT; from .test_episode_loop import REPORT_CHECK; from .test_episode_loop import scenario
import telohelm.taskpacks.tanker.world.episode as E

pytestmark = pytest.mark.skipif(not S.available(), reason="no tested sandbox route on this platform")

PREDICTING = '''def decide(ctx):
    r = predict({"goal_version": ctx.goal_version,
                 "start": {"x_nm": ctx.x, "y_nm": ctx.y, "heading_deg": ctx.heading_deg, "tas_nm_s": ctx.tas_nm_s},
                 "headings_deg": [ctx.heading_deg] * 5, "wind_nm_s": [ctx.wind_x_nm_s, ctx.wind_y_nm_s],
                 "orbit": {"cx_nm": ctx.tanker_centre_x, "cy_nm": ctx.tanker_centre_y, "radius_nm": ctx.tanker_radius_nm,
                           "speed_nm_s": ctx.tanker_speed_nm_s, "angle_rad": 0.0}})
    return {"heading_deg": 90.0 if r["ok"] else 0.0}
'''


def tool(args):
    n = len(args["headings_deg"])
    return {"positions_nm": [[0.0, 0.0]] * n, "closest_approach_nm": 30.0, "closest_at_s": float(n)}


def library(tmp_path):
    lib = Library(tmp_path / "library")
    for kind, decision, source in (("procedure", "heading", PREDICTING), ("check", "heading", HEADING_CHECK),
                                   ("procedure", "report", REPORT), ("check", "report", REPORT_CHECK)):
        sha = lib.add({"kind": kind, "decision": decision, "conditions": [], "source": source, "origin": {"calls": []}})
        lib.set_status(sha, "adopted", "invented")
    return lib


def episode(tmp_path, steps=6, monkeypatch=None, alternative_at=None, replay=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    sc = scenario()
    log = TrustedLog(tmp_path / "log", tmp_path / "heads", "invented-run")
    blog = TrustedLog(tmp_path / "blog", tmp_path / "bheads", "invented-run")
    ledger = Ledger()
    br = B.Broker(sc.episode_id, blog, ledger, tool, condition="telohelm", stream="invented-stream", replay=replay)
    if alternative_at is not None:
        real = E.Episode.outcome

        def with_alternative(self):
            out = real(self)
            out["alternatives"] = [dict(out, end_reason="join", ended_at=alternative_at,
                                        action_steps=alternative_at, alternatives=[])]
            return out
        monkeypatch.setattr(E.Episode, "outcome", with_alternative)
    r = run_episode(sc, library(tmp_path), NoModels(), log, ledger, max_steps=steps, broker=br)
    return r, br, ledger, tmp_path / "blog"


def test_action_steps_are_the_worlds_and_links_are_recorded(tmp_path):
    r, br, ledger, blog = episode(tmp_path)
    steps = [c.action_step for c in ledger.calls]
    assert steps == list(range(6)) and [c.sim_second for c in ledger.calls] == list(range(6))
    assert all(c.condition == "telohelm" and c.stream == "invented-stream" and c.answered_step == c.action_step
               for c in ledger.calls)
    assert r["predictions_used"] == 6 and len(r["actions"]) == 6
    assert verify(blog, tmp_path / "bheads")["ok"]


def test_ending_cuts_before_at_and_after_one_step_with_prefixes(tmp_path, monkeypatch):
    r, br, ledger, _ = episode(tmp_path, monkeypatch=monkeypatch, alternative_at=3)
    nominal, alternative = r["ending_cuts"]
    assert nominal["action_steps"] == 6 and len(nominal["counted"]) == 6
    assert alternative["action_steps"] == 3
    made = {c.request_id: c.action_step for c in ledger.calls}
    assert sorted(made[i] for i in alternative["counted"]) == [0, 1, 2]       # before the ending
    assert sorted(made[i] for i in alternative["excluded"]) == [3, 4, 5]      # at and after it
    assert alternative["straddling"] == [] and alternative["straddling_unknown"] == []
    assert alternative["event_prefix_sha256"] and alternative["event_prefix_sha256"] != nominal["event_prefix_sha256"]


def test_episode_end_feedback_and_closed_calls(tmp_path):
    r, br, _, blog = episode(tmp_path)
    (fb,) = br.take_feedback()
    assert fb["kind"] == "episode_end" and fb["goal_met"] is False and fb["failed_part"] in ("join", "both")
    ev = [json.loads(l)["payload"] for l in blog.read_text().splitlines()[1:]]
    opened = {e["call"] for e in ev if e["kind"] == "call_open"}
    closed = {e["call"] for e in ev if e["kind"] == "call_close"}
    assert opened == closed and ev[-1]["kind"] == "feedback"
    with pytest.raises(B.BrokerError):
        br.open_call("heading")


def test_a_broker_for_another_episode_is_refused(tmp_path):
    sc = scenario()
    br = B.Broker("other", TrustedLog(tmp_path / "b", tmp_path / "bh", "r"), Ledger(), tool)
    with pytest.raises(ValueError):
        run_episode(sc, library(tmp_path), NoModels(), TrustedLog(tmp_path / "l", tmp_path / "h", "r"), Ledger(),
                    max_steps=2, broker=br)


def test_native_run_closes_every_call_and_accounts_every_prediction(tmp_path):
    r, br, ledger, blog = episode(tmp_path)
    ev = [json.loads(l)["payload"] for l in blog.read_text().splitlines()[1:]]
    closes = [e for e in ev if e["kind"] == "call_close"]
    assert closes and all(e["status"] == "completed" for e in closes)
    starts = [e["request_id"] for e in ev if e["kind"] == "prediction_start"]
    ends = [e["request_id"] for e in ev if e["kind"] == "prediction_end"]
    assert starts == ends == [c.request_id for c in ledger.calls] and len(starts) == r["predictions_used"]
    kinds = [e["kind"] for e in ev]
    assert kinds.count("world_step") == len(r["actions"]) and kinds.count("bundle") == len(r["actions"])


def test_same_second_cut_excludes_calls_made_after_the_ending_in_event_order(tmp_path, monkeypatch):
    """A join found over [t-1, t] ends the episode at t before the actions of second t, so a call made at second t
    is excluded even though it has the same simulated second as the ending (B10, D9)."""
    r, br, ledger, _ = episode(tmp_path, monkeypatch=monkeypatch, alternative_at=4)
    alternative = r["ending_cuts"][1]
    by_id = {c.request_id: c for c in ledger.calls}
    same_second = [i for i, c in by_id.items() if c.sim_second == alternative["ended_at"]]
    assert same_second and all(i in alternative["excluded"] for i in same_second)
    assert all(by_id[i].sim_second < alternative["ended_at"] for i in alternative["counted"])


def test_repeated_live_runs_give_the_same_event_prefixes(tmp_path):
    """Two live runs of the same episode are deterministic. This is not a replay test (corrected in TURN 60)."""
    a = episode(tmp_path / "a")
    b = episode(tmp_path / "b")
    assert [c["event_prefix_sha256"] for c in a[0]["ending_cuts"]] == [c["event_prefix_sha256"] for c in b[0]["ending_cuts"]]
    assert [c.action_step for c in a[2].calls] == [c.action_step for c in b[2].calls]



def test_saved_inputs_are_rebuilt_from_the_log_and_checked(tmp_path):
    r, br, _, blog = episode(tmp_path)
    inputs = B.load_inputs(blog, tmp_path / "bheads", "loop-test")
    assert inputs["briefing"]["own_callsign"] and [b["action_step"] for b in inputs["bundles"]] == list(range(6))
    lines = blog.read_text().splitlines()
    for i, line in enumerate(lines):
        rec = json.loads(line)
        if rec["payload"].get("kind") == "bundle":
            rec["payload"]["bundle"]["t"] = 999                # edit the content but not its recorded digest
            lines[i] = json.dumps(rec)
            break
    edited = tmp_path / "edited"
    edited.write_text("\n".join(lines) + "\n")
    with pytest.raises(B.BrokerError):
        B.load_inputs(edited, tmp_path / "bheads", "loop-test")


def test_replay_mode_keeps_the_logical_prefix_and_changes_only_accounting(tmp_path):
    live = episode(tmp_path / "live")
    replay = episode(tmp_path / "replay", replay=True)
    assert live[0]["actions"] == replay[0]["actions"]
    assert [c["logical_prefix_sha256"] for c in live[0]["ending_cuts"]] == \
           [c["logical_prefix_sha256"] for c in replay[0]["ending_cuts"]]
    assert live[0]["ending_cuts"][0]["event_prefix_sha256"] != replay[0]["ending_cuts"][0]["event_prefix_sha256"]
    assert all(c.scope == "replay" for c in replay[2].calls)


def test_decisions_replayed_from_saved_inputs_match_the_logged_decisions(tmp_path):
    """Logical replay from the log alone: rebuild the contexts from the saved briefing and bundles, run the same
    library through a replay-mode broker, and compare each heading decision with the logged one."""
    from telohelm.runtime.chooser import choose
    from telohelm.runtime.context import ContextBuilder
    r, br, _, blog = episode(tmp_path / "live")
    logged = [json.loads(l)["payload"] for l in blog.read_text().splitlines()[1:]]
    logged = [e["part"] for e in logged if e["kind"] == "decision" and e["decision"] == "heading"]
    inputs = B.load_inputs(blog, tmp_path / "live" / "bheads", "loop-test")
    rb = B.Broker("loop-test", TrustedLog(tmp_path / "rlog", tmp_path / "rheads", "replay"), Ledger(), tool,
                  replay=True, condition="telohelm", stream="invented-stream")
    builder = ContextBuilder(inputs["briefing"])
    lib = library(tmp_path / "replay")
    replayed = []
    for item in inputs["bundles"]:
        ctx = builder.update(item["bundle"])
        rb.at(t=item["t"], action_step=item["action_step"], goal_version=ctx["goal_version"])
        replayed.append(choose("heading", ctx, lib, broker=rb).get("output"))
    assert replayed == logged and len(replayed) == 6
