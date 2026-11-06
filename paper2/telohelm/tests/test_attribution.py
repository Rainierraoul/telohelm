"""Share-of-seconds attribution (TURN 77): derivation from the broker's authenticated log of real episodes with a
stand-in chooser, the scripted model provider and a stub prediction tool. Invented scenarios; no model contacted."""
from __future__ import annotations

import json

import pytest

import telohelm.runtime.attribution as A
import telohelm.runtime.broker as B
import telohelm.runtime.episode_loop as L
import telohelm.runtime.models as M
from telohelm.runtime.delivery import DelayedModels
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.library import Library
from telohelm.runtime.trusted_log import TrustedLog; from telohelm.runtime.trusted_log import verify
from .test_broker_channel import tool
from .test_episode_loop import scenario


def fly(tmp_path, monkeypatch, plan, latency=2.4, steps=6, text=None, replay=False, name="e", delayed=True, sc=None):
    """plan(t) gives the stand-in chooser's heading output at second t: a number, or None to climb."""
    root = tmp_path / name
    root.mkdir(parents=True)
    sc = sc or scenario()
    ledger = Ledger()
    br = B.Broker(sc.episode_id, TrustedLog(root / "blog", root / "bheads", "r"), ledger, tool,
                  condition="telohelm", stream="s", replay=replay)
    if delayed:
        reply = text if text is not None else json.dumps({"heading_deg": 123.0})
        gw = M.ModelGateway(sc.episode_id, "telohelm", "s",
                            M.ScriptedProvider(lambda r: M.ModelResponse("answered", reply, 5, 1)),
                            TrustedLog(root / "glog", root / "gheads", "r"), ledger, root / "journal", replay=replay)
        models = DelayedModels(gw, lambda d, r: "small_model", lambda d, c, r: [("user", str(c["t"]))],
                               {"small_model": latency, "reasoning_model": 60.0}, lambda d, t: json.loads(t))
    else:
        models = L.NoModels()

    def choose(decision, ctx, library, **kw):
        if decision != "heading":
            return {"use": "invented", "output": {}, "tried": [], "checks": []}
        h = plan(ctx["t"])
        if h is None:
            return {"climb": "missing_tool", "tried": []}
        return {"use": "invented-procedure", "output": {"heading_deg": h}, "tried": [], "checks": []}
    monkeypatch.setattr(L, "choose", choose)
    r = L.run_episode(sc, Library(root / "lib"), models, TrustedLog(root / "log", root / "heads", "r"), ledger,
                      max_steps=steps, broker=br)
    return r, root, sc.episode_id


def derive(root, ep, **kw):
    return A.derive(root / "blog", root / "bheads", ep, **kw)


def cats(d):
    return [s["category"] for s in d["steps"]]


def test_a_procedure_controlling_every_second(tmp_path, monkeypatch):
    r, root, ep = fly(tmp_path, monkeypatch, lambda t: 90.0, delayed=False)
    d = derive(root, ep)
    assert cats(d) == ["procedure"] * 6 and d["K"] == 6 and sum(d["counts"].values()) == 6
    assert d["shares"]["procedure"] == 1.0 and abs(sum(v for v in d["shares"].values()) - 1.0) < 1e-12


def test_pending_then_delivery_then_holding(tmp_path, monkeypatch):
    r, root, ep = fly(tmp_path, monkeypatch, lambda t: None, latency=2.4, steps=5)
    assert cats(derive(root, ep)) == ["holding_pending"] * 3 + ["small_model", "holding_pending"]


def test_zero_latency_reply_controls_its_own_second(tmp_path, monkeypatch):
    r, root, ep = fly(tmp_path, monkeypatch, lambda t: None, latency=0.0, steps=2)
    d = derive(root, ep)
    assert cats(d) == ["small_model", "small_model"] and d["steps"][0]["request_id"]


def test_a_heading_the_world_refuses_is_holding(tmp_path, monkeypatch):
    r, root, ep = fly(tmp_path, monkeypatch, lambda t: None, latency=0.0, steps=2,
                      text=json.dumps({"heading_deg": 361.0}))
    assert cats(derive(root, ep))[0] in ("holding_pending", "holding_none")


def test_holding_records_who_set_the_command_in_force(tmp_path, monkeypatch):
    r, root, ep = fly(tmp_path, monkeypatch, lambda t: 90.0 if t == 0 else None, delayed=False, steps=3)
    d = derive(root, ep)
    assert cats(d) == ["procedure", "holding_none", "holding_none"]
    assert d["steps"][1]["held_by"] == "procedure"


def test_cuts_empty_and_truncated_episodes(tmp_path, monkeypatch):
    r, root, ep = fly(tmp_path, monkeypatch, lambda t: 90.0, delayed=False, steps=4)
    full = derive(root, ep)
    assert full["complete"] is False and full["scope"] == "truncated"   # stopped by max_steps, not an ending
    cut = derive(root, ep, prefix=2)
    assert cut["K"] == 2 and cut["counts"]["procedure"] == 2 and cut["scope"] == "prefix" and not cut["complete"]
    empty = derive(root, ep, prefix=0)
    assert empty["K"] == 0 and all(v is None for v in empty["shares"].values())
    for bad in ({"prefix": 5}, {"ending": "nominal"}, {"ending": "alternative 0"}, {"ending": "nominal", "prefix": 1}):
        with pytest.raises(A.AttributionError):
            derive(root, ep, **bad)                                    # no such cut, or no recorded ending


def test_a_completed_episode_is_complete(tmp_path, monkeypatch):
    from .test_delivery import joining_scenario
    # The invented joining receiver is stationary with no turn rate, so it is commanded the heading it already holds.
    r, root, ep = fly(tmp_path, monkeypatch, lambda t: 0.0, delayed=False, steps=10, sc=joining_scenario())
    d = derive(root, ep)
    assert d["complete"] is True and d["K"] == r["outcome"]["action_steps"] == 1
    assert d["scope"] == "ending:nominal" and derive(root, ep, ending="nominal") == d


def test_comparator_labels_and_pooled_versus_equal_episode_means(tmp_path, monkeypatch):
    a = derive(*fly(tmp_path, monkeypatch, lambda t: 90.0, delayed=False, steps=4, name="a")[1:])
    b = derive(*fly(tmp_path, monkeypatch, lambda t: 90.0 if t == 0 else None, delayed=False, steps=2, name="b")[1:])
    p = A.pool([a, b])
    assert p["scope"] == "incomplete" and p["K"] == 6 and p["counts"]["procedure"] == 5
    assert abs(p["pooled_shares"]["procedure"] - 5 / 6) < 1e-12
    assert abs(p["equal_episode_mean_shares"]["procedure"] - (1.0 + 0.5) / 2) < 1e-12
    labelled = derive(tmp_path / "a", "loop-test", labels={"procedure": "hand-written controller"})
    assert labelled["counts_by_label"]["hand-written controller"] == 4
    with pytest.raises(A.AttributionError):
        derive(tmp_path / "a", "loop-test", labels={"invented": "x"})


def test_live_and_replay_give_the_same_attribution(tmp_path, monkeypatch):
    live = derive(*fly(tmp_path, monkeypatch, lambda t: None, name="live")[1:])
    rep = derive(*fly(tmp_path, monkeypatch, lambda t: None, name="rep", replay=True)[1:])
    assert cats(live) == cats(rep)


def test_an_unauthenticated_or_incomplete_log_fails(tmp_path, monkeypatch):
    r, root, ep = fly(tmp_path, monkeypatch, lambda t: 90.0, delayed=False, steps=2)
    lines = (root / "blog").read_text().splitlines()
    kept = [l for l in lines if json.loads(l)["payload"].get("kind") != "decision"]
    (root / "blog").write_text("\n".join(kept) + "\n")
    with pytest.raises(A.AttributionError):
        derive(root, ep)


# TURN 79 (Codex's AT78.PROV, AT78.HELD, AT78.POOL and the cut evidence).

def signed(tmp_path, mutate=None, endings=(("nominal", 1),)):
    """A one-second trace written by the trusted writer, with every link a credited small-model second depends on.
    mutate(rows) changes it before writing. Invented values; it is authenticated, so only the derivation's own checks
    can refuse it."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    log, heads = tmp_path / "trace", tmp_path / "theads"
    w = TrustedLog(log, heads, "invented")
    ctx = "b" * 64
    rows = {
        "bundle": {"kind": "bundle", "bundle": {"t": 0}},
        "reply": {"kind": "model_reply", "request_id": "rq", "decision": "heading", "role": "small_model",
                  "goal_version": 1, "issued_second": 0, "issued_step": 0, "available_second": 0,
                  "context_sha256": ctx, "status": "answered", "reply_sha256": "c" * 64, "stale": False,
                  "part": {"heading_deg": 90.}, "refusal": None, "reportable": False, "subphase": "post-issuance"},
        "heading": {"kind": "decision", "decision": "heading", "goal_version": 1, "method": "model:rq",
                    "part": {"heading_deg": 90.}, "source": "small_model", "request_id": "rq", "context_sha256": ctx},
        "report": {"kind": "decision", "decision": "report", "goal_version": 1, "method": "invented-procedure",
                   "part": {}, "source": "procedure", "request_id": None, "context_sha256": ctx},
        "action": {"kind": "action", "action": {"heading_deg": 90., "goal_version": 1},
                   "field_writers": {"heading_deg": "heading"}, "pending": {"heading": False}},
        "world": {"kind": "world_step", "accepted": True, "ended": True, "action_steps": 1},
    }
    if mutate:
        mutate(rows)
    out = [r for r in rows.values() if r is not None]
    out += [{"kind": "ending", "ending": e[0], "end_reason": e[3] if len(e) > 3 else "join",
             "ended_at": e[2] if len(e) > 2 else 1, "action_steps": e[1]} for e in endings]
    for row in out:
        w.append({"episode": "inv", "t": 0, "action_step": 0, **row})
    assert verify(log, heads)["ok"]
    return log, heads


def test_the_intact_signed_trace_derives(tmp_path):
    d = A.derive(*signed(tmp_path), "inv")
    assert cats(d) == ["small_model"] and d["complete"] and d["scope"] == "ending:nominal"
    assert d["steps"][0]["set"]["issued"]["subphase"] == "post-issuance"


def test_codex_signed_trace_derives_when_unchanged(tmp_path):
    from .test_turn78_review_codex import signed_trace
    log, heads = signed_trace(tmp_path, None)
    # Codex's trace ends the world without an ending record, which is itself inconsistent; once the record is added
    # it derives, so each of Codex's changes is what its control refuses.
    with pytest.raises(A.AttributionError, match="nominal ending"):
        A.derive(log, heads, "invented-episode")
    w = TrustedLog(tmp_path / "t2", tmp_path / "h2", "invented")
    for line in log.read_text().splitlines()[1:]:
        w.append(json.loads(line)["payload"])
    w.append({"episode": "invented-episode", "t": 0, "action_step": 0, "kind": "ending", "ending": "nominal",
              "end_reason": "join", "ended_at": 1, "action_steps": 1})
    assert cats(A.derive(tmp_path / "t2", tmp_path / "h2", "invented-episode")) == ["procedure"]


def _set(path, value):
    def mutate(rows):
        target, *keys = rows, *path
        for k in keys[:-1]:
            target = target[k]
        target[keys[-1]] = value
    return mutate


def _sourceless_report_writes(rows):
    rows["report"].update(source="none", part={"report": "x"})


@pytest.mark.parametrize("mutate", [
    _set(("reply", "stale"), True),                                   # an out-of-date reply never controls
    _set(("reply", "refusal"), "invalid_part"),
    _set(("reply", "role"), "reasoning_model"),                       # the credited role is not the delivery's
    _set(("reply", "decision"), "report"),
    _set(("reply", "part"), {"heading_deg": 91.}),                    # the decision wrote what was not proposed
    _set(("reply", "issued_second"), 3),                              # post-issuance from another second
    _set(("reply",), None),                                           # no delivery at all
    _set(("heading", "method"), "model:other"),
    _set(("heading", "source"), "procedure"),                         # a procedure with a model method
    _set(("heading", "goal_version"), 2),                             # decisions under different goal versions
    _set(("heading", "context_sha256"), "not a digest"),
    _set(("heading", "t"), 1),                                        # a decision recorded at another second
    _set(("action", "action", "heading_deg"), 92.),                   # the action differs from the decision's part
    _set(("action", "action", "goal_version"), 7),
    _set(("action", "field_writers"), {"heading_deg": "heading", "report": "report"}),
    _set(("report", "decision"), "heading"),                          # two heading decisions
    _sourceless_report_writes,                                        # a sourceless decision may not write
    _set(("heading", "goal_version"), True),                          # TURN 83 (AT82.GOAL): typed before compared
    _set(("action", "action", "goal_version"), 1.0),
    _set(("world", "action_steps"), 2),
    _set(("world", "ended"), "yes"),
])
def test_further_broken_links_fail(tmp_path, mutate):
    with pytest.raises(A.AttributionError):
        A.derive(*signed(tmp_path, mutate), "inv")


def test_endings_bind_cuts_to_the_protected_log(tmp_path):
    def none_heading(rows):
        rows["reply"] = None
        rows["heading"].update(source="none", method="climb:missing_tool", request_id=None, part=None)
        rows["action"].update(action={}, field_writers={})
        rows["world"]["ended"] = False
    log, heads = signed(tmp_path, none_heading, endings=(("alternative 0", 1),))
    alt = A.derive(log, heads, "inv", ending="alternative 0")
    assert alt["complete"] and alt["K"] == 1 and alt["scope"] == "ending:alternative 0" and cats(alt) == ["holding_none"]
    whole = A.derive(log, heads, "inv")
    assert whole["scope"] == "truncated" and whole["recorded_endings"] == ["alternative 0"]
    # TURN 81 (Codex's AT80.CUT): outside the steps, before any action, recorded twice, at a second that does not
    # follow its last action, or without a reason or time.
    for i, endings in enumerate([(("alternative 0", 2),), (("alternative 0", 0),),
                                 (("alternative 0", 1), ("alternative 0", 1)), (("alternative 0", 1, 5),),
                                 (("alternative 0", 1, 1, ""),), (("alternative 0", 1, None),)]):
        with pytest.raises(A.AttributionError):
            A.derive(*signed(tmp_path / str(i), none_heading, endings=endings), "inv")
    with pytest.raises(A.AttributionError):                           # the world ended but no ending was recorded
        A.derive(*signed(tmp_path / "x", endings=()), "inv")


def test_held_seconds_keep_the_setting_command_identity(tmp_path, monkeypatch):
    r, root, ep = fly(tmp_path, monkeypatch, lambda t: None, latency=0.0, steps=2)
    d = derive(root, ep)
    assert cats(d) == ["small_model", "small_model"]
    r, root, ep = fly(tmp_path, monkeypatch, lambda t: 90.0 if t == 0 else None, delayed=False, steps=3, name="p")
    d = derive(root, ep)
    held = d["steps"][2]["held"]
    assert held["source"] == "procedure" and held["method"] == "invented-procedure" and held["step"] == 0
    assert held["request_id"] is None and len(held["context_sha256"]) == 64 and held == d["steps"][0]["set"]


def test_pool_separates_scopes_unless_mixed_is_asked_for(tmp_path, monkeypatch):
    from .test_delivery import joining_scenario
    done = derive(*fly(tmp_path, monkeypatch, lambda t: 0.0, delayed=False, steps=3, name="c", sc=joining_scenario())[1:])
    cut = derive(*fly(tmp_path, monkeypatch, lambda t: 90.0, delayed=False, steps=2, name="t")[1:])
    with pytest.raises(A.AttributionError):
        A.pool([done, cut])
    assert A.pool([done])["scope"] == "completed"
    m = A.pool([done, cut], mixed=True)
    assert m["scope"] == "mixed" and m["completed"]["K"] == 1 and m["incomplete"]["K"] == 2
    assert m["mixed_completed_and_incomplete"]["K"] == 3


def test_live_and_replay_keep_the_same_held_provenance(tmp_path, monkeypatch):
    live = derive(*fly(tmp_path, monkeypatch, lambda t: None, name="live")[1:])
    rep = derive(*fly(tmp_path, monkeypatch, lambda t: None, name="rep", replay=True)[1:])
    assert live["steps"] == rep["steps"]
