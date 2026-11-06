"""TURN59 independent repair/retention controls, using invented finite fixtures only."""
from __future__ import annotations
import hashlib
import json
import math
import pytest
import telohelm.runtime.broker as B; import telohelm.runtime.sandbox as S; import telohelm.runtime.episode_loop as L
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.library import Library
from telohelm.runtime.trusted_log import TrustedLog; from telohelm.runtime.trusted_log import verify
from .test_broker_channel import ARGS; from .test_broker_channel import tool
import telohelm.taskpacks.tanker.world.episode as E; import telohelm.taskpacks.tanker.world.physics as P

@pytest.mark.parametrize("body", [
    b'{"ok":true,"ok":false,"result":1}', b'{"ok":true,"result":{"a":1,"a":2}}',
    b'{"ok":true,"result":1e309}', b'{"ok":true,"result":[[Infinity]]}',
    b'{"ok":false,"error":"x","extra":1}', b'{"ok":1,"result":1}',
    b'{"ok":false}'])
def test_more_invalid_final_envelopes_are_refused(body):
    assert S._final_envelope(body) is None

@pytest.mark.parametrize("body", [b'{"ok":true,"result":null}',
                                  b'{"ok":true,"result":{"a":[true,0,"x"]}}',
                                  b'{"ok":false,"error":"invented"}'])
def test_valid_exact_final_envelopes(body):
    assert S._final_envelope(body) == json.loads(body)

@pytest.fixture
def make(tmp_path):
    n = [0]
    def create(replay=False):
        n[0] += 1
        root = tmp_path / str(n[0]); root.mkdir()
        ledger = Ledger()
        br = B.Broker("invented-boundary", TrustedLog(root / "blog", root / "heads", "turn59"),
                      ledger, tool, replay=replay, condition="invented", stream="invented")
        return br, ledger, root
    return create

def events(root):
    return [json.loads(x)["payload"] for x in (root / "blog").read_text().splitlines()[1:]]

@pytest.mark.skipif(not S.available(), reason="no native sandbox route")
def test_malformed_final_after_prediction_preserves_its_charge(make):
    br, ledger, root = make()
    code = (f"import json,os,sys\ndef decide(ctx):\n    predict(json.loads({ARGS!r}))\n"
            "    sys.stdout.buffer.write(b'F0000000004null'); sys.stdout.flush(); os._exit(0)\n")
    r = S.run_with_broker(code,"decide",{},br,"invented")
    assert r.status == "bad_reply" and br.predictions_used == len(ledger.calls) == 1
    assert ledger.calls[0].status == "answered" and events(root)[-1]["status"] == "bad_reply"
    assert verify(root / "blog", root / "heads")["ok"]

@pytest.mark.skipif(not S.available(), reason="no native sandbox route")
def test_valid_final_then_blocked_exit_hits_same_deadline(make, monkeypatch):
    br, _, root = make()
    monkeypatch.setattr(S,"RUNNER_BROKER", S.RUNNER_BROKER + "\nimport time\ntime.sleep(2)\n")
    r = S.run_with_broker("def decide(ctx):\n    return 1\n","decide",{},br,"invented",wall_seconds=0.2)
    assert r.status == "timeout" and r.wall_seconds < 1.5
    assert events(root)[-1]["status"] == "timeout"


def boundary_scenario():
    # Stationary invented geometry at the threshold: actual assessment saves flagged or unresolved alternatives.
    receiver = P.ReceiverState(0.,0.,0.,math.radians(90),math.radians(90),0.,0.)
    tankers = (E.Tanker("SHELL 1",P.Orbit(0.,-.75,1.,0.,0.)),
               E.Tanker("SHELL 2",P.Orbit(10.,10.,1.,0.,0.)))
    return E.Scenario(episode_id="invented-boundary",receiver=receiver,wind=(0.,0.),tankers=tankers,
                      goals=((0,E.Goal(1,"SHELL 1",20.)),))


def flight(make, monkeypatch, replay=False):
    br, ledger, root = make(replay)
    bundles = []
    original_update = L.ContextBuilder.update
    def capture(self, bundle):
        bundles.append(json.loads(json.dumps(bundle)))
        return original_update(self, bundle)
    # Isolate broker/world ordering from process launch; stub predictor, no models and no substituted world endings.
    def choose(decision,ctx,library,**kw):
        if decision == "heading":
            args = json.loads(ARGS); args["goal_version"] = ctx["goal_version"]
            with br.open_call("invented:decide") as call:
                assert json.loads(call.request(json.dumps({"capability":"predict","args":args})))["ok"]
            part = {"heading_deg":90.}
        else:
            part = {}
        return {"use":"invented", "output":part, "tried":[], "checks":[]}
    with monkeypatch.context() as patch:
        patch.setattr(L.ContextBuilder,"update",capture)
        patch.setattr(L,"choose",choose)
        result = L.run_episode(boundary_scenario(),Library(root / "lib"),L.NoModels(),
            TrustedLog(root / "elog",root / "eheads","turn59"),ledger,max_steps=3,broker=br)
    return result,br,ledger,root,bundles


def test_real_boundary_same_second_endings_and_chain_cuts(make,monkeypatch):
    result,br,ledger,root,_ = flight(make,monkeypatch)
    alternatives = result["ending_cuts"][1:]
    assert [a["ended_at"] for a in alternatives] == [1,2,3]
    assert [a["action_steps"] for a in alternatives] == [1,2,3]
    assert all(a["join_assessment"]["lower"] <= .25 <= a["join_assessment"]["upper"]
               for a in result["outcome"]["alternatives"])
    by_id = {c.request_id:c for c in ledger.calls}
    first = alternatives[0]
    assert [by_id[i].sim_second for i in first["counted"]] == [0]
    assert [by_id[i].sim_second for i in first["excluded"]] == [1,2]
    assert first["straddling"] == first["straddling_unknown"] == []
    prefix = "0"*64; cuts = {}
    for e in events(root):
        body = {k:v for k,v in e.items() if k != "event_prefix_sha256"}
        prefix = hashlib.sha256((prefix + json.dumps(body,sort_keys=True,separators=(",",":"),
                                                 ensure_ascii=False,allow_nan=False)).encode()).hexdigest()
        assert prefix == e["event_prefix_sha256"]
        if e["kind"] == "world_step": cuts[e["action_steps"]] = prefix
    assert all(c["event_prefix_sha256"] == cuts[c["action_steps"]] for c in result["ending_cuts"])
    assert verify(root / "blog",root / "heads")["ok"]

# Fixed in TURN 60 (Claude): BR59.STORE: bundle digest alone cannot reconstruct the received input
def test_received_bundles_are_durably_reconstructible(make,monkeypatch):
    _,_,_,root,bundles = flight(make,monkeypatch)
    checkpoints = [e for e in events(root) if e["kind"] == "bundle"]
    # Inline bytes are one valid repair; content-addressed blobs may replace this assertion with equivalent loading.
    assert [e.get("bundle") for e in checkpoints] == bundles


def test_repeated_live_trace_is_deterministic_and_replay_scope_is_distinct(make,monkeypatch):
    a = flight(make,monkeypatch)
    b = flight(make,monkeypatch)
    replay = flight(make,monkeypatch,replay=True)
    assert a[0]["ending_cuts"] == b[0]["ending_cuts"]
    assert a[0]["actions"] == replay[0]["actions"]
    assert all(c.scope == "replay" for c in replay[2].calls)
    # Existing prefixes include the replay accounting mode, so same-action reruns do not establish mode-independent replay.
    assert a[0]["ending_cuts"][0]["event_prefix_sha256"] != replay[0]["ending_cuts"][0]["event_prefix_sha256"]
