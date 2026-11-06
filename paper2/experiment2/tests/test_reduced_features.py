"""The reduced-feature option (packs/tanker_pack.py, TANKER_REDUCED_PACK; paper2/notes/PLAN-23-OCTOBER.md, "Orientation
Telohelm builds itself"). It removes five calculated fields; the context builder still selects the goal tanker, the
newest report and a bounded history, so this is a step towards raw observation, not raw observation itself.

Procedures, checks and models see the observations without the quantities the context builder derives, and the
learner is told the shorter field list; the runtime's own rules still read the full context; trial flights and the
learner's examples see what live flight sees; with the option off nothing changes.
"""
import json
import math
from dataclasses import replace

import telohelm.learning as G
import paper2.experiment2.mvp_seed as seed
from telohelm.taskpacks.tanker.pack import TANKER_DERIVED; from telohelm.taskpacks.tanker.pack import TANKER_PACK; from telohelm.taskpacks.tanker.pack import TANKER_REDUCED_PACK
from telohelm.runtime.episode_loop import NoModels; from telohelm.runtime.episode_loop import run_episode
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.library import Library
from telohelm.runtime.trusted_log import TrustedLog
import telohelm.taskpacks.tanker.world.episode as E
import telohelm.taskpacks.tanker.world.physics as P

PEEK = ("def decide(ctx):\n"
        "    seen = [f for f in ('wind_x_nm_s', 'report_age_s', 'observed_range_nm') if hasattr(ctx, f)]\n"
        "    return {'heading_deg': 10.0 if seen else 20.0}\n")
ACCEPT = "def check(ctx, out):\n    return 'accept'\n"


def scenario():
    rcv = P.ReceiverState(0.0, 0.0, 0.0, math.radians(90), math.radians(90), P.kt(450.0), P.turn_rate(450.0))
    tankers = (E.Tanker("SHELL 1", P.Orbit(40.0, 10.0, 8.0, P.kt(280.0), math.radians(200))),
               E.Tanker("SHELL 2", P.Orbit(40.0, 60.0, 8.0, P.kt(280.0), 0.0)))
    return E.Scenario(episode_id="reduced-test", receiver=rcv, wind=(0.0, 0.0), tankers=tankers,
                      goals=((0, E.Goal(1, "SHELL 1", 20.0)),))


def fly(tmp_path, pack, name):
    lib = Library(tmp_path / f"lib-{name}", pack=pack.public)
    for kind, source in (("procedure", PEEK), ("check", ACCEPT)):
        sha = lib.add({"kind": kind, "decision": "heading", "conditions": [], "source": source, "origin": {"calls": []}})
        lib.set_status(sha, "adopted", "test")
    (tmp_path / name).mkdir()
    log = TrustedLog(tmp_path / name / "log", tmp_path / name / "heads", "reduced-run")
    run_episode(scenario(), lib, NoModels(), log, Ledger(), max_steps=3, pack=pack)
    return [json.loads(l)["payload"] for l in (tmp_path / name / "log").read_text().splitlines()]


def test_the_reduced_pack_hides_only_derived_fields_and_tells_the_learner_the_shorter_list():
    assert TANKER_DERIVED < TANKER_PACK.public.context_fields
    assert TANKER_REDUCED_PACK.public.context_fields == TANKER_PACK.public.context_fields - TANKER_DERIVED
    assert not TANKER_PACK.hidden and TANKER_REDUCED_PACK.hidden == TANKER_DERIVED
    for field in ("x", "y", "heading_deg", "track_deg", "tas_nm_s", "ground_speed_nm_s", "report_x", "report_y",
                  "report_measured_at", "reports", "tanker_centre_x", "goal_tanker", "t"):
        assert field in TANKER_REDUCED_PACK.public.context_fields


def test_procedures_see_the_derived_fields_only_when_the_pack_shows_them(tmp_path):
    full = [p for p in fly(tmp_path, TANKER_PACK, "full") if p.get("kind") == "decision" and p["decision"] == "heading"]
    reduced = [p for p in fly(tmp_path, TANKER_REDUCED_PACK, "reduced") if p.get("kind") == "decision" and p["decision"] == "heading"]
    assert {p["part"]["heading_deg"] for p in full} == {10.0}
    assert {p["part"]["heading_deg"] for p in reduced} == {20.0}
    assert {p["context_sha256"] for p in full}.isdisjoint({p["context_sha256"] for p in reduced})


def test_the_learner_is_told_the_fields_its_pack_shows(tmp_path):
    lib = Library(tmp_path / "library", pack=TANKER_REDUCED_PACK.public)
    seed.stock(lib)
    co = G.Coordinator(lib, tmp_path, lambda m, meta: "no json", lambda s: True, seed.stand_in_tool,
                       pack=TANKER_REDUCED_PACK.public, flight_pack=TANKER_REDUCED_PACK, procedures_only=True)
    body = json.loads(co.messages([{"episode": "e", "failed": "join"}])[-1][1])
    assert set(body["context_fields"]) == set(TANKER_REDUCED_PACK.public.context_fields)
    assert not TANKER_DERIVED & set(body["context_fields"])


def test_reduced_features_refuse_learned_checks_at_the_entry_point_and_inside_the_learner(tmp_path):
    import pytest
    import paper2.experiment2.run_stream as RS
    with pytest.raises(ValueError, match="procedure-only learning"):
        RS.run_stream(1, 0, tmp_path / "out", learn={"base_url": "http://127.0.0.1:1", "model": "m"},
                      reduced_features=True)                      # no trusted guards: checks would be learnable
    lib = Library(tmp_path / "library", pack=TANKER_REDUCED_PACK.public)
    seed.stock(lib)
    with pytest.raises(ValueError, match="procedure-only"):
        G.Coordinator(lib, tmp_path, lambda m, meta: "no json", lambda s: True, seed.stand_in_tool,
                      pack=TANKER_REDUCED_PACK.public, flight_pack=TANKER_REDUCED_PACK)
    co = G.Coordinator(lib, tmp_path, lambda m, meta: "no json", lambda s: True, seed.stand_in_tool,
                       pack=TANKER_REDUCED_PACK.public, flight_pack=TANKER_REDUCED_PACK, procedures_only=True)
    passed, detail = co.check_replay("def check(ctx, out):\n    return 'accept'\n", None, "heading")
    assert passed is False and "hides fields" in detail["refused"]
    co.no_replay = True
    assert co.check_replay("def check(ctx, out):\n    return 'accept'\n", None, "heading")[0] is False
