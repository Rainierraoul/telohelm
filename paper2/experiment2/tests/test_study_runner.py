"""The scored-study runner on development fixtures only (unreserved streams 4493 and 4495, C seed 6601, adviser seed
6401) with stand-in models. Covered: the fixed C table; freeze records, which must match the protocol file, the
generator source and the flight profile before anything is drawn or flown; prepared inputs, their receipts and the cost
of preparing them; the meaning of each phase 5 certificate and witness, and the unflown world bound privately; L, C and
F flying the schedule, C and F each position in a fresh process; the condition-wide pause across streams, for L through
the stream runner's hook; the serial dispatch lock; and the shared profile. Nothing here draws a scored scenario or
calls a model: the scored controls replace the generator with a sentinel that fails if called."""
import ast
from dataclasses import asdict
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import types

import pytest

from paper2.experiment2 import comparators as K, outcomecases as O, run_stream as RS, study_runner as S

ROOT = Path(S.__file__).resolve().parents[2]
PROTOCOL = "paper2/protocol/EXPERIMENT-2-PROTOCOL.md"
STREAM, OTHER, C_SEED, ADVISER_SEED = 4493, 4495, 6601, 6401
SCORED = min(S.SCORED_STREAMS)
STAND_IN = {"stand_in": True, "local": None, "learn": None, "small_interval_s": 30, "watch_briefing": True,
            "model_report_control": True, "learn_candidates": 3, "learn_regression": 8}
REAL = {**STAND_IN, "stand_in": False, "local": {"base_url": "http://127.0.0.1:9", "model": "m"},
        "learn": {"base_url": "http://127.0.0.1:9", "model": "m"}}


def _study(tmp_path, *stream_list):
    study = tmp_path / "study"
    S.declare_study(study, list(stream_list))
    return study


def _prepared(tmp_path, phases, *stream_list):
    stream_list = stream_list or (STREAM,)
    study = _study(tmp_path, *stream_list)
    for s in stream_list:
        S.prepare(study / str(s), s, c_seed=C_SEED, adviser_seed=ADVISER_SEED, phases=phases)
    return study


def _good_freeze():
    return {"frozen": True, "protocol_path": PROTOCOL,
            "protocol_sha256": hashlib.sha256((ROOT / PROTOCOL).read_bytes()).hexdigest(),
            "generator_sha256": S.generator_sha256(), "profile_sha256": "0" * 64}


@pytest.fixture
def no_generator(monkeypatch):
    def sentinel(*args, **kwargs):
        raise AssertionError("the generator was called")
    monkeypatch.setattr(S.streams, "episode", sentinel)
    monkeypatch.setattr(S.streams, "draw_help", sentinel)


def _bad(change):
    return lambda: change(_good_freeze())


BAD_FREEZES = {
    "none": lambda: None,
    "empty": lambda: {},
    "old shape with x": lambda: {"frozen": True, "protocol_sha256": "x"},
    "not frozen": _bad(lambda f: {**f, "frozen": False}),
    "empty protocol digest": _bad(lambda f: {**f, "protocol_sha256": ""}),
    "x protocol digest": _bad(lambda f: {**f, "protocol_sha256": "x"}),
    "x generator digest": _bad(lambda f: {**f, "generator_sha256": "x"}),
    "empty profile digest": _bad(lambda f: {**f, "profile_sha256": ""}),
    "wrong protocol digest": _bad(lambda f: {**f, "protocol_sha256": "0" * 64}),
    "wrong generator digest": _bad(lambda f: {**f, "generator_sha256": "0" * 64}),
    "missing protocol file": _bad(lambda f: {**f, "protocol_path": "paper2/protocol/missing.md"}),
    "extra key": _bad(lambda f: {**f, "note": "frozen"}),
    "missing profile digest": _bad(lambda f: {k: v for k, v in f.items() if k != "profile_sha256"}),
}


def test_the_scored_c_table_is_the_protocols_and_reproduces_from_its_seeds():
    text = (ROOT / PROTOCOL).read_text()
    for stream, table in S.SCORED_C_TABLE.items():
        assert S.c_draw(S.SCORED_C_SEEDS[stream]) == table
        row = re.search(rf"^\| {stream} \| (.+) \|$", text, re.M).group(1)
        assert [tuple(int(x) for x in cell.split(", ")) for cell in row.split(" | ")] == [table[p] for p in (1, 2, 3, 4)]


@pytest.mark.parametrize("name", sorted(BAD_FREEZES))
def test_scored_preparation_refuses_an_unverified_freeze_record_and_draws_nothing(tmp_path, no_generator, name):
    study = _study(tmp_path, SCORED)
    with pytest.raises(ValueError):
        S.prepare(study / str(SCORED), SCORED, c_seed=S.SCORED_C_SEEDS[SCORED],
                  adviser_seed=S.SCORED_ADVISER_SEEDS[SCORED], freeze=BAD_FREEZES[name]())
    assert not (study / str(SCORED)).exists()


def test_a_verified_freeze_still_needs_the_fixed_seeds_and_all_phases_and_admission_goes_straight_to_the_generator(
        tmp_path, no_generator):
    good = S.verify_freeze(_good_freeze())
    study = _study(tmp_path, SCORED)
    root = study / str(SCORED)
    c_seed, adviser_seed = S.SCORED_C_SEEDS[SCORED], S.SCORED_ADVISER_SEEDS[SCORED]
    for kwargs in ({"c_seed": c_seed + 1, "adviser_seed": adviser_seed},
                   {"c_seed": c_seed, "adviser_seed": adviser_seed, "phases": (1,)}):
        with pytest.raises(ValueError):
            S.prepare(root, SCORED, freeze=good, **kwargs)
        assert not root.exists()
    # Admitted: the next thing is the generator, here the sentinel. The failed preparation keeps its cost.
    with pytest.raises(AssertionError, match="generator was called"):
        S.prepare(root, SCORED, c_seed=c_seed, adviser_seed=adviser_seed, freeze=good)
    prep = json.loads((root / "preparation.json").read_text())
    assert prep["status"] == "failed" and prep["preparation_wall_s"] >= 0 and prep["preparation_process_s"] >= 0
    assert not (root / "schedule.json").exists()


def test_the_generator_digest_follows_its_source(tmp_path, monkeypatch):
    fake = tmp_path / "fake_world.py"
    fake.write_text("X = 1\n")
    monkeypatch.setattr(S, "GENERATOR", S.GENERATOR + (types.SimpleNamespace(__name__="fake_world",
                                                                             __file__=str(fake)),))
    freeze = S.verify_freeze(_good_freeze())
    fake.write_text("X = 2\n")
    with pytest.raises(ValueError, match="generator source"):
        S.verify_freeze(freeze)


def test_the_generator_digest_covers_every_module_the_generator_imports():
    names = {m.__name__ for m in S.GENERATOR}
    for m in S.GENERATOR:
        for node in ast.walk(ast.parse(Path(m.__file__).read_text())):
            if isinstance(node, ast.ImportFrom):
                base = (importlib.util.resolve_name("." * node.level + (node.module or ""), m.__package__)
                        if node.level else node.module)
                targets = [base] if node.module else [f"{base}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            else:
                continue
            for t in targets:
                if t.startswith(("telohelm", "paper2")):
                    assert t in names, (m.__name__, t)


RETIRED, REPLACEMENT = min(O.EXPOSED_SOURCES), min(O.EXPOSED_SOURCES.values())     # read, never literal (identity guard)


@pytest.mark.parametrize("stream, c_seed, adviser_seed", [(RETIRED, 6601, 6401), (REPLACEMENT, 6601, 6401),
                                                          (4302, 6601, 6401), (STREAM, 6201, 6401), (STREAM, 6601, 6001)])
def test_development_preparation_refuses_reserved_identities_and_scored_seeds(tmp_path, stream, c_seed, adviser_seed):
    study = tmp_path / "dev"
    with pytest.raises(ValueError):
        S.prepare(study / str(stream), stream, c_seed=c_seed, adviser_seed=adviser_seed)
    assert not study.exists()


def test_preparation_needs_a_declared_study_and_the_streams_own_folder(tmp_path):
    seeds = {"c_seed": C_SEED, "adviser_seed": ADVISER_SEED, "phases": (1,)}
    with pytest.raises(ValueError, match="declared study"):
        S.prepare(tmp_path / str(STREAM), STREAM, **seeds)
    study = _study(tmp_path, STREAM)
    with pytest.raises(ValueError, match="named by the stream"):
        S.prepare(study / str(OTHER), STREAM, **seeds)
    with pytest.raises(ValueError, match="not declared"):
        S.prepare(study / str(OTHER), OTHER, **seeds)
    with pytest.raises(ValueError, match="scored or development"):
        S.declare_study(tmp_path / "mixed", [SCORED, STREAM])
    assert not (study / str(OTHER)).exists() and not (tmp_path / "mixed").exists()


def test_prepared_inputs_bind_their_receipts_and_record_their_cost(tmp_path):
    root = _prepared(tmp_path, (1,)) / str(STREAM)
    env = S.load(root)
    body = env["schedule"]
    assert body["layout"] == "v2" and sum(r["c"] for r in body["rows"]) == 5
    prep = json.loads((root / "preparation.json").read_text())
    assert prep["status"] == "ready" and prep["schedule_sha256"] == env["sha256"]
    assert [p["episode"] for p in prep["positions"]] == [r["episode"] for r in body["rows"]]
    assert prep["preparation_wall_s"] > 0 and prep["preparation_process_s"] > 0
    schedule = json.loads((root / "schedule.json").read_text())
    schedule["schedule"]["rows"][0]["c"] = not schedule["schedule"]["rows"][0]["c"]
    (root / "schedule.json").write_text(json.dumps(schedule))
    with pytest.raises(ValueError):
        S.load(root)


def test_a_failed_preparation_keeps_its_cost_and_its_error(tmp_path, monkeypatch):
    study = _study(tmp_path, STREAM)
    real, calls = S.streams.episode, []

    def second_fails(*args):
        calls.append(args)
        if len(calls) == 2:
            raise RuntimeError("stopped for the test")
        return real(*args)
    monkeypatch.setattr(S.streams, "episode", second_fails)
    with pytest.raises(RuntimeError):
        S.prepare(study / str(STREAM), STREAM, c_seed=C_SEED, adviser_seed=ADVISER_SEED, phases=(1,))
    prep = json.loads((study / str(STREAM) / "preparation.json").read_text())
    assert prep["status"] == "failed" and "stopped for the test" in prep["error"] and len(prep["positions"]) == 20
    assert prep["preparation_wall_s"] > 0 and prep["preparation_process_s"] > 0
    assert not (study / str(STREAM) / "schedule.json").exists()


# -- phase 5: the certificate, the witness and the unflown world -------------------------------------------------------
@pytest.fixture(scope="module")
def phase5(tmp_path_factory):
    study = tmp_path_factory.mktemp("p5") / "study"
    S.declare_study(study, [STREAM])
    S.prepare(study / str(STREAM), STREAM, c_seed=C_SEED, adviser_seed=ADVISER_SEED, phases=(5,))
    return study


def _copy(study, tmp_path):
    shutil.copytree(study, tmp_path / "study")
    return tmp_path / "study" / str(STREAM)


def _receipts(root):
    return [json.loads(line) for line in (root / "generation.jsonl").read_text().splitlines()]


def _rechain(root, k, change):
    """Change receipt k and re-hash everything above it, so only the receipt's meaning can refuse it."""
    lines = (root / "generation.jsonl").read_text().splitlines()
    rec = json.loads(lines[k])
    change(rec)
    lines[k] = json.dumps(rec, sort_keys=True)
    (root / "generation.jsonl").write_text("\n".join(lines) + "\n")
    env = json.loads((root / "schedule.json").read_text())
    env["schedule"]["rows"][k]["generation_sha256"] = S._sha(rec)
    env["sha256"] = S._sha(env["schedule"])
    (root / "schedule.json").write_text(json.dumps(env))


def _rewitness(root, k, change):
    path = root / _receipts(root)[k]["witness"]["file"]
    data = gzip.compress(json.dumps(change(json.loads(gzip.decompress(path.read_bytes()))), sort_keys=True,
                                    default=str).encode(), mtime=0)
    path.write_bytes(data)
    _rechain(root, k, lambda rec: rec["witness"].update(sha256=hashlib.sha256(data).hexdigest()))


def test_phase_5_inputs_keep_their_certificates_witnesses_and_unflown_worlds(phase5, tmp_path):
    root = phase5 / str(STREAM)
    receipts = _receipts(root)
    assert len(receipts) == 20 and all("witness" in r and "alternative_world" in r for r in receipts)
    S.load(root)
    first = receipts[0]
    accepted = first["attempts"][-1]
    d, flown = S.streams.draw_help(STREAM, 0, accepted["attempt"])
    assert flown == accepted["flown"] and first["alternative_world"]["world"] != flown
    kept = json.loads((root / first["alternative_world"]["file"]).read_text())
    unflown = S._final(d.world_b if flown == "a" else d.world_a, accepted["assessment"]["deadline"])
    assert kept == json.loads(json.dumps(asdict(unflown), sort_keys=True))
    assert kept["goals"][0][1]["deadline"] == accepted["assessment"]["deadline"]
    copy = _copy(phase5, tmp_path)
    witness = copy / first["witness"]["file"]
    witness.write_bytes(witness.read_bytes() + b" ")
    with pytest.raises(ValueError, match="witness digest"):
        S.load(copy)


def test_preparation_refuses_an_empty_certificate_before_publishing(phase5, tmp_path, monkeypatch):
    raw = S.load(phase5 / str(STREAM))["schedule"]["rows"][0]["scenario"]
    monkeypatch.setattr(S.streams, "episode", lambda stream, phase, index, layout: S.streams.Generated(
        S._scenario(raw), attempts=[], witness={}, layout=layout))
    root = _study(tmp_path, STREAM) / str(STREAM)
    with pytest.raises(ValueError, match="certified"):
        S.prepare(root, STREAM, c_seed=C_SEED, adviser_seed=ADVISER_SEED, phases=(5,))
    assert not (root / "schedule.json").exists()
    assert json.loads((root / "preparation.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("change, match", [
    (lambda w: {}, "certified"),
    (lambda w: {**w, "label": "not certified"}, "certified"),
    (lambda w: {**w, "deadline": w["deadline"] + 1}, "assessment"),
], ids=["empty witness", "wrong label", "changed deadline"])
def test_a_rehashed_witness_without_its_certified_meaning_is_refused(phase5, tmp_path, change, match):
    root = _copy(phase5, tmp_path)
    _rewitness(root, 3, change)
    with pytest.raises(ValueError, match=match):
        S.load(root)


@pytest.mark.parametrize("change, match", [
    (lambda r: r.pop("alternative_world"), "unflown world is not bound"),
    (lambda r: r["alternative_world"].update(world=r["attempts"][-1]["flown"]), "unflown world is not bound"),
    (lambda r: r["attempts"][-1].update(help_certified=False), "certified"),
], ids=["no unflown world", "unflown world is the flown one", "accepted attempt not certified"])
def test_a_rehashed_receipt_without_its_certified_meaning_is_refused(phase5, tmp_path, change, match):
    root = _copy(phase5, tmp_path)
    _rechain(root, 5, change)
    with pytest.raises(ValueError, match=match):
        S.load(root)


def _reschedule(root, k, change):
    """Change schedule row k's scenario and re-hash its fingerprint and the schedule."""
    env = json.loads((root / "schedule.json").read_text())
    row = env["schedule"]["rows"][k]
    change(row["scenario"])
    row["scenario_sha256"] = K.scenario_fingerprint(S._scenario(row["scenario"]))
    env["sha256"] = S._sha(env["schedule"])
    (root / "schedule.json").write_text(json.dumps(env))


def _realternative(root, k, change):
    """Rewrite receipt k's private unflown world and re-hash the receipt chain."""
    path = root / _receipts(root)[k]["alternative_world"]["file"]
    data = json.dumps(change(json.loads(path.read_text())), sort_keys=True).encode()
    path.write_bytes(data)
    _rechain(root, k, lambda rec: rec["alternative_world"].update(sha256=hashlib.sha256(data).hexdigest()))


def _move_receiver(scenario):
    scenario["receiver"]["x"] += 100


def _drop_deadline(world):
    world["goals"][0][1]["deadline"] = None
    return world


def _swap_world(rec):
    flown = rec["attempts"][-1]["flown"]
    rec["attempts"][-1]["flown"] = rec["alternative_world"]["world"]
    rec["alternative_world"]["world"] = flown


@pytest.mark.parametrize("change, match", [
    (lambda root: _rechain(root, 2, lambda r: r["attempts"][-1].update(seed=r["attempts"][-1]["seed"] + 1)), "seed"),
    (lambda root: _rechain(root, 2, _swap_world), "selected world"),
    (lambda root: _reschedule(root, 2, _move_receiver), "flown scenario is not the certified one"),
    (lambda root: _realternative(root, 2, lambda w: {}), "not a scenario"),
    (lambda root: _realternative(root, 2, _drop_deadline), "unflown world is not the certified one"),
    (lambda root: _realternative(root, 2, lambda w: {**w, "episode_id": w["episode_id"] + "x"}),
     "unflown world is not the certified one"),
    (lambda root: _rewitness(root, 2, lambda w: {k: v for k, v in w.items() if k != "binding"}), "bind"),
    (lambda root: _rewitness(root, 2, lambda w: {**w, "binding": {**w["binding"], "seed": w["seed"] + 1}}), "seed"),
], ids=["accepted seed", "selected world", "moved receiver", "empty unflown world", "unflown world without deadline",
        "unflown world of another episode", "certificate without binding", "binding with another seed"])
def test_retained_phase_5_records_that_contradict_each_other_are_refused(phase5, tmp_path, change, match):
    root = _copy(phase5, tmp_path)
    change(root)
    with pytest.raises(ValueError, match=match):
        S.load(root)


def test_a_changed_unflown_world_or_scenario_deadline_is_refused(phase5, tmp_path):
    root = _copy(phase5, tmp_path / "a")
    alt = root / _receipts(root)[7]["alternative_world"]["file"]
    alt.write_text(alt.read_text().replace("SHELL", "SHELX", 1))
    with pytest.raises(ValueError, match="unflown world's record differs"):
        S.load(root)
    root = _copy(phase5, tmp_path / "b")
    env = json.loads((root / "schedule.json").read_text())
    row = env["schedule"]["rows"][9]
    row["scenario"]["goals"][0][1]["deadline"] += 1
    row["scenario_sha256"] = K.scenario_fingerprint(S._scenario(row["scenario"]))
    env["sha256"] = S._sha(env["schedule"])
    (root / "schedule.json").write_text(json.dumps(env))
    with pytest.raises(ValueError, match="deadline"):
        S.load(root)


# -- flying ------------------------------------------------------------------------------------------------------------
def test_l_c_and_f_fly_the_schedule_with_c_and_f_each_position_in_a_fresh_process(tmp_path):
    root = _prepared(tmp_path, (1, 2, 3)) / str(STREAM)
    with pytest.raises(ValueError, match="snapshot"):
        S.run_arm(root, "F", profile=STAND_IN, max_steps=10)
    (root / "F-state.json").unlink()
    states = {arm: S.run_arm(root, arm, profile=STAND_IN, max_steps=10) for arm in ("L", "C", "F")}
    assert all(s["status"] == "returned" and s["stop"] is None for s in states.values())
    assert [len(states[a]["positions"]) for a in ("L", "C", "F")] == [60, 15, 10]
    for state in states.values():
        assert all(p["input_matches_schedule"] is True and p["status"] == "censored" for p in state["positions"])
    processes = [json.loads((root / "arms" / a / p["episode"] / "process.json").read_text())
                 for a in ("C", "F") for p in states[a]["positions"]]
    assert all(p["process_exit"] == 0 for a in ("C", "F") for p in states[a]["positions"])
    pids = [p["pid"] for p in processes]
    assert len(set(pids)) == 25 and os.getpid() not in pids and all(p["ppid"] == os.getpid() for p in processes)
    snap = json.loads((root / "inputs/F-snapshot/snapshot.json").read_text())
    assert snap["after_episode"] == f"s{STREAM}-p2-e19"
    first_f = states["F"]["positions"][0]["episode"]
    copied = json.loads((root / "arms/F" / first_f / "flight/library-from.json").read_text())
    assert copied["files"] == snap["files"]
    assert all(p["copy_matches_snapshot"] is True for p in states["F"]["positions"])
    source = json.loads((root / "arms/F" / first_f / "f-source.json").read_text())
    assert source["files"] == snap["files"] and source["after_episode"] == f"s{STREAM}-p2-e19"
    for a in ("C", "F"):
        for p in states[a]["positions"]:
            worker = json.loads((root / "arms" / a / f"{p['episode']}.worker.json").read_text())
            process = json.loads((root / "arms" / a / p["episode"] / "process.json").read_text())
            assert worker["dispatcher_pid"] == os.getpid() == process["ppid"]
            assert worker["pid"] == worker["pgid"] == process["pid"] == process["pgid"]
            assert re.fullmatch(r"[0-9a-f]{32}", worker["token"])
    with pytest.raises(ValueError, match="running dispatch"):
        S._fly_one(root, "C", states["C"]["positions"][0]["episode"], 10)



def test_c_full_flies_cs_positions_with_only_the_small_model_ration_lifted(tmp_path):
    """Protocol amendment 7: C-full is C with no interval between small-model calls and an allowance that cannot run
    out; every other flight setting is the shared profile's."""
    root = _prepared(tmp_path, (1,)) / str(STREAM)
    states = {arm: S.run_arm(root, arm, profile=STAND_IN, max_steps=10) for arm in ("C", "C-full")}
    assert all(s["status"] == "returned" and s["stop"] is None for s in states.values())
    episodes = {arm: [p["episode"] for p in states[arm]["positions"]] for arm in states}
    assert episodes["C"] == episodes["C-full"] and len(episodes["C"]) == 5
    assert all(p["process_exit"] == 0 and p["input_matches_schedule"] is True for p in states["C-full"]["positions"])
    for arm, interval, limit in (("C", 30, None), ("C-full", 0, S.HORIZON_S)):
        for ep in episodes[arm]:
            flight = json.loads((root / "arms" / arm / ep / "flight" / "stream-summary.json").read_text())["flight"]
            assert flight["small_interval_s"] == interval and flight.get("small_call_limit") == limit
            assert flight["model_report_control"] is True and flight["watch_briefing"] == {"tolerance": 1.0}
    calls = {arm: [json.loads((root / "arms" / arm / ep / "flight" / "stream-summary.json").read_text())["model_calls"]
                   for ep in episodes[arm]] for arm in episodes}
    assert all(full > rationed for full, rationed in zip(calls["C-full"], calls["C"])), calls
    assert S.HORIZON_S == 3600 and "C-full" in S.PAUSED and S._rows_for("C-full", [{"c": True}, {"c": False}]) == [{"c": True}]

def _bad_call(episode, rid):
    return {"kind": "reasoning_model", "scope": "running", "status": "timed_out", "episode": episode,
            "request_id": rid, "recovers": None}


def test_the_condition_is_judged_across_streams_and_a_stop_reaches_every_stream(tmp_path):
    study = _prepared(tmp_path, (1,), STREAM, OTHER)
    a, b = study / str(STREAM), study / str(OTHER)
    c_a = [r["episode"] for r in S.load(a)["schedule"]["rows"] if r["c"]]
    c_b = [r["episode"] for r in S.load(b)["schedule"]["rows"] if r["c"]]
    assert S.pause_anchor(study, "C") == [x for pair in zip(c_a, c_b) for x in pair]
    good = [{**_bad_call(c_b[i % 5], f"b{i}"), "status": "answered"} for i in range(9)]
    assert S.pause_authority(study, "C", OTHER, good) is None
    # One bad call in stream a would pause a stream judged alone; over the condition it is 1 in 10, not over 10%.
    assert S.pause_authority(study, "C", STREAM, [_bad_call(c_a[0], "a1")]) is None
    # A call outside the anchor does not count.
    outside = next(r["episode"] for r in S.load(a)["schedule"]["rows"] if not r["c"])
    assert S.pause_authority(study, "C", STREAM, [_bad_call(c_a[0], "a1"), _bad_call(outside, "a2")]) is None
    stop = S.pause_authority(study, "C", STREAM, [_bad_call(c_a[0], "a1"), _bad_call(c_a[1], "a3")])
    assert stop["due"] and stop["started"] == 11 and stop["bad"] == 2 and stop["detected_in_stream"] == STREAM
    assert S.stop_record(study, "C") == stop and S.stop_record(study, "L") is None
    state = S.run_arm(b, "C", profile=STAND_IN, max_steps=5)
    assert state["stop"] == stop and all(p["status"] == "paused" for p in state["positions"])
    assert not list((b / "arms" / "C").rglob("process.json"))


def test_adviser_records_first_counterfactual_pause_without_stopping_any_stream(tmp_path):
    study = _prepared(tmp_path, (1,), STREAM, OTHER)
    anchor = S.pause_anchor(study, "adviser")
    first = _bad_call(anchor[0], "adviser-timeout")
    assert S.pause_authority(study, "adviser", STREAM, [first]) is None
    observed = S.counterfactual_stop_record(study, "adviser")
    assert observed["due"] and observed["started"] == observed["bad"] == 1
    assert observed["enforced"] is False and observed["detected_in_stream"] == STREAM
    assert S.stop_record(study, "adviser") is None
    # A later stream keeps the first record and never creates an effective stop.
    assert S.pause_authority(study, "adviser", OTHER, [_bad_call(anchor[1], "second-timeout")]) is None
    assert S.counterfactual_stop_record(study, "adviser") == observed
    assert not (study / "stops/adviser.json").exists()


def test_adviser_continues_through_all_prepared_positions_after_counterfactual_pause(tmp_path, monkeypatch):
    study = _prepared(tmp_path, (1,))
    root = study / str(STREAM)
    visited = []
    def position(scenario, out, label, provider, max_steps=None):
        visited.append(scenario.episode_id)
    monkeypatch.setattr(S.adviser, "run_adviser_episode", position)
    monkeypatch.setattr(S, "_stream_rows_for", lambda root, arm, rows: [_bad_call(visited[0], "timeout")])
    body = S.load(root)["schedule"]
    rows = S._rows_for("adviser", body["rows"])
    S._fly(root, study, "adviser", body, rows, None, object(), None, S._Owned())
    assert visited == [row["episode"] for row in rows]
    assert S.counterfactual_stop_record(study, "adviser")["due"]
    assert S.stop_record(study, "adviser") is None


def test_l_stops_through_the_stream_runners_hook(tmp_path):
    study = _prepared(tmp_path, (1,))
    root = study / str(STREAM)
    rows = S.load(root)["schedule"]["rows"][:4]
    episodes = [r["episode"] for r in rows]
    calls = []

    def judge(seen, pause_rows):
        calls.append(list(seen))
        return {"stopped": "for the test"} if len(seen) == 2 else None
    scenarios = tuple(S._scenario(r["scenario"]) for r in rows)
    RS.run_stream(20, 1, tmp_path / "direct", stream="hook", stream_seed=STREAM, layout="v2",
                  prepared_scenarios=scenarios, max_steps=5, pause_authority=judge)
    assert calls == [[], episodes[:1], episodes[:2]]
    paused = json.loads((tmp_path / "direct/paused.json").read_text())
    assert paused["after"] == episodes[1] and paused["authority"] == {"stopped": "for the test"}
    assert not (tmp_path / "direct" / episodes[2]).exists()
    with pytest.raises(ValueError, match="no_pause"):
        RS.run_stream(20, 1, tmp_path / "both", stream="hook", stream_seed=STREAM, layout="v2",
                      prepared_scenarios=scenarios, max_steps=5, no_pause=True, pause_authority=judge)
    stop = S.pause_authority(study, "L", STREAM, [_bad_call(episodes[0], "l1")])
    state = S.run_arm(root, "L", profile=STAND_IN, max_steps=5)
    assert state["stop"] == stop and all(p["status"] == "paused" for p in state["positions"])
    assert json.loads((root / "arms/L/paused.json").read_text())["after"] is None


def test_a_serial_dispatch_holds_the_lock_throughout_and_refuses_a_second(tmp_path, monkeypatch):
    root = _prepared(tmp_path, (1,)) / str(STREAM)
    with S._parent_lock(root, wait=False):
        with pytest.raises(BlockingIOError):
            S.run_arm(root, "calculator", max_steps=5)
    assert not (root / "calculator-state.json").exists()
    seen = []

    def probe(*args, **kwargs):
        try:
            with S._parent_lock(root, wait=False):
                seen.append("free")
        except BlockingIOError:
            seen.append("held")
    monkeypatch.setattr(S, "_fly", probe)
    S.run_arm(root, "calculator", max_steps=5)
    S.run_arm(root, "adaptive_controller", max_steps=5, concurrent=True)
    assert seen == ["held", "free"]


def test_scored_flights_need_the_frozen_profile_real_models_and_no_step_limit():
    record = S._profile_record(REAL)
    freeze = {**_good_freeze(), "profile_sha256": S.profile_digest(record)}
    body = {"mode": "scored", "freeze": freeze}
    S._admit(body, record, None)
    for rec, steps, match in [(record, 5, "step limit"),
                              (S._profile_record({**REAL, "small_interval_s": 20}), None, "frozen one"),
                              (S._profile_record(STAND_IN), None, "real profile")]:
        with pytest.raises(ValueError, match=match):
            S._admit(body, rec, steps)
    with pytest.raises(ValueError):
        S._admit({"mode": "scored", "freeze": {**freeze, "protocol_sha256": "x"}}, record, None)
    S._admit({"mode": "development", "freeze": None}, S._profile_record(STAND_IN), 5)


def test_telohelm_conditions_share_one_complete_flight_profile(tmp_path):
    root = _prepared(tmp_path, (1,)) / str(STREAM)
    with pytest.raises(ValueError, match="fixed booleans"):
        S.run_arm(root, "C", profile={**STAND_IN, "watch_briefing": None})
    S.run_arm(root, "C", profile=STAND_IN, max_steps=5)
    with pytest.raises(ValueError, match="share one flight profile"):
        S.run_arm(root, "L", profile={**STAND_IN, "small_interval_s": 20}, max_steps=5)


# -- the lasting stop record -------------------------------------------------------------------------------------------
def _due_rows(study, arm="C"):
    return [_bad_call(S.pause_anchor(study, arm)[0], "s1")]


def test_readers_see_no_stop_record_or_the_whole_one_while_it_is_published(tmp_path, monkeypatch):
    study = _prepared(tmp_path, (1,))
    seen, errors, done = [], [], threading.Event()

    def reader():
        while not done.is_set():
            try:
                seen.append(S.stop_record(study, "C"))
            except Exception as exc:          # a partly written record would show up here
                errors.append(repr(exc))
    real_link = os.link

    def slow_link(src, dst):
        time.sleep(0.3)                       # the written record waits under its temporary name
        return real_link(src, dst)
    monkeypatch.setattr(S.os, "link", slow_link)
    readers = [threading.Thread(target=reader) for _ in range(4)]
    for t in readers:
        t.start()
    stop = S.pause_authority(study, "C", STREAM, _due_rows(study))
    time.sleep(0.1)
    done.set()
    for t in readers:
        t.join()
    assert not errors and None in seen and stop in seen
    assert all(x is None or x == stop for x in seen)
    assert sorted(p.name for p in (study / "stops").iterdir()) == ["C.json"]


def test_an_interrupted_publication_leaves_no_stop_record_and_a_later_one_succeeds(tmp_path, monkeypatch):
    study = _prepared(tmp_path, (1,))

    def interrupted(src, dst):
        raise KeyboardInterrupt
    monkeypatch.setattr(S.os, "link", interrupted)
    with pytest.raises(KeyboardInterrupt):
        S.pause_authority(study, "C", STREAM, _due_rows(study))
    assert S.stop_record(study, "C") is None and not (study / "stops" / "C.json").exists()
    monkeypatch.undo()
    stop = S.pause_authority(study, "C", STREAM, _due_rows(study))
    assert stop["due"] and S.stop_record(study, "C") == stop


def test_the_stop_record_is_published_once(tmp_path):
    path = tmp_path / "C.json"
    assert S._publish_once(path, {"first": True}) is True
    assert S._publish_once(path, {"second": True}) is False
    assert json.loads(path.read_text()) == {"first": True} and [p.name for p in tmp_path.iterdir()] == ["C.json"]


def test_a_study_folder_that_cannot_make_hard_links_is_refused(tmp_path, monkeypatch):
    def no_links(src, dst):
        raise OSError("hard links not supported")
    monkeypatch.setattr(S.os, "link", no_links)
    with pytest.raises(ValueError, match="hard links"):
        S.declare_study(tmp_path / "study", [STREAM])
    assert not (tmp_path / "study").exists()


# -- stopping the dispatcher -------------------------------------------------------------------------------------------
SLEEPER = """
import json, os, signal, subprocess, sys, time
if sys.argv[2] == "ignore":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
with open(sys.argv[1] + ".tmp", "w") as f:
    json.dump({"pid": os.getpid(), "grandchild": grandchild.pid}, f)
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
time.sleep(120)
"""


def _gone(pid, within=5.0):
    end = time.monotonic() + within
    while time.monotonic() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def _sleeping_positions(monkeypatch, tmp_path, mode="plain"):
    marks = []

    def argv(root, arm, episode, max_steps, payload=None):
        marks.append(tmp_path / f"{episode}.pids.json")
        return [sys.executable, "-c", SLEEPER, str(marks[-1]), mode]
    monkeypatch.setattr(S, "_position_argv", argv)
    return marks


def _signal_when(path_list, sig=signal.SIGTERM):
    def wait_then_signal():
        end = time.monotonic() + 30
        while time.monotonic() < end and not (path_list and path_list[0].exists()):
            time.sleep(0.05)
        os.kill(os.getpid(), sig)
    t = threading.Thread(target=wait_then_signal, daemon=True)
    t.start()
    return t


@pytest.fixture
def server():
    """A process outside the dispatcher's worker group, standing in for a shared model server."""
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    yield p
    p.kill()
    p.wait()


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT], ids=["SIGTERM", "SIGINT"])
def test_a_stopped_dispatcher_closes_and_reaps_its_position_worker(tmp_path, monkeypatch, server, sig):
    root = _prepared(tmp_path, (1,)) / str(STREAM)
    marks = _sleeping_positions(monkeypatch, tmp_path)
    before = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
    _signal_when(marks, sig)
    with pytest.raises(S.DispatchStopped):
        S.run_arm(root, "C", profile=STAND_IN, max_steps=5)
    pids = json.loads(marks[0].read_text())
    assert _gone(pids["pid"]) and _gone(pids["grandchild"]) and server.poll() is None
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)) == before
    first = marks[0].name.split(".")[0]
    receipt = json.loads((root / "arms/C" / f"{first}.exit.json").read_text())
    assert receipt["interrupted"] is True and receipt["exit"] == -signal.SIGTERM and receipt["signals"] == [sig]
    state = json.loads((root / "C-state.json").read_text())
    assert state["status"] == "stopped" and state["error"]["type"] == "DispatchStopped"
    assert [p["status"] for p in state["positions"]] == ["interrupted"] + ["unattempted"] * 4
    assert len(marks) == 1


def test_a_worker_that_ignores_sigterm_is_killed_after_the_grace_period(tmp_path, monkeypatch, server):
    root = _prepared(tmp_path, (1,)) / str(STREAM)
    marks = _sleeping_positions(monkeypatch, tmp_path, mode="ignore")
    monkeypatch.setattr(S, "GRACE_S", 0.5)
    _signal_when(marks)
    with pytest.raises(S.DispatchStopped):
        S.run_arm(root, "C", profile=STAND_IN, max_steps=5)
    pids = json.loads(marks[0].read_text())
    assert _gone(pids["pid"]) and _gone(pids["grandchild"]) and server.poll() is None
    receipt = json.loads((root / "arms/C" / f"{marks[0].name.split('.')[0]}.exit.json").read_text())
    assert receipt["interrupted"] is True and receipt["exit"] == -signal.SIGKILL


def test_a_stop_that_arrives_while_the_worker_is_being_created_still_reaches_it(tmp_path, monkeypatch, server):
    root = _prepared(tmp_path, (1,)) / str(STREAM)
    _sleeping_positions(monkeypatch, tmp_path)
    created, real = [], subprocess.Popen

    def popen_then_stop(*args, **kwargs):
        p = real(*args, **kwargs)
        created.append(p.pid)
        os.kill(os.getpid(), signal.SIGTERM)  # handled before the dispatcher has registered the worker
        return p
    monkeypatch.setattr(S.subprocess, "Popen", popen_then_stop)
    with pytest.raises(S.DispatchStopped):
        S.run_arm(root, "C", profile=STAND_IN, max_steps=5)
    assert len(created) == 1 and _gone(created[0]) and server.poll() is None
    state = json.loads((root / "C-state.json").read_text())
    assert state["status"] == "stopped" and state["positions"][0]["status"] == "interrupted"


def test_a_short_write_is_finished_before_the_stop_record_is_published(tmp_path, monkeypatch):
    real, calls = os.write, []

    def half(fd, data):
        calls.append(len(data))
        return real(fd, bytes(data[:max(1, len(data) // 2)]))   # a legal short write
    monkeypatch.setattr(S.os, "write", half)
    value = {"arm": "C", "note": "x" * 80}
    assert S._publish_once(tmp_path / "C.json", value) is True
    assert len(calls) > 1 and json.loads((tmp_path / "C.json").read_text()) == value
    assert [p.name for p in tmp_path.iterdir()] == ["C.json"]


@pytest.mark.parametrize("write", [lambda fd, data: 0, lambda fd, data: (_ for _ in ()).throw(OSError(28, "full"))],
                         ids=["nothing written", "write fails"])
def test_a_failed_write_publishes_no_stop_record(tmp_path, monkeypatch, write):
    monkeypatch.setattr(S.os, "write", write)
    with pytest.raises(OSError):
        S._publish_once(tmp_path / "C.json", {"arm": "C"})
    assert not list(tmp_path.iterdir())


LEADER = """
import json, os, subprocess, sys
grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
with open(sys.argv[1], "w") as f:
    json.dump({"pid": os.getpid(), "grandchild": grandchild.pid}, f)
"""


def test_a_worker_that_exits_normally_still_has_its_group_closed(tmp_path, server):
    owned, mark = S._Owned(), tmp_path / "pids.json"
    S._run_position(owned, tmp_path, "e", [sys.executable, "-c", LEADER, str(mark)])
    pids = json.loads(mark.read_text())
    receipt = json.loads((tmp_path / "e.exit.json").read_text())
    assert receipt["exit"] == 0 and receipt["group_left_running"] is True and "interrupted" not in receipt
    assert owned.child is None and _gone(pids["grandchild"]) and server.poll() is None


def test_an_error_after_the_worker_exits_still_closes_its_group(tmp_path, monkeypatch, server):
    class FailsAfterExit(subprocess.Popen):
        def wait(self, timeout=None):
            super().wait(timeout=30)
            raise RuntimeError("for the test")
    monkeypatch.setattr(S.subprocess, "Popen", FailsAfterExit)
    owned, mark = S._Owned(), tmp_path / "pids.json"
    with pytest.raises(RuntimeError, match="for the test"):
        S._run_position(owned, tmp_path, "e", [sys.executable, "-c", LEADER, str(mark)])
    pids = json.loads(mark.read_text())
    receipt = json.loads((tmp_path / "e.exit.json").read_text())
    assert receipt["interrupted"] is True and receipt["error"] == "RuntimeError" and receipt["exit"] == 0
    assert receipt["group_left_running"] is True
    assert owned.child is None and _gone(pids["grandchild"]) and server.poll() is None


def test_a_second_stop_during_close_out_still_leaves_a_complete_state(tmp_path, monkeypatch):
    root = _prepared(tmp_path, (1,)) / str(STREAM)
    monkeypatch.setattr(S, "_position_argv", lambda root, arm, episode, max_steps, payload=None: [sys.executable, "-c", "pass"])
    real, handlers = S.receipts, []

    def receipts_then_stop(*args, **kwargs):
        handlers.append(signal.getsignal(signal.SIGTERM))
        os.kill(os.getpid(), signal.SIGTERM)  # a termination while the final receipts are written
        return real(*args, **kwargs)
    monkeypatch.setattr(S, "receipts", receipts_then_stop)
    state = S.run_arm(root, "C", profile=STAND_IN, max_steps=5)
    assert isinstance(getattr(handlers[0], "__self__", None), S._Owned)
    assert state["status"] == "returned" and state["signals"] == [signal.SIGTERM]
    assert json.loads((root / "C-state.json").read_text())["signals"] == [signal.SIGTERM]
    assert all(p["process_exit"] == 0 for p in state["positions"])


def _snapshot(root, episode):
    library = root.parent / "library-source"
    library.mkdir(parents=True, exist_ok=True)
    (library / "events.log").write_text("one\n")
    (library / "entry.json").write_text("{}")
    RS._snapshot_library(library, root / "inputs" / "F-snapshot", episode)
    return root / "inputs" / "F-snapshot"


@pytest.mark.parametrize("change, match", [
    (None, None),
    (lambda f: f.joinpath("snapshot.json").write_text(f.joinpath("snapshot.json").read_text().replace("p2-e19", "p2-e18")),
     "p2-e19"),
    (lambda f: f.joinpath("library/entry.json").write_text('{"changed": true}'), "manifest"),
    (lambda f: f.joinpath("library/extra.json").write_text("{}"), "manifest"),
    (lambda f: f.joinpath("snapshot.json").write_text('{"after_episode": "s4493-p2-e19", "files": {'), "snapshot"),
], ids=["valid", "wrong boundary", "changed library", "extra file", "damaged marker"])
def test_f_uses_only_this_streams_p2_e19_snapshot_with_its_whole_manifest(tmp_path, change, match):
    root = tmp_path / str(STREAM)
    folder = _snapshot(root, S.streams.episode_id(STREAM, 2, 19))
    if change is None:
        marker = S.verify_snapshot(root, STREAM)
        assert marker["after_episode"] == f"s{STREAM}-p2-e19" and set(marker["files"]) == {"events.log", "entry.json"}
        assert marker["snapshot_sha256"] == hashlib.sha256((folder / "snapshot.json").read_bytes()).hexdigest()
        return
    change(folder)
    with pytest.raises(ValueError, match=match):
        S.verify_snapshot(root, STREAM)


def test_another_streams_snapshot_is_refused(tmp_path):
    root = tmp_path / str(STREAM)
    _snapshot(root, S.streams.episode_id(OTHER, 2, 19))
    with pytest.raises(ValueError, match="p2-e19"):
        S.verify_snapshot(root, STREAM)


def test_an_f_copy_is_joined_to_the_snapshot_it_recorded(tmp_path):
    root = tmp_path / str(STREAM)
    folder = root / "arms" / "F" / "e1"
    (folder / "flight").mkdir(parents=True)
    (folder / "f-source.json").write_text(json.dumps({"files": {"events.log": "a"}}))
    (folder / "flight" / "library-from.json").write_text(json.dumps({"files": {"events.log": "a"}}))
    assert S._f_copy_matches(root, "e1") is True
    (folder / "flight" / "library-from.json").write_text(json.dumps({"files": {"events.log": "b"}}))
    assert S._f_copy_matches(root, "e1") is False
    (folder / "f-source.json").unlink()
    assert S._f_copy_matches(root, "e1") is False



def test_the_frozen_profile_digest_leaves_out_only_the_server_addresses():
    template = {**REAL, "local": {**REAL["local"], "base_url": "{small_model}"},
                "learn": {**REAL["learn"], "base_url": "{reasoning_model}"}}
    frozen = S.profile_digest(S._profile_record(template, executable=False))
    for ports in ((32010, 32011), (39998, 39999)):
        filled = {**REAL, "local": {**REAL["local"], "base_url": f"http://127.0.0.1:{ports[0]}"},
                  "learn": {**REAL["learn"], "base_url": f"http://127.0.0.1:{ports[1]}"}}
        assert S.profile_digest(S._profile_record(filled)) == frozen
    for changed in ({**REAL, "learn": {**REAL["learn"], "model": "other/model"}},
                    {**REAL, "local": {**REAL["local"], "model": "other/model"}},
                    {**REAL, "small_interval_s": 20},
                    {**REAL, "watch_briefing": False}):
        assert S.profile_digest(S._profile_record(changed)) != frozen


def test_scored_admission_accepts_any_job_ports_for_the_frozen_profile_and_nothing_else():
    template = {**REAL, "local": {**REAL["local"], "base_url": "{small_model}"},
                "learn": {**REAL["learn"], "base_url": "{reasoning_model}"}}
    body = {"mode": "scored", "freeze": {**_good_freeze(),
                                         "profile_sha256": S.profile_digest(S._profile_record(template, executable=False))}}
    filled = {**REAL, "local": {**REAL["local"], "base_url": "http://127.0.0.1:32010"},
              "learn": {**REAL["learn"], "base_url": "http://127.0.0.1:32011"}}
    S._admit(body, S._profile_record(filled), None)
    with pytest.raises(ValueError, match="frozen one"):
        S._admit(body, S._profile_record({**filled, "learn": {**filled["learn"], "model": "other/model"}}), None)



@pytest.mark.parametrize("url", ["http://10.0.0.5:8000", "http://127.0.0.1:8000/v1", "http://localhost:8000",
                                 "https://127.0.0.1:8000", "http://127.0.0.1", "{reasoning_model}", None],
                         ids=["remote", "with a path", "a name", "https", "no port", "placeholder", "missing"])
def test_an_executable_profile_needs_real_loopback_origins(url):
    with pytest.raises(ValueError, match="loopback origin"):
        S._profile_record({**REAL, "learn": {**REAL["learn"], "base_url": url}})
    S._profile_record(REAL)                                   # http, a numeric loopback address and a port


def test_only_a_freeze_template_may_name_servers_by_role():
    template = {**REAL, "local": {**REAL["local"], "base_url": "{small_model}"},
                "learn": {**REAL["learn"], "base_url": "{reasoning_model}"}}
    S._profile_record(template, executable=False)
    with pytest.raises(ValueError, match="loopback origin"):
        S._profile_record(template)
    with pytest.raises(ValueError, match="loopback origin"):
        S._profile_record({**template, "learn": {**template["learn"], "base_url": "http://10.0.0.5:8000"}},
                          executable=False)
