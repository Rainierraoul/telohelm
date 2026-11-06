"""Comparison input/dispatch controls with authored invented scenarios and fake adapters; no real models."""
from dataclasses import replace
import fcntl
import json
from pathlib import Path

import pytest

import paper2.experiment2.comparison as C; import paper2.experiment2.comparators as K; import paper2.experiment2.run_stream as R
import telohelm.taskpacks.tanker.world.episode as E; import telohelm.taskpacks.tanker.world.physics as P; import paper2.experiment2.streams as G


def invented(stream, phase, index):
    s = E.Scenario(G.episode_id(stream, phase, index), P.ReceiverState(0., 0., 0., 0., 0., P.kt(450), P.turn_rate(450)),
                   (0., 0.), (E.Tanker("SHELL 1", P.Orbit(60., 40., 8., P.kt(280), .5)),),
                   ((0, E.Goal(1, "SHELL 1", 20., 10)),))
    # This fixture checks transport only, not help certification.
    return G.Generated(s, [{"fixture": "authored input"}], {"fixture": "not a certificate"} if phase == 5 else None)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "episode", invented)
    root = tmp_path / "parent"
    C.prepare(root, 4401, 6401)
    return root


def profile(tmp_path):
    store = tmp_path / "baseline-store"
    store.mkdir()
    (store / "cases.log").write_text("baseline-only\n")
    guard = tmp_path / "guard.py"
    guard.write_text("def check(_): return True\n")
    return {"local": {"model": "fake-small", "base_url": "fake"},
            "learn": {"model": "fake-reasoner", "base_url": "fake", "max_output_tokens": 131072,
                      "timeout_s": 3600., "effort": "xhigh", "input_cap": 16384},
            "dev_cases": store, "trusted_heading_guard": guard, "trusted_report_guard": guard}


def emit(out, s, complete=True):
    d = out / s.episode_id
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"episode": s.episode_id, "complete": complete}))
    (d / "ledger.jsonl").write_text("")


def test_once_only_generation_and_full_matched_schedule(prepared, monkeypatch):
    monkeypatch.setattr(G, "episode", lambda *a: pytest.fail("input regenerated"))
    body = C.load(prepared)["schedule"]
    assert len(body["rows"]) == 100 and sum(r["adviser"] for r in body["rows"]) == 25
    phase3 = [r["index"] for r in body["rows"] if r["phase"] == 3 and r["adviser"]]
    assert {i // 5 for i in phase3} == {0, 1, 2, 3}
    for r in body["rows"]:
        assert K.scenario_fingerprint(C._scenario(r["scenario"])) == r["scenario_sha256"]
    with pytest.raises(FileExistsError):
        C.prepare(prepared, 4401, 6401)


@pytest.mark.parametrize("change", ["scenario", "receipt", "witness", "membership", "position"])
def test_input_drift_refused_before_dispatch(prepared, change, monkeypatch):
    path = prepared / "schedule.json"
    e = json.loads(path.read_text())
    if change == "scenario":
        e["schedule"]["rows"][0]["scenario"]["receiver"]["x"] += 1
    elif change == "receipt":
        p = prepared / "generation.jsonl"
        p.write_text(p.read_text().replace('"fixture": "authored input"', '"fixture": "changed"', 1))
    elif change == "witness":
        next((prepared / "phase5-witness").glob("*.gz")).write_bytes(b"changed")
    elif change == "membership":
        e["schedule"]["rows"][0]["adviser"] = not e["schedule"]["rows"][0]["adviser"]
    else:
        e["schedule"]["rows"][0]["index"] = 1
    # Rehashing a changed manifest must still fail on its bound source receipt or semantic schedule checks.
    e["sha256"] = C._sha(e["schedule"])
    path.write_text(json.dumps(e))
    monkeypatch.setattr(K, "run_pilot_episode", lambda *a, **kw: pytest.fail("dispatch before verification"))
    with pytest.raises(ValueError):
        C.run_arm(prepared, "calculator")
    assert not (prepared / "calculator-state.json").exists()


def test_all_five_dispatches_share_exact_inputs_and_keep_separate_mutable_stores(prepared, tmp_path, monkeypatch):
    seen = {}
    opts = profile(tmp_path)

    def learning(*args, **kw):
        out = args[2]
        arm = out.name
        seen[arm] = [(s.episode_id, K.scenario_fingerprint(s)) for s in kw["prepared_scenarios"]]
        assert kw["empty_library"] and kw["learn_checks"] and kw["floor_after_cold_start"]
        assert kw["no_replay"] is (arm == "no_replay")
        assert kw["dev_cases"] != opts["dev_cases"]
        assert (kw["dev_cases"] / "cases.log").read_text() == "baseline-only\n"
        (kw["dev_cases"] / "cases.log").write_text("arm-owned appended confirmation\n")
        for s in kw["prepared_scenarios"]:
            emit(out, s)
        (out / "learning-ledger.jsonl").write_text("")

    def policy(s, condition, out, stream, max_steps=None):
        seen.setdefault(condition.name, []).append((s.episode_id, K.scenario_fingerprint(s)))
        emit(out.parent, s)

    def advice(s, out, stream, provider, max_steps=None):
        seen.setdefault("reasoning_alone", []).append((s.episode_id, K.scenario_fingerprint(s)))
        emit(out, s)

    monkeypatch.setattr(R, "run_stream", learning)
    monkeypatch.setattr(K, "run_pilot_episode", policy)
    monkeypatch.setattr(C.adviser, "run_adviser_episode", advice)
    for arm in C.ARMS:
        state = C.run_arm(prepared, arm, learning_options=opts if arm in ("full", "no_replay") else None,
                          adviser_provider=object() if arm == "reasoning_alone" else None)
        assert all(r["status"] == "complete" for r in state["positions"])
    assert seen["full"] == seen["no_replay"] == seen["calculator"] == seen["adaptive_controller"]
    assert len(seen["reasoning_alone"]) == 25
    assert set(seen["reasoning_alone"]) <= set(seen["full"])
    assert (opts["dev_cases"] / "cases.log").read_text() == "baseline-only\n"
    index = json.loads((prepared / "cost-index.json").read_text())
    assert index["parent_allocated_gpu_h"] is None
    assert len(index["arms"]["full"]["ledger_files"]) == 101  #100 flight +1 learning; references only
    with pytest.raises(FileExistsError):
        C.run_arm(prepared, "full", learning_options=opts)


def test_ablation_profile_drift_is_refused(prepared, tmp_path, monkeypatch):
    opts = profile(tmp_path)
    monkeypatch.setattr(R, "run_stream", lambda *a, **kw: None)
    C.run_arm(prepared, "full", learning_options=opts)
    opts["learn"]["timeout_s"] = 7200
    with pytest.raises(ValueError, match="profiles differ"):
        C.run_arm(prepared, "no_replay", learning_options=opts)
    assert not (prepared / "no_replay-state.json").exists()


def test_input_store_drift_is_refused_between_learning_arms(prepared, tmp_path, monkeypatch):
    opts = profile(tmp_path)
    monkeypatch.setattr(R, "run_stream", lambda *a, **kw: None)
    C.run_arm(prepared, "full", learning_options=opts)
    (opts["dev_cases"] / "cases.log").write_text("changed")
    with pytest.raises(ValueError, match="profiles differ"):
        C.run_arm(prepared, "no_replay", learning_options=opts)


def test_failed_adapter_retains_failure_and_does_not_invent_unattempted_receipts(prepared, monkeypatch):
    def fail(s, condition, out, *a, **kw):
        out.mkdir(parents=True)
        (out / "summary.json").write_text(json.dumps({"episode": s.episode_id, "status": "policy_failed", "complete": False}))
        raise K.PolicyFailed("authored witness")
    monkeypatch.setattr(K, "run_pilot_episode", fail)
    with pytest.raises(K.PolicyFailed):
        C.run_arm(prepared, "calculator")
    state = json.loads((prepared / "calculator-state.json").read_text())
    assert state["status"] == "failed" and state["positions"][0]["status"] == "policy_failed"
    assert all(r["status"] == "unresolved" for r in state["positions"][1:])
    with pytest.raises(FileExistsError):
        C.run_arm(prepared, "calculator")


def test_pause_return_leaves_known_unused_slots_and_censored_prefixes_separate(prepared, tmp_path, monkeypatch):
    def pause(*args, **kw):
        emit(args[2], kw["prepared_scenarios"][0], complete=False)
    monkeypatch.setattr(R, "run_stream", pause)
    state = C.run_arm(prepared, "full", learning_options=profile(tmp_path))
    assert state["positions"][0]["status"] == "censored"
    assert all(r["status"] == "unattempted" for r in state["positions"][1:])


def test_serial_slice_refuses_concurrent_dispatch(prepared):
    with (prepared / "dispatch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            C.run_arm(prepared, "calculator")
    assert not (prepared / "calculator-state.json").exists()


@pytest.mark.parametrize("stream,seed", [(4301, 6401), (4401, 6001), (True, 6401)])
def test_scored_or_implicit_inputs_are_not_acquired(tmp_path, stream, seed):
    with pytest.raises(ValueError):
        C.prepare(tmp_path / "no", stream, seed)
    assert not (tmp_path / "no").exists()


def test_cap_evidence_is_kept_and_no_partial_schedule_can_run(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "episode", lambda *a: (_ for _ in ()).throw(G.GenerationError("cap", [{"attempt": 0}])))
    root = tmp_path / "parent"
    with pytest.raises(G.GenerationError):
        C.prepare(root, 4401, 6401)
    assert json.loads((root / "preparation.json").read_text())["status"] == "generation_failed"
    assert json.loads((root / "generation.jsonl").read_text())["attempts"] == [{"attempt": 0}]
    assert not (root / "schedule.json").exists()


def test_prepared_stream_uses_inputs_without_generation(tmp_path, monkeypatch):
    s = invented(4401, 1, 0).scenario
    monkeypatch.setattr(G, "episode", lambda *a: pytest.fail("regeneration"))
    out = tmp_path / "run"
    result = R.run_stream(1, 0, out, stream_seed=4401, empty_library=True, ladder=False, small_interval_s=0,
                          prepared_scenarios=(s,), max_steps=2)
    assert result["summary"]["censored_by_step_limit"] == 1
    receipt = json.loads((out / "prepared-inputs.json").read_text())
    assert receipt == [{"episode": s.episode_id, "scenario_sha256": K.scenario_fingerprint(s)}]
    assert not (out / "generation.jsonl").exists()


def test_invalid_prepared_inputs_fail_before_creating_an_arm(tmp_path):
    s = invented(4401, 1, 0).scenario
    for items in ((), (s, s), ("not a scenario",)):
        out = tmp_path / str(len(items))
        with pytest.raises(ValueError):
            R.run_stream(1, 0, out, stream_seed=4401, prepared_scenarios=items)
        assert not out.exists()


def test_arms_cannot_silently_use_different_episode_limits(prepared, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "run_stream", lambda *a, **kw: None)
    C.run_arm(prepared, "full", learning_options=profile(tmp_path), max_steps=2)
    with pytest.raises(ValueError, match="execution limits differ"):
        C.run_arm(prepared, "calculator", max_steps=3)
    assert not (prepared / "calculator-state.json").exists()


def test_torn_summary_cannot_mask_original_failure_or_count_as_censored(prepared, monkeypatch):
    def fail(s, condition, out, *a, **kw):
        out.mkdir(parents=True)
        (out / "summary.json").write_text('{"episode":')
        raise K.PolicyFailed("original failure")
    monkeypatch.setattr(K, "run_pilot_episode", fail)
    with pytest.raises(K.PolicyFailed, match="original failure"):
        C.run_arm(prepared, "calculator")
    state = json.loads((prepared / "calculator-state.json").read_text())
    assert state["error"]["message"] == "original failure"
    assert state["positions"][0]["status"] == "unresolved"
    assert state["positions"][0]["receipt_error"]
    assert (prepared / "cost-index.json").exists()


def test_source_store_cannot_contain_the_comparison_output(prepared, tmp_path):
    opts = profile(tmp_path)
    opts["dev_cases"] = tmp_path
    with pytest.raises(ValueError, match="output cannot be inside"):
        C.run_arm(prepared, "full", learning_options=opts)
    assert not (prepared / "full-state.json").exists()
