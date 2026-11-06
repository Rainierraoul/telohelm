"""Development comparison runner, first slice: one prepared input schedule, isolated arm dispatch.

No servers/jobs are started here. Plantings, amortisation and scored collection remain separate work.
The parent owns private scenarios/witnesses. Policies receive their usual public interfaces only.

Shared serving (Claude, TURN 202): with concurrent=True, or `run --concurrent`, each arm runs in its own process. Its
setup (checks, the claim, the shared profile files) and its close-out (state and cost index) run under the parent
lock, and its flights run outside it, so several arms can share one pair of model servers. The job limits requests
in flight per server with runtime/slots.py. Without concurrent the dispatch stays serial and refuses a second arm.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

from . import adviser as adviser; from . import comparators as comparators; from . import run_stream as run_stream
import telohelm.taskpacks.tanker.world.episode as E; import telohelm.taskpacks.tanker.world.physics as P; from . import streams as streams

ARMS = ("full", "no_replay", "reasoning_alone", "calculator", "adaptive_controller")


def _bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _write(path: Path, value: Any) -> None:
    # Atomic replacement of owned metadata; refuses to reuse an arm directory elsewhere. The temporary name carries the
    # process, so two arm processes writing the same file never share one.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(_bytes(value) + b"\n")
    tmp.replace(path)


def _scenario(raw: dict) -> E.Scenario:
    return E.Scenario(raw["episode_id"], P.ReceiverState(**raw["receiver"]), tuple(raw["wind"]),
                      tuple(E.Tanker(t["callsign"], P.Orbit(**t["orbit"])) for t in raw["tankers"]),
                      tuple((s, E.Goal(**g)) for s, g in raw["goals"]), raw["control_callsign"],
                      raw["delay_s"], tuple(tuple(d) for d in raw["dropouts"]))


def prepare(root: Path, stream: int, matching_seed: int, phases: tuple[int, ...] = (1, 2, 3, 4, 5)) -> dict:
    """Generate each development input once, keeping all generation attempts before any arm can run.

    The per-phase count is20, including for shorter phase subsets; matched positions use the existing D22 sampler.
    Matching seeds6400..6499 are explicitly development-only, separate from frozen scored matching seeds6001..6004.
    """
    if type(stream) is not int or not 4400 <= stream <= 4499:
        raise ValueError("development streams4400..4499 only")
    if type(matching_seed) is not int or not 6400 <= matching_seed <= 6499:
        raise ValueError("use a development matching seed6400..6499")
    if not phases or any(type(p) is not int or p not in range(1, 6) for p in phases) or tuple(sorted(set(phases))) != phases:
        raise ValueError("phases must be unique and increasing, in1..5")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    positions = [{"phase": p, "index": i, "episode": streams.episode_id(stream, p, i)} for p in phases for i in range(20)]
    _write(root / "preparation.json", {"status": "preparing", "positions": positions})
    draws = adviser.matched_indices(matching_seed)
    rows = []
    with (root / "generation.jsonl").open("x") as log:
        for pos in positions:
            try:
                g = streams.episode(stream, pos["phase"], pos["index"])
            except streams.GenerationError as exc:
                log.write(json.dumps(streams.cap_record(stream, pos["phase"], pos["index"], exc)) + "\n")
                log.flush()
                _write(root / "preparation.json", {"status": "generation_failed", "episode": pos["episode"],
                                                    "error": str(exc), "positions": positions})
                raise
            if g.scenario.episode_id != pos["episode"]:
                raise ValueError("generator returned a different episode identity")
            rec = streams.generation_record(g)
            if g.witness is not None:
                rec["witness"] = run_stream._write_witness(root, g.scenario.episode_id, g.witness)
            if pos["phase"] == 5 and g.witness is None:
                raise ValueError("phase5 input has no retained witness")
            log.write(json.dumps(rec, sort_keys=True) + "\n")
            log.flush()
            rows.append({**pos, "scenario": asdict(g.scenario), "scenario_sha256": comparators.scenario_fingerprint(g.scenario),
                         "generation_sha256": _sha(rec),
                         "adviser": pos["index"] in draws[str(pos["phase"])]["chronological_indices"]})
    body = {"version": 1, "development": True, "stream": stream, "matching_seed": matching_seed,
            "phases": list(phases), "arm_order": list(ARMS), "matched_draws": draws, "rows": rows}
    envelope = {"sha256": _sha(body), "schedule": body}
    _write(root / "schedule.json", envelope)
    _write(root / "preparation.json", {"status": "ready", "schedule_sha256": envelope["sha256"]})
    return envelope


def load(root: Path) -> dict:
    """Check schedule, every scenario and generation receipt, and all retained witness bytes. No regeneration."""
    root = Path(root)
    envelope = json.loads((root / "schedule.json").read_text())
    body = envelope["schedule"]
    if envelope["sha256"] != _sha(body):
        raise ValueError("schedule digest mismatch")
    if body["version"] != 1 or body["development"] is not True or body["arm_order"] != list(ARMS):
        raise ValueError("unsupported comparison schedule")
    if type(body["stream"]) is not int or not 4400 <= body["stream"] <= 4499:
        raise ValueError("development streams only")
    if type(body["matching_seed"]) is not int or not 6400 <= body["matching_seed"] <= 6499:
        raise ValueError("development matching seeds only")
    phases = body["phases"]
    if not phases or any(type(p) is not int or p not in range(1, 6) for p in phases) or sorted(set(phases)) != phases:
        raise ValueError("invalid phase schedule")
    expected = [(p, i, streams.episode_id(body["stream"], p, i)) for p in phases for i in range(20)]
    if [(r["phase"], r["index"], r["episode"]) for r in body["rows"]] != expected:
        raise ValueError("schedule positions mismatch")
    draws = adviser.matched_indices(body["matching_seed"])
    if body["matched_draws"] != draws:
        raise ValueError("matched draw mismatch")
    generation = [json.loads(line) for line in (root / "generation.jsonl").read_text().splitlines()]
    if len(generation) != len(body["rows"]):
        raise ValueError("generation receipt count mismatch")
    for row, rec in zip(body["rows"], generation):
        if row["generation_sha256"] != _sha(rec) or rec["episode"] != row["episode"]:
            raise ValueError("generation receipt mismatch")
        s = _scenario(row["scenario"])
        if s.episode_id != row["episode"] or comparators.scenario_fingerprint(s) != row["scenario_sha256"]:
            raise ValueError("scenario digest mismatch")
        if row["adviser"] is not (row["index"] in draws[str(row["phase"])]["chronological_indices"]):
            raise ValueError("matched membership mismatch")
        if row["phase"] == 5 and "witness" not in rec:
            raise ValueError("phase5 witness missing")
        if "witness" in rec:
            w = rec["witness"]
            path = (root / w["file"]).resolve()
            if path.parent != (root / "phase5-witness").resolve():
                raise ValueError("witness outside its owned directory")
            if hashlib.sha256(path.read_bytes()).hexdigest() != w["sha256"]:
                raise ValueError("witness digest mismatch")
    return envelope


def _receipts(root: Path, arm: str, rows: list[dict], returned: bool) -> list[dict]:
    result = []
    for row in rows:
        path = root / "arms" / arm / row["episode"] / "summary.json"
        if path.exists():
            item = {"episode": row["episode"], "status": "unresolved", "summary": str(path.relative_to(root)),
                    "summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            try:
                r = json.loads(path.read_text())
                if r.get("episode") != row["episode"] or type(r.get("complete")) is not bool:
                    raise ValueError("summary episode/completeness is invalid")
                item["status"] = "policy_failed" if r.get("status") == "policy_failed" else (
                    "complete" if r["complete"] else "censored")
            except (ValueError, TypeError, AttributeError) as exc:
                item["receipt_error"] = f"{type(exc).__name__}: {exc}"
        else:
            # A crashed worker's missing receipt does not establish that a request/flight never started.
            item = {"episode": row["episode"], "status": "unattempted" if returned else "unresolved"}
        result.append(item)
    return result


def cost_index(root: Path) -> dict:
    """References to original arm ledgers, not an additional charge or invented per-arm GPU allocation."""
    root = Path(root)
    arms = {}
    for arm in ARMS:
        out = root / "arms" / arm
        arms[arm] = {"status_file": f"{arm}-state.json" if (root / f"{arm}-state.json").exists() else None,
                     "ledger_files": [{"path": str(p.relative_to(root)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                                      for p in sorted(out.rglob("*ledger.jsonl"))],
                     "cpu_work_files": [str(p.relative_to(root)) for p in sorted(out.rglob("work.jsonl"))]}
    result = {"arms": arms, "parent_allocated_gpu_h": None,
              "note": "Index only; shared allocation is measured once at parent/job level. Missing is unknown, never zero."}
    _write(root / "cost-index.json", result)
    return result


@contextmanager
def _parent_lock(root: Path, wait: bool):
    with (root / "dispatch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _serial_dispatch(root: Path):
    # The serial mode holds the lock for the whole arm and refuses a second dispatch.
    return _parent_lock(root, wait=False)


def _tree_digest(path: Path) -> str:
    if not path.is_dir() or path.is_symlink():
        raise ValueError("case store must be an ordinary directory")
    if not (path / "cases.log").is_file():
        raise ValueError("case store has no cases.log")
    files = sorted(path.rglob("*"))
    if any(p.is_symlink() for p in files):
        raise ValueError("case store symlinks are not supported")
    return _sha({str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()})


def run_arm(root: Path, arm: str, *, learning_options: dict | None = None,
            adviser_provider: Any = None, max_steps: int | None = None, concurrent: bool = False) -> dict:
    """Dispatch one fresh arm. Both learning arms have exactly the same settings except no_replay.

    Serial by default, refusing a concurrent dispatch. With concurrent=True only setup and close-out hold the parent
    lock (see the module notes). It never starts servers; the supplied existing policy adapters perform their calls.
    The caller must supply both model configurations for a learning arm or an adviser provider explicitly.
    """
    root = Path(root)
    if concurrent:
        return _run_arm(root, arm, learning_options=learning_options, adviser_provider=adviser_provider,
                        max_steps=max_steps, lock=lambda: _parent_lock(root, wait=True))
    with _serial_dispatch(root):
        return _run_arm(root, arm, learning_options=learning_options, adviser_provider=adviser_provider,
                        max_steps=max_steps, lock=nullcontext)


def _run_arm(root: Path, arm: str, *, learning_options: dict | None,
             adviser_provider: Any, max_steps: int | None, lock: Any) -> dict:
    began_epoch, began = time.time(), time.monotonic()
    with lock():
        setup = _setup(root, arm, learning_options, adviser_provider, max_steps)
    schedule, body, rows, scenarios, profile, profile_record, out, state_path = setup
    error = None
    returned = False
    try:
        if arm in ("full", "no_replay"):
            _fly_learning(root, arm, body, profile, profile_record, out, scenarios, max_steps)
        else:
            out.mkdir(exist_ok=False)
            for s in scenarios:
                if arm == "reasoning_alone":
                    adviser.run_adviser_episode(s, out, f"comparison/{arm}/{body['stream']}", adviser_provider,
                                                max_steps=max_steps)
                else:
                    comparators.run_pilot_episode(s, comparators.conditions()[arm], out / s.episode_id,
                                                  f"comparison/{arm}/{body['stream']}", max_steps=max_steps)
        returned = True
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        finished_epoch, elapsed = time.time(), time.monotonic() - began
        with lock():
            state = {"status": "returned" if returned else "failed", "schedule_sha256": schedule["sha256"],
                     "positions": _receipts(root, arm, rows, returned), "error": error,
                     "worker_pid": os.getpid(), "execution_started_epoch_s": began_epoch,
                     "execution_finished_epoch_s": finished_epoch, "execution_wall_s": elapsed}
            if returned and any("receipt_error" in p for p in state["positions"]):
                state["status"] = "invalid_receipts"
            _write(state_path, state)
            cost_index(root)
    return state


def _setup(root: Path, arm: str, learning_options: dict | None, adviser_provider: Any, max_steps: int | None):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError("max_steps must be a positive integer or None")
    schedule = load(root)
    body = schedule["schedule"]
    execution = {"max_steps": max_steps}
    execution_path = root / "execution-profile.json"
    if execution_path.exists() and json.loads(execution_path.read_text()) != execution:
        raise ValueError("arm execution limits differ")
    rows = [r for r in body["rows"] if arm != "reasoning_alone" or r["adviser"]]
    scenarios = tuple(_scenario(r["scenario"]) for r in rows)
    if arm in ("full", "no_replay"):
        allowed = {"local", "learn", "dev_cases", "trusted_heading_guard", "trusted_report_guard", "no_pause", "small_interval_s"}
        required = allowed - {"no_pause", "small_interval_s"}
        if learning_options is None or set(learning_options) - allowed or not required <= set(learning_options):
            raise ValueError("learning arms require an explicit shared learning profile, store and guards")
        if any(learning_options[k] is None for k in required):
            raise ValueError("no implicit stand-in learning arm")
        profile = deepcopy(learning_options)
        profile["dev_cases"] = str(Path(profile["dev_cases"]).resolve())
        if root.resolve().is_relative_to(Path(profile["dev_cases"])):
            raise ValueError("comparison output cannot be inside the input case store")
        for key in ("trusted_heading_guard", "trusted_report_guard"):
            profile[key] = str(Path(profile[key]).resolve())
        profile.setdefault("no_pause", False)
        profile.setdefault("small_interval_s", 30)
        # The two sequential dispatches must not silently change the ablation's common profile.
        profile_record = json.loads(json.dumps(profile, default=str))
        profile_record["store_sha256"] = _tree_digest(Path(profile["dev_cases"]))
        profile_record["guard_sha256"] = {k: hashlib.sha256(Path(profile[k]).read_bytes()).hexdigest()
                                          for k in ("trusted_heading_guard", "trusted_report_guard")}
        prior = root / "learning-profile.json"
        if prior.exists() and json.loads(prior.read_text()) != profile_record:
            raise ValueError("learning arm profiles differ")
    else:
        profile_record = None
        if learning_options is not None:
            raise ValueError("learning options apply only to learning arms")
    if arm == "reasoning_alone" and adviser_provider is None:
        raise ValueError("adviser needs an explicit provider")
    out = root / "arms" / arm
    out.parent.mkdir(exist_ok=True)
    # Claim before any dispatch; refuse retries and reuse, including a crash left at this boundary.
    state_path = root / f"{arm}-state.json"
    with state_path.open("x") as f:
        f.write(json.dumps({"status": "running", "schedule_sha256": schedule["sha256"],
                            "worker_pid": os.getpid(), "execution_claimed_epoch_s": time.time(),
                            "positions": [{"episode": r["episode"], "status": "scheduled"} for r in rows]}))
    _write(execution_path, execution)
    if profile_record is not None:
        _write(root / "learning-profile.json", profile_record)
    return schedule, body, rows, scenarios, (profile if profile_record is not None else None), profile_record, out, \
        state_path


def _fly_learning(root: Path, arm: str, body: dict, profile: dict, profile_record: dict, out: Path,
                  scenarios: tuple, max_steps: int | None) -> None:
    # CaseStore can append confirmed wrong cases. Copy it as well as guards so one arm cannot train the other.
    inputs = root / "inputs" / arm
    inputs.mkdir(parents=True, exist_ok=False)
    shutil.copytree(profile["dev_cases"], inputs / "case-store")
    if _tree_digest(inputs / "case-store") != profile_record["store_sha256"]:
        raise ValueError("case store changed while copying")
    runtime = deepcopy(profile)
    runtime["dev_cases"] = inputs / "case-store"
    for key in ("trusted_heading_guard", "trusted_report_guard"):
        target = inputs / f"{key}.py"
        shutil.copyfile(profile[key], target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != profile_record["guard_sha256"][key]:
            raise ValueError("guard changed while copying")
        runtime[key] = target
    run_stream.run_stream(20, body["stream"], out, stream=f"comparison/{arm}/{body['stream']}",
                          stream_seed=body["stream"], phases=tuple(body["phases"]), empty_library=True,
                          learn_checks=True, floor_after_cold_start=True, no_replay=arm == "no_replay",
                          prepared_scenarios=scenarios, max_steps=max_steps, **runtime)


def main(argv: list[str] | None = None) -> None:
    """`prepare` (also the default, for the original form with no command) or `run` one arm."""
    argv = list(argv if argv is not None else __import__("sys").argv[1:])
    if argv and argv[0] == "run":
        p = argparse.ArgumentParser(description="Fly one arm on a prepared schedule.")
        p.add_argument("--root", type=Path, required=True)
        p.add_argument("--arm", choices=ARMS, required=True)
        p.add_argument("--learning-profile", type=Path, help="JSON file: the learning arms' shared options")
        p.add_argument("--adviser-url", help="the reasoning model server, for reasoning_alone")
        p.add_argument("--adviser-model", default="Qwen/Qwen3.8-27B")
        p.add_argument("--max-steps", type=int)
        p.add_argument("--concurrent", action="store_true", help="hold the parent lock only for setup and close-out")
        a = p.parse_args(argv[1:])
        learning = json.loads(a.learning_profile.read_text()) if a.learning_profile else None
        provider = None
        if a.arm == "reasoning_alone":
            if not a.adviser_url:
                p.error("reasoning_alone needs --adviser-url")
            import telohelm.local_model as local_model
            provider = local_model.provider(a.adviser_url, a.adviser_model, adviser.TIMEOUT_S, role="reasoning_model")
        state = run_arm(a.root, a.arm, learning_options=learning, adviser_provider=provider, max_steps=a.max_steps,
                        concurrent=a.concurrent)
        print(json.dumps({"arm": a.arm, "status": state["status"]}))
        return
    if argv and argv[0] == "prepare":
        argv = argv[1:]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stream", type=int, required=True)
    p.add_argument("--matching-seed", type=int, required=True)
    p.add_argument("--phases", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    a = p.parse_args(argv)
    print(json.dumps({"schedule_sha256": prepare(a.out, a.stream, a.matching_seed, tuple(a.phases))["sha256"],
                      "note": "Inputs prepared only; no condition or model executed."}))


if __name__ == "__main__":
    main()
