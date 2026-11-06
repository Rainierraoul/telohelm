"""Scored-study runner for the accepted Experiment 2 design (protocol 0.12 with the 26 October amendments).

A study is a directory holding study.json, which declares its streams before anything flies, and one prepared schedule
per stream in a subdirectory named by the stream. Each schedule is generated once, with every generation receipt, phase
5 certificate and with-help witness kept, the unflown phase 5 world bound privately, every scenario fingerprinted and
the preparation's own cost recorded. The conditions fly those schedules:

- L, Telohelm learning: all positions of a stream in one run_stream process, procedures only (both permanent guards,
  no check or pair learning), learning through phase 5, the cold-start floor on. It copies its library after episode
  p2-e19 and its learning boundary for F.
- C, Telohelm without learning: the C positions, each in a fresh process with an empty library plus the two guards and
  learning off, the reasoning model kept for flight.
- F, Telohelm with a frozen library: phase 3 indices 5 to 9 and 15 to 19, each in a fresh process from a new copy of
  L's snapshot, learning off.
- the intercept calculator and the adaptive controller on every position, and the optional adviser on its matched
  positions.

The protocol's pause counts a condition's distinct reasoning calls over its first 20 scheduled positions. Here that
anchor is explicit across streams: the condition's positions are taken in turn from the declared streams (the first of
each, then the second of each, and so on) and the first 20 form the anchor. After every position, and for L after
every episode and learning boundary through run_stream's pause_authority hook, the stream's current pause rows are
written to the study and the condition's rows from every stream are judged together. A due pause writes a lasting stop
record for that condition, and every stream of the condition stops before its next position.
Adviser amendment 2 instead records the first pause the same predicate would make, without stopping the adviser.
The other conditions retain their original pause enforcement.

Scored streams 4301 to 4304 are bound to the protocol's fixed C table (seeds 6201 to 6204) and adviser draw (6001 to
6004). Scored preparation and execution need a verified freeze record: the protocol file's digest, the generator
source's digest and the shared flight profile's digest, all checked before anything is drawn or flown; stand-in models
and step limits are refused. Development studies use unreserved development streams, C seeds 6600 to 6699 and adviser
seeds 6400 to 6499.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np

from . import adviser as adviser
from . import comparators as comparators
from . import outcomecases as outcomecases
from . import run_stream as run_stream
from . import streams as streams
from .comparison import _parent_lock, _scenario, _sha, _write
import telohelm.qualification.store as store
import telohelm.taskpacks.tanker.world.episode as world_episode
import telohelm.taskpacks.tanker.world.phase5 as world_phase5
import telohelm.taskpacks.tanker.world.physics as world_physics
import telohelm.taskpacks.tanker.world.pilot as world_pilot
import telohelm.taskpacks.tanker.dynamics as world_dynamics
from telohelm.ladder import PAUSE_WINDOW, pause_due

ARMS = ("L", "C", "F", "calculator", "adaptive_controller", "adviser", "C-full")
TELOHELM = ("L", "C", "F", "C-full")
PAUSED = ("L", "C", "F", "adviser", "C-full")              # the conditions that make reasoning calls
UNRATIONED = ("C-full",)       # C's positions with the small-model ration lifted (protocol amendment 7)
HORIZON_S = world_episode.HORIZON                       # seconds; the unrationed allowance cannot run out
PAUSE_FIELDS = ("kind", "scope", "status", "episode", "request_id", "recovers")    # what the pause rule reads
PHASES = (1, 2, 3, 4, 5)
PER_PHASE = 20
LAYOUT = "v2"
F_INDICES = tuple(range(5, 10)) + tuple(range(15, 20))       # phase 3: episodes 46 to 50 and 56 to 60
FORK = (2, 19)                                              # L's library is copied after p2-e19 and its boundary
SCORED_STREAMS = tuple(store.SCORED_STREAMS)
SCORED_C_SEEDS = {4301: 6201, 4302: 6202, 4303: 6203, 4304: 6204}
SCORED_ADVISER_SEEDS = {4301: 6001, 4302: 6002, 4303: 6003, 4304: 6004}
# The protocol's published C table (section 4), reproduced by c_draw from the seeds above; the runner uses the table.
SCORED_C_TABLE = {
    4301: {1: (2, 6, 9, 11, 18), 2: (5, 6, 8, 9, 10), 3: (1, 4, 6, 13, 16), 4: (2, 3, 7, 11, 19)},
    4302: {1: (2, 3, 13, 16, 17), 2: (2, 5, 9, 14, 19), 3: (0, 4, 8, 11, 17), 4: (1, 8, 12, 14, 18)},
    4303: {1: (1, 3, 5, 11, 18), 2: (6, 9, 11, 14, 17), 3: (1, 2, 7, 10, 14), 4: (2, 7, 9, 11, 15)},
    4304: {1: (4, 9, 11, 12, 15), 2: (2, 4, 7, 15, 17), 3: (0, 7, 14, 16, 17), 4: (2, 4, 5, 13, 14)},
}
DEV_C_SEEDS = range(6600, 6700)
DEV_ADVISER_SEEDS = range(6400, 6500)
STAND_IN_LEARN = {"base_url": "http://127.0.0.1:9", "model": "stand-in", "timeout_s": 1.0}
REPO = Path(__file__).resolve().parents[2]
FREEZE_KEYS = {"frozen", "protocol_path", "protocol_sha256", "generator_sha256", "profile_sha256"}
HEX64 = re.compile(r"[0-9a-f]{64}")
GRACE_S = 10.0          # how long a position's worker has to stop after SIGTERM before its group is killed
WORKER_TOKEN = "TELOHELM_WORKER_TOKEN"   # a fresh random marker in each worker's environment, kept in its record


class DispatchStopped(Exception):
    """The dispatcher was asked to stop. Its position worker, if any, was closed and reaped first."""

    def __init__(self, signals: list[int]):
        super().__init__(f"dispatcher stopped by signal {signals[0]}")
        self.signals = list(signals)


def c_draw(seed: int) -> dict[int, tuple[int, ...]]:
    """C's matched indices for phases 1 to 4 (protocol section 4): 5 of 20 in phases 1, 2 and 4; in phase 3, 2 of 0 to
    9, then 2 of 10 to 19, then one of the 16 left. Indices only; no scenario is generated."""
    rng = np.random.Generator(np.random.PCG64(seed))
    out = {}
    for phase in (1, 2, 3, 4):
        if phase == 3:
            first = [int(i) for i in rng.choice(10, 2, replace=False)]
            second = [10 + int(i) for i in rng.choice(10, 2, replace=False)]
            rest = [i for i in range(PER_PHASE) if i not in first + second]
            picks = first + second + [rest[int(rng.integers(0, len(rest)))]]
        else:
            picks = [int(i) for i in rng.choice(PER_PHASE, 5, replace=False)]
        out[phase] = tuple(sorted(picks))
    return out


def reserved_identities() -> set[int]:
    part = outcomecases.partition()
    sources = {x for roles in part.values() for block in roles.values() for x in block}
    return (sources | set(outcomecases.EXPOSED_SOURCES) | set(outcomecases.C0_DEVELOPMENT)
            | set(outcomecases.RUN_STREAMS) | set(SCORED_STREAMS))


GENERATOR = (streams, world_phase5, world_episode, world_physics, world_pilot, world_dynamics)


def generator_sha256() -> str:
    """The digest of the generator source actually imported: the stream generator, the tanker world it draws and the
    reference pilot that certifies phase 5 help."""
    return _sha({m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in GENERATOR})


def verify_freeze(freeze: Any, protocol_root: Path = REPO) -> dict:
    """A freeze record is accepted only if the protocol file and the generator source still have its digests."""
    if not isinstance(freeze, dict) or set(freeze) != FREEZE_KEYS or freeze["frozen"] is not True:
        raise ValueError(f"a freeze record needs exactly {sorted(FREEZE_KEYS)} with frozen true")
    for key in ("protocol_sha256", "generator_sha256", "profile_sha256"):
        if not isinstance(freeze[key], str) or not HEX64.fullmatch(freeze[key]):
            raise ValueError(f"the freeze record's {key} is not a SHA-256 digest")
    path = (Path(protocol_root) / freeze["protocol_path"]).resolve()
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != freeze["protocol_sha256"]:
        raise ValueError("the protocol file does not have the frozen digest")
    if generator_sha256() != freeze["generator_sha256"]:
        raise ValueError("the generator source does not have the frozen digest")
    return freeze


def _draws(mode: str, stream: int, c_seed: int, adviser_seed: int) -> tuple[dict, dict]:
    if mode == "scored":
        table = SCORED_C_TABLE[stream]
        if {p: tuple(v) for p, v in c_draw(c_seed).items()} != table:
            raise ValueError("the fixed C table and its seed disagree")
        c = {str(p): list(v) for p, v in table.items()}
    else:
        c = {str(p): list(v) for p, v in c_draw(c_seed).items()}
    return c, adviser.matched_indices(adviser_seed)


def _check_identity(mode: str, stream: int, c_seed: int, adviser_seed: int, phases: tuple, freeze: Any):
    if mode == "scored":
        if stream not in SCORED_STREAMS or c_seed != SCORED_C_SEEDS[stream] or adviser_seed != SCORED_ADVISER_SEEDS[stream]:
            raise ValueError("scored streams use their own fixed C and adviser seeds")
        verify_freeze(freeze)
        if tuple(phases) != PHASES:
            raise ValueError("a scored schedule has all five phases")
    else:
        if freeze is not None:
            raise ValueError("a development schedule has no freeze record")
        if type(stream) is not int or stream not in store.DEVELOPMENT_STREAMS or stream in reserved_identities():
            raise ValueError("development schedules use unreserved development streams only")
        if c_seed not in DEV_C_SEEDS or adviser_seed not in DEV_ADVISER_SEEDS:
            raise ValueError("development seeds: C 6600 to 6699, adviser 6400 to 6499")
    if not phases or tuple(sorted(set(phases))) != tuple(phases) or any(p not in PHASES for p in phases):
        raise ValueError("phases must be unique and increasing, in 1 to 5")


def declare_study(study: Path, stream_list: list[int]) -> dict:
    """Declare a study's streams, in order, before anything is prepared or flown; the pause anchor depends on it."""
    study = Path(study)
    if not stream_list or len(set(stream_list)) != len(stream_list) or any(type(s) is not int for s in stream_list):
        raise ValueError("a study declares distinct integer streams")
    scored = [s in SCORED_STREAMS for s in stream_list]
    if any(scored) and not all(scored):
        raise ValueError("a study is either scored or development, never both")
    study.mkdir(parents=True, exist_ok=False)
    probe, failed = study / ".link-probe", None
    probe.write_bytes(b"")
    try:
        os.link(probe, study / ".link-probe-2")
    except OSError as exc:
        failed = exc
    for p in (probe, study / ".link-probe-2"):
        p.unlink(missing_ok=True)
    if failed is not None:
        study.rmdir()
        raise ValueError(f"the study folder cannot make hard links, which publish the stop record ({failed})")
    record = {"streams": list(stream_list), "arms": list(ARMS), "pause_window": PAUSE_WINDOW}
    _write(study / "study.json", record)
    return record


def _study(root: Path) -> tuple[Path, dict]:
    study = Path(root).parent
    path = study / "study.json"
    if not path.is_file():
        raise ValueError("a stream schedule lives in a declared study (study.json beside it)")
    record = json.loads(path.read_text())
    if Path(root).name not in {str(s) for s in record["streams"]}:
        raise ValueError("this stream is not declared in its study")
    return study, record


def _final(world: Any, deadline: int) -> Any:
    return replace(world, goals=((0, replace(world.goals[0][1], deadline=deadline)),))


def _bind_phase5(root: Path, stream: int, index: int, g: Any) -> tuple[dict, dict]:
    """Check the certificate's meaning and return what binds it to its two final scenarios: the accepted seed, the
    flown world and both scenarios' fingerprints with the certified deadline. The unflown one is kept privately, so
    reproduction needs no later draw."""
    accepted = [a for a in g.attempts if a.get("help_certified")]
    if len(accepted) != 1 or accepted[0] is not g.attempts[-1] or g.witness.get("label") != "help certified":
        raise ValueError("phase 5 needs exactly one certified attempt, the last one")
    d, flown = streams.draw_help(stream, index, accepted[0]["attempt"])
    if d.seed != accepted[0]["seed"] or d.seed != g.witness["seed"] or flown != accepted[0]["flown"]:
        raise ValueError("the certified attempt's seed or flown world does not reproduce")
    final = {"a": _final(d.world_a, g.witness["deadline"]), "b": _final(d.world_b, g.witness["deadline"])}
    if comparators.scenario_fingerprint(final[flown]) != comparators.scenario_fingerprint(g.scenario):
        raise ValueError("the flown world is not the certified one")
    other = "b" if flown == "a" else "a"
    private = root / "private" / "alternative-world"
    private.mkdir(parents=True, exist_ok=True)
    data = json.dumps(asdict(final[other]), sort_keys=True).encode()
    (private / f"{g.scenario.episode_id}.json").write_bytes(data)
    binding = {"seed": d.seed, "flown": flown,
               "scenario_sha256": {k: comparators.scenario_fingerprint(w) for k, w in final.items()}}
    alternative = {"file": f"private/alternative-world/{g.scenario.episode_id}.json",
                   "sha256": hashlib.sha256(data).hexdigest(), "world": other}
    return binding, alternative


def _check_phase5(root: Path, row: dict, rec: dict, scenario: Any) -> None:
    """Join the retained records without regenerating anything: the witness and its binding, the accepted attempt,
    the scheduled (flown) scenario and the private unflown one must all name the same seed, world and deadline, and
    both scenarios must have the fingerprints the certificate bound."""
    w = rec["witness"]
    path = (root / w["file"]).resolve()
    if path.parent != (root / "phase5-witness").resolve():
        raise ValueError("witness outside its owned directory")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != w["sha256"]:
        raise ValueError("witness digest mismatch")
    witness = json.loads(gzip.decompress(raw))
    binding = witness.pop("binding", None)
    accepted = [a for a in rec["attempts"] if a.get("help_certified")]
    if witness.get("label") != "help certified" or len(accepted) != 1 or accepted[0] != rec["attempts"][-1]:
        raise ValueError("the witness is not one certified, accepted attempt")
    if json.loads(json.dumps(streams._digest_traces(witness), default=str)) != accepted[0]["assessment"]:
        raise ValueError("the witness does not match the accepted attempt's assessment")
    shas = binding.get("scenario_sha256") if isinstance(binding, dict) else None
    if (set(binding or {}) != {"seed", "flown", "scenario_sha256"} or not isinstance(shas, dict) or set(shas) != {"a", "b"}
            or not all(isinstance(v, str) and HEX64.fullmatch(v) for v in shas.values())):
        raise ValueError("the certificate does not bind its two final scenarios")
    flown = accepted[0]["flown"]
    if flown not in ("a", "b") or binding["flown"] != flown or not (accepted[0]["seed"] == binding["seed"] == witness.get("seed")):
        raise ValueError("the accepted seed or selected world is not the certificate's")
    if scenario.goals[0][1].deadline != witness["deadline"]:
        raise ValueError("the scenario's deadline is not the certificate's")
    if comparators.scenario_fingerprint(scenario) != shas[flown]:
        raise ValueError("the flown scenario is not the certified one")
    other = "b" if flown == "a" else "a"
    alt = rec.get("alternative_world")
    if not isinstance(alt, dict) or alt.get("world") != other:
        raise ValueError("the unflown world is not bound")
    alt_path = (root / str(alt.get("file", ""))).resolve()
    data = alt_path.read_bytes() if alt_path.is_file() else b""
    if alt_path.parent != (root / "private" / "alternative-world").resolve() or \
            hashlib.sha256(data).hexdigest() != alt.get("sha256"):
        raise ValueError("the unflown world's record differs")
    try:
        unflown = _scenario(json.loads(data))
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"the unflown world's record is not a scenario ({type(exc).__name__})") from None
    if (unflown.episode_id != scenario.episode_id or unflown.goals[0][1].deadline != witness["deadline"]
            or comparators.scenario_fingerprint(unflown) != shas[other]):
        raise ValueError("the unflown world is not the certified one")


def prepare(root: Path, stream: int, *, c_seed: int, adviser_seed: int, phases: tuple = PHASES,
            freeze: dict | None = None) -> dict:
    """Generate every input once, with every attempt, certificate and witness kept and the unflown phase 5 world
    bound privately, before any condition can fly. The preparation's wall and processor time are recorded, also on
    failure."""
    mode = "scored" if stream in SCORED_STREAMS else "development"
    phases = tuple(phases)
    _check_identity(mode, stream, c_seed, adviser_seed, phases, freeze)
    root = Path(root)
    if root.name != str(stream):
        raise ValueError("a stream's schedule lives in a folder named by the stream")
    _study(root)
    c, adv = _draws(mode, stream, c_seed, adviser_seed)
    root.mkdir(parents=False, exist_ok=False)
    began, cpu = time.time(), time.process_time()
    positions = [{"phase": p, "index": i, "episode": streams.episode_id(stream, p, i)}
                 for p in phases for i in range(PER_PHASE)]
    status, envelope = {"status": "preparing"}, None
    _write(root / "preparation.json", {**status, "positions": positions})
    try:
        rows = []
        with (root / "generation.jsonl").open("x") as log:
            for pos in positions:
                try:
                    g = streams.episode(stream, pos["phase"], pos["index"], LAYOUT)
                except streams.GenerationError as exc:
                    log.write(json.dumps(streams.cap_record(stream, pos["phase"], pos["index"], exc), default=str) + "\n")
                    log.flush()
                    status = {"status": "generation_failed", "episode": pos["episode"], "error": str(exc)}
                    raise
                if g.scenario.episode_id != pos["episode"]:
                    raise ValueError("generator returned a different episode identity")
                rec = streams.generation_record(g)
                if pos["phase"] == 5:
                    if g.witness is None:
                        raise ValueError("phase 5 input has no retained certificate and witness")
                    binding, alternative = _bind_phase5(root, stream, pos["index"], g)
                    rec["witness"] = run_stream._write_witness(root, g.scenario.episode_id,
                                                               {**g.witness, "binding": binding})
                    rec["alternative_world"] = alternative
                rec = json.loads(json.dumps(rec, default=str))
                log.write(json.dumps(rec, sort_keys=True) + "\n")
                log.flush()
                p, i = pos["phase"], pos["index"]
                rows.append({**pos, "scenario": asdict(g.scenario),
                             "scenario_sha256": comparators.scenario_fingerprint(g.scenario),
                             "generation_sha256": _sha(rec),
                             "c": p in (1, 2, 3, 4) and i in c[str(p)],
                             "adviser": i in adv[str(p)]["chronological_indices"],
                             "f": p == 3 and i in F_INDICES})
        body = {"version": 2, "mode": mode, "stream": stream, "c_seed": c_seed, "adviser_seed": adviser_seed,
                "layout": LAYOUT, "phases": list(phases), "arm_order": list(ARMS), "c_draw": c, "adviser_draw": adv,
                "freeze": freeze, "rows": rows}
        envelope = {"sha256": _sha(body), "schedule": body}
        _write(root / "schedule.json", envelope)
        status = {"status": "ready", "schedule_sha256": envelope["sha256"]}
        return envelope
    except BaseException as exc:
        if status["status"] == "preparing":
            status = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        raise
    finally:
        _write(root / "preparation.json", {**status, "positions": positions, "preparation_wall_s": time.time() - began,
                                           "preparation_process_s": time.process_time() - cpu})


def load(root: Path) -> dict:
    """Check the schedule, its draws and memberships, every scenario and generation receipt, and every phase 5
    certificate, witness and private world. Nothing is regenerated."""
    root = Path(root)
    _study(root)
    envelope = json.loads((root / "schedule.json").read_text())
    body = envelope["schedule"]
    if envelope["sha256"] != _sha(body):
        raise ValueError("schedule digest mismatch")
    if body["version"] != 2 or body["arm_order"] != list(ARMS) or body["layout"] != LAYOUT:
        raise ValueError("unsupported study schedule")
    if root.name != str(body["stream"]):
        raise ValueError("a stream's schedule lives in a folder named by the stream")
    phases = tuple(body["phases"])
    _check_identity(body["mode"], body["stream"], body["c_seed"], body["adviser_seed"], phases, body["freeze"])
    c, adv = _draws(body["mode"], body["stream"], body["c_seed"], body["adviser_seed"])
    if body["c_draw"] != c or body["adviser_draw"] != adv:
        raise ValueError("matched draw mismatch")
    expected = [(p, i, streams.episode_id(body["stream"], p, i)) for p in phases for i in range(PER_PHASE)]
    if [(r["phase"], r["index"], r["episode"]) for r in body["rows"]] != expected:
        raise ValueError("schedule positions mismatch")
    generation = [json.loads(line) for line in (root / "generation.jsonl").read_text().splitlines()]
    if len(generation) != len(body["rows"]):
        raise ValueError("generation receipt count mismatch")
    for row, rec in zip(body["rows"], generation):
        p, i = row["phase"], row["index"]
        if row["generation_sha256"] != _sha(rec) or rec["episode"] != row["episode"] or rec.get("layout") != LAYOUT:
            raise ValueError("generation receipt mismatch")
        s = _scenario(row["scenario"])
        if s.episode_id != row["episode"] or comparators.scenario_fingerprint(s) != row["scenario_sha256"]:
            raise ValueError("scenario digest mismatch")
        if (row["c"] is not (p in (1, 2, 3, 4) and i in c[str(p)])
                or row["adviser"] is not (i in adv[str(p)]["chronological_indices"])
                or row["f"] is not (p == 3 and i in F_INDICES)):
            raise ValueError("membership mismatch")
        if p == 5:
            if "witness" not in rec:
                raise ValueError("phase 5 certificate and witness missing")
            _check_phase5(root, row, rec, s)
    return envelope


def verify_snapshot(root: Path, stream: int) -> dict:
    """L's library snapshot for F, checked: its marker names this stream's p2-e19, and its manifest lists exactly the
    library's files with their digests. Returns the marker with the digest of its bytes."""
    folder = Path(root) / "inputs" / "F-snapshot"
    try:
        raw = (folder / "snapshot.json").read_bytes()
        marker = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise ValueError(f"F needs L's library snapshot after p2-e19 ({type(exc).__name__})") from None
    if not isinstance(marker, dict) or marker.get("after_episode") != streams.episode_id(stream, *FORK):
        raise ValueError("the snapshot is not this stream's p2-e19 boundary")
    files = marker.get("files")
    if not isinstance(files, dict) or "events.log" not in files or run_stream._files(folder / "library") != files:
        raise ValueError("the snapshot's library does not match its manifest")
    return {**marker, "snapshot_sha256": hashlib.sha256(raw).hexdigest()}


def _profile_record(profile: dict, executable: bool = True) -> dict:
    """The one flight profile every Telohelm condition shares, refused if incomplete, with the guards' hashes. An
    executable profile (the default) must name each model server by a real origin, checked by the loopback transport
    that will carry its requests: http, a numeric loopback address, a port and nothing after it. A freeze template,
    made with executable=False, may name them by role instead, as profile_digest does."""
    keys = {"stand_in", "local", "learn", "small_interval_s", "watch_briefing", "model_report_control",
            "learn_candidates", "learn_regression"}
    if set(profile) != keys:
        raise ValueError(f"the flight profile needs exactly {sorted(keys)}")
    if type(profile["watch_briefing"]) is not bool or type(profile["model_report_control"]) is not bool:
        raise ValueError("the watch and report-control settings must be fixed booleans")
    if profile["stand_in"] is not True and (not profile["local"] or not profile["learn"]):
        raise ValueError("a real profile names both model servers")
    if profile["stand_in"] is not True:
        from telohelm.runtime.transport import LoopbackTransport
        for key, role in SERVER_ROLES.items():
            url = profile[key].get("base_url") if isinstance(profile[key], dict) else None
            if not executable and url == role:
                continue
            try:
                LoopbackTransport(url, 1)
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"the profile's {key} server is not a loopback origin ({exc})") from None
    record = json.loads(json.dumps(profile))
    record["guard_sha256"] = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in _guards().items()}
    return record


SERVER_ROLES = {"local": "{small_model}", "learn": "{reasoning_model}"}


def profile_digest(record: dict) -> str:
    """The frozen identity of a flight profile record: every field, with each model server's address replaced by its
    role. Addresses depend on the ports a job is given, so they cannot be frozen; which model each role serves, and
    every other setting, can."""
    view = json.loads(json.dumps(record))
    for key, role in SERVER_ROLES.items():
        if isinstance(view.get(key), dict) and "base_url" in view[key]:
            view[key]["base_url"] = role
    return _sha(view)


def _admit(body: dict, record: dict | None, max_steps: int | None) -> None:
    """Scored flights need the verified freeze, the frozen profile with real models, and no step limit."""
    if body["mode"] != "scored":
        return
    verify_freeze(body["freeze"])
    if max_steps is not None:
        raise ValueError("scored flights have no step limit")
    if record is not None:
        if record["stand_in"] is not False:
            raise ValueError("scored flights use the real profile")
        if profile_digest(record) != body["freeze"]["profile_sha256"]:
            raise ValueError("the flight profile is not the frozen one")


def _guards() -> dict:
    import telohelm
    root = Path(telohelm.__file__).resolve().parent / "taskpacks/tanker/guards"
    return {"heading": root / "poc-heading-guard.py", "report": root / "poc-report-guard.py"}


def _flight(profile: dict, arm: str = "L") -> dict:
    """The shared flight settings. C-full keeps every one of them except the small model's ration: no interval
    between calls, and an allowance of the horizon in seconds, which cannot run out (run_stream.run_stream)."""
    flight = {"small_interval_s": profile["small_interval_s"], "own_reports": True,
              "watch_briefing": profile["watch_briefing"], "model_report_control": profile["model_report_control"]}
    if arm in UNRATIONED:
        flight.update(small_interval_s=0, small_call_limit=HORIZON_S)
    return flight


def _rows_for(arm: str, rows: list[dict]) -> list[dict]:
    if arm in ("C", "C-full"):
        return [r for r in rows if r["c"]]
    if arm == "F":
        return [r for r in rows if r["f"]]
    if arm == "adviser":
        return [r for r in rows if r["adviser"]]
    return rows


# -- the condition-wide pause ------------------------------------------------------------------------------------------
def pause_anchor(study: Path, arm: str) -> list[str]:
    """The condition's first 20 scheduled positions, taken in turn from the declared streams, every one of which must
    already be prepared, so the anchor cannot change once anything flies."""
    record = json.loads((Path(study) / "study.json").read_text())
    lists = []
    for s in record["streams"]:
        schedule = Path(study) / str(s) / "schedule.json"
        if not schedule.is_file():
            raise ValueError(f"declared stream {s} is not prepared; the pause anchor needs every stream")
        envelope = json.loads(schedule.read_text())
        if envelope["sha256"] != _sha(envelope["schedule"]):
            raise ValueError("schedule digest mismatch")
        lists.append([r["episode"] for r in _rows_for(arm, envelope["schedule"]["rows"])])
    merged = [lst[k] for k in range(max((len(x) for x in lists), default=0)) for lst in lists if k < len(lst)]
    return merged[:record["pause_window"]]


def stop_record(study: Path, arm: str) -> dict | None:
    path = Path(study) / "stops" / f"{arm}.json"
    return json.loads(path.read_text()) if path.is_file() else None


def counterfactual_stop_record(study: Path, arm: str) -> dict | None:
    """The first recorded pause that amendment 2 observes without enforcing for the adviser."""
    path = Path(study) / "counterfactual-stops" / f"{arm}.json"
    return json.loads(path.read_text()) if path.is_file() else None


def _publish_once(path: Path, value: dict) -> bool:
    """Publish complete JSON at path exactly once. It is written and synced under a private temporary name, then hard
    linked into place: a reader sees no file or the whole record, a second publisher's link fails, and an interrupted
    publisher leaves at most its temporary file."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    rest = memoryview(json.dumps(value, default=str).encode())
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            while rest:                          # a write may legally take only part of what it is given
                n = os.write(fd, rest)
                if n <= 0:
                    raise OSError("the stop record could not be written")
                rest = rest[n:]
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.link(tmp, path)
            return True
        except FileExistsError:
            return False
    finally:
        tmp.unlink(missing_ok=True)


def pause_authority(study: Path, arm: str, stream: int, rows: list[dict]) -> dict | None:
    """Record this stream's current pause rows, judge the condition's rows from every stream against the shared
    anchor, write the lasting stop record if a pause is due, and return the stop record if there is one."""
    study = Path(study)
    if arm not in PAUSED:
        raise ValueError("this condition makes no reasoning calls")
    folder = study / "pause" / arm
    folder.mkdir(parents=True, exist_ok=True)
    _write(folder / f"{stream}.json", [{k: r.get(k) for k in PAUSE_FIELDS} for r in rows])
    every = []
    for path in sorted(folder.glob("*.json")):
        if ".tmp" not in path.name:
            every += json.loads(path.read_text())
    pause = pause_due(every, pause_anchor(study, arm))
    if arm == "adviser":
        # Adviser amendment 2: preserve the frozen predicate and its first due record, but let bounded episodes
        # continue. Timeouts remain charged and recorded. No other condition's pause changes.
        if pause["due"] and counterfactual_stop_record(study, arm) is None:
            observed = study / "counterfactual-stops"
            observed.mkdir(exist_ok=True)
            _publish_once(observed / f"{arm}.json", {
                "arm": arm, "detected_in_stream": stream, **pause, "at_epoch_s": time.time(),
                "enforced": False, "amendment": "adviser-2-20261104",
                "note": "first observed frozen-rule pause; adviser amendment 2 records it without stopping"})
        return None
    stop = stop_record(study, arm)
    if pause["due"] and stop is None:
        stops = study / "stops"
        stops.mkdir(exist_ok=True)
        stop = {"arm": arm, "detected_in_stream": stream, **pause, "at_epoch_s": time.time(),
                "note": "the protocol's pause; every stream of this condition stops before its next position"}
        if not _publish_once(stops / f"{arm}.json", stop):
            stop = stop_record(study, arm)
    return stop


# -- flying ------------------------------------------------------------------------------------------------------------
def _fly_one(root: Path, arm: str, episode: str, max_steps: int | None) -> int:
    """One C or F position in this fresh process, from the verified schedule and the shared profile on record."""
    study, _ = _study(root)
    envelope = load(root)
    body = envelope["schedule"]
    path = root / f"{arm}-state.json"
    state = json.loads(path.read_text()) if path.is_file() else {}
    dispatcher = os.getppid()
    if (state.get("status") != "running" or state.get("worker_pid") != dispatcher
            or state.get("schedule_sha256") != envelope["sha256"]):
        raise ValueError("a position flies only under the condition's running dispatch")
    if stop_record(study, arm) is not None:
        raise ValueError("this condition is stopped")
    row = next(r for r in body["rows"] if r["episode"] == episode)
    if not (row["c"] if arm in ("C", "C-full") else row["f"]):
        raise ValueError("this position is not scheduled for this condition")
    record = json.loads((root / "flight-profile.json").read_text())
    profile = {k: v for k, v in record.items() if k != "guard_sha256"}
    if _profile_record(profile) != record:
        raise ValueError("the guards changed since the profile was recorded")
    _admit(body, record, max_steps)
    stand_in = profile["stand_in"] is True
    source = verify_snapshot(root, body["stream"]) if arm == "F" else None
    out = root / "arms" / arm / episode
    out.mkdir(parents=True, exist_ok=False)
    (out / "process.json").write_text(json.dumps({"pid": os.getpid(), "pgid": os.getpgid(0), "ppid": dispatcher,
                                                  "episode": episode}))
    if source is not None:
        _write(out / "f-source.json", source)
    g = _guards()
    snapshot = root / "inputs" / "F-snapshot"
    run_stream.run_stream(1, 1, out / "flight", stream=f"study/{arm}/{body['stream']}", stream_seed=body["stream"],
                          layout=LAYOUT, prepared_scenarios=(_scenario(row["scenario"]),),
                          local=None if stand_in else profile["local"],
                          learn=STAND_IN_LEARN if stand_in else profile["learn"], learning=False, empty_library=True,
                          trusted_heading_guard=None if arm == "F" else g["heading"],
                          trusted_report_guard=None if arm == "F" else g["report"],
                          library_from=snapshot / "library" if arm == "F" else None, no_pause=True,
                          timing=True, record_inputs=True, max_steps=max_steps, **_flight(profile, arm))
    if source is not None and json.loads((out / "flight" / "library-from.json").read_text())["files"] != source["files"]:
        raise ValueError("this F copy is not the snapshot's library")
    return 0


BOOT = ("import sys,runpy;sys.path.insert(0,sys.argv.pop(1));"
        "runpy.run_module(sys.argv.pop(1),run_name='__main__',alter_sys=True)")


def module_command(python: str, module: str, args: list[str], payload: str | Path | None = None) -> list[str]:
    """How every process of a study is started. Without a payload: python -B -m MODULE, from the inherited path (a
    working-tree rehearsal). With one: python -I -B -c BOOT PAYLOAD MODULE, isolated, so no environment variable,
    user site folder or working directory can put code on the path, and the payload root comes first."""
    if payload is None:
        return [python, "-B", "-m", module, *args]
    return [python, "-I", "-B", "-c", BOOT, str(Path(payload).resolve()), module, *args]


def _position_argv(root: Path, arm: str, episode: str, max_steps: int | None, payload: str | None = None) -> list[str]:
    args = ["fly-one", "--root", str(root), "--arm", arm, "--episode", episode]
    return module_command(sys.executable, "paper2.experiment2.study_runner",
                          args + (["--max-steps", str(max_steps)] if max_steps else []), payload)


def _signal_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):    # gone already, or (macOS) a group whose leader is a zombie
        pass


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:                         # (macOS) only zombies are left, which are already dead
        return False


def _close_group(child: subprocess.Popen) -> bool:
    """End everything in the worker's group, whether or not its leader has exited: SIGTERM, then SIGKILL after
    GRACE_S, reaping the leader. Returns whether anything in the group was still running when cleanup began."""
    pgid = child.pid
    running = child.poll() is None or _group_alive(pgid)
    _signal_group(pgid, signal.SIGTERM)
    end = time.monotonic() + GRACE_S
    while time.monotonic() < end:
        if child.poll() is not None and not _group_alive(pgid):
            break
        time.sleep(0.05)
    else:
        _signal_group(pgid, signal.SIGKILL)
        child.wait()
        end = time.monotonic() + 5.0
        while _group_alive(pgid) and time.monotonic() < end:
            time.sleep(0.05)
    return running


class _Owned:
    """The dispatcher's one position worker and the stop signals it has received. The handler only records the
    signal and passes SIGTERM to a worker already registered; it never raises, so it cannot break process creation."""

    def __init__(self):
        self.child: subprocess.Popen | None = None
        self.signals: list[int] = []

    def on_signal(self, signum, frame):
        self.signals.append(signum)
        if self.child is not None:
            _signal_group(self.child.pid, signal.SIGTERM)


class _handling:
    """SIGTERM and SIGINT go to the owned worker's handler while the C or F positions fly (main thread only), and the
    previous handlers come back afterwards."""

    def __init__(self, owned: _Owned):
        self.owned, self.previous = owned, {}

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            self.previous = {s: signal.signal(s, self.owned.on_signal) for s in (signal.SIGTERM, signal.SIGINT)}
        return self.owned

    def __exit__(self, *exc):
        for s, handler in self.previous.items():
            signal.signal(s, handler)
        return False


def _run_position(owned: _Owned, out: Path, episode: str, argv: list[str]) -> None:
    """Fly one position in a worker that leads its own process group. If the dispatcher is asked to stop, the worker
    gets SIGTERM, then SIGKILL after GRACE_S. Whatever happens, the whole group is closed and the worker reaped before
    ownership is released and the receipt written; a stop or error keeps an interrupted receipt. Nothing outside the
    group, such as a shared model server, is signalled."""
    if owned.signals:
        raise DispatchStopped(owned.signals)
    receipt = {"stdout": f"{episode}.stdout.txt", "stderr": f"{episode}.stderr.txt"}
    with open(out / receipt["stdout"], "wb") as so, open(out / receipt["stderr"], "wb") as se:
        try:
            token = secrets.token_hex(16)
            owned.child = subprocess.Popen(argv, stdout=so, stderr=se, start_new_session=True,
                                           env={**os.environ, WORKER_TOKEN: token})
            _write(out / f"{episode}.worker.json", {"pid": owned.child.pid, "pgid": owned.child.pid,
                                                     "dispatcher_pid": os.getpid(), "episode": episode,
                                                     "token": token})
            if owned.signals:                    # a stop arrived while the worker was being created
                _signal_group(owned.child.pid, signal.SIGTERM)
            receipt["pid"] = owned.child.pid
            asked_at = None
            while True:
                if owned.signals and asked_at is None:
                    asked_at = time.monotonic()
                try:
                    code = owned.child.wait(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    if asked_at is not None and time.monotonic() - asked_at > GRACE_S:
                        _signal_group(owned.child.pid, signal.SIGKILL)
        except BaseException as exc:
            if owned.child is not None:
                receipt["group_left_running"] = _close_group(owned.child)
            _write(out / f"{episode}.exit.json", {**receipt, "exit": None if owned.child is None else owned.child.returncode,
                                                  "interrupted": True, "error": type(exc).__name__})
            owned.child = None
            raise
    receipt["group_left_running"] = _close_group(owned.child)    # anything the worker left behind
    if owned.signals:
        _write(out / f"{episode}.exit.json", {**receipt, "exit": code, "interrupted": True, "signals": owned.signals})
        owned.child = None
        raise DispatchStopped(owned.signals)
    _write(out / f"{episode}.exit.json", {**receipt, "exit": code})
    owned.child = None


def _position_dir(root: Path, arm: str, episode: str) -> Path:
    """Where a position's run_stream records live: C and F nest their flight under the position's process folder."""
    base = root / "arms" / arm / episode
    return base / "flight" / episode if arm in ("C", "C-full", "F") else base


def _fly(root: Path, study: Path, arm: str, body: dict, rows: list[dict], profile: dict, adviser_provider: Any,
         max_steps: int | None, owned: _Owned, payload: str | None = None) -> None:
    out = root / "arms" / arm
    stream, label = body["stream"], f"study/{arm}/{body['stream']}"
    stand_in = profile is not None and profile["stand_in"] is True
    if arm == "L":
        g = _guards()
        phases = body["phases"]
        snapshot = ((streams.episode_id(stream, *FORK), root / "inputs" / "F-snapshot")
                    if 2 in phases and 3 in phases else None)
        learn = None if stand_in else profile["learn"]
        (root / "inputs").mkdir(exist_ok=True)
        run_stream.run_stream(PER_PHASE, 1, out, stream=label, stream_seed=stream, layout=LAYOUT,
                              prepared_scenarios=tuple(_scenario(r["scenario"]) for r in rows),
                              local=None if stand_in else profile["local"], learn=learn, learning=True,
                              empty_library=True, trusted_heading_guard=g["heading"], trusted_report_guard=g["report"],
                              learn_examples=learn is not None,
                              learn_candidates=profile["learn_candidates"] if learn else 1,
                              learn_regression=profile["learn_regression"] if learn else 0,
                              floor_after_cold_start=learn is not None, no_pause=False, snapshot_after=snapshot,
                              timing=True, record_inputs=True, max_steps=max_steps,
                              pause_authority=lambda seen, pause_rows: pause_authority(study, "L", stream, pause_rows),
                              **_flight(profile))
        return
    if arm == "F":
        verify_snapshot(root, stream)
    out.mkdir(parents=True, exist_ok=False)
    if arm in ("C", "C-full", "F"):
        for row in rows:
            if owned.signals:
                raise DispatchStopped(owned.signals)
            if stop_record(study, arm) is not None:
                break
            _run_position(owned, out, row["episode"], _position_argv(root, arm, row["episode"], max_steps, payload))
            pause_authority(study, arm, stream, _stream_rows_for(root, arm, rows))
    elif arm == "adviser":                       # in this process; it owns no position worker
        for row in rows:
            if stop_record(study, arm) is not None:
                break
            adviser.run_adviser_episode(_scenario(row["scenario"]), out, label, adviser_provider, max_steps=max_steps)
            pause_authority(study, arm, stream, _stream_rows_for(root, arm, rows))
    else:
        for row in rows:
            s = _scenario(row["scenario"])
            comparators.run_pilot_episode(s, comparators.conditions()[arm], out / s.episode_id, label,
                                          max_steps=max_steps)


def _stream_rows_for(root: Path, arm: str, rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        ledger = _position_dir(root, arm, row["episode"]) / "ledger.jsonl"
        if ledger.is_file():
            out += [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
    return out


def _f_copy_matches(root: Path, episode: str) -> bool:
    """Whether this F position's copied library is the manifest of the snapshot it recorded before flying."""
    folder = root / "arms" / "F" / episode
    try:
        source = json.loads((folder / "f-source.json").read_text())
        copied = json.loads((folder / "flight" / "library-from.json").read_text())
    except (OSError, ValueError):
        return False
    return copied.get("files") == source.get("files")


def receipts(root: Path, study: Path, arm: str, rows: list[dict], returned: bool, stopped_by_signal: bool = False) -> list[dict]:
    """Each scheduled position's state, its summary hash, and for Telohelm whether the flown scenario's recorded
    fingerprint equals the schedule's."""
    stopped = stop_record(study, arm) is not None
    flown_inputs = {}
    for path in (root / "arms" / arm).rglob("inputs.jsonl"):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            flown_inputs.setdefault(r["episode"], set()).add(r["scenario_sha256"])
    result = []
    for row in rows:
        path = _position_dir(root, arm, row["episode"]) / "summary.json"
        item = {"episode": row["episode"]}
        if path.is_file():
            r = json.loads(path.read_text())
            item.update(summary=str(path.relative_to(root)),
                        summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        status="policy_failed" if r.get("status") == "policy_failed" else
                        ("complete" if r.get("complete") is True else "censored"))
        else:
            item["status"] = ("paused" if stopped and returned else
                              "unattempted" if returned or stopped_by_signal else "unresolved")
        exit_file = root / "arms" / arm / f"{row['episode']}.exit.json"
        if exit_file.is_file():
            exit_record = json.loads(exit_file.read_text())
            item["process_exit"] = exit_record["exit"]
            if exit_record.get("interrupted"):
                item.update(status="interrupted", interrupted=True)
        if arm in TELOHELM and "summary" in item:
            item["input_matches_schedule"] = flown_inputs.get(row["episode"]) == {row["scenario_sha256"]}
        if arm == "F" and "summary" in item:
            item["copy_matches_snapshot"] = _f_copy_matches(root, row["episode"])
        result.append(item)
    return result


def run_arm(root: Path, arm: str, *, profile: dict | None = None, adviser_provider: Any = None,
            max_steps: int | None = None, concurrent: bool = False, payload: str | None = None) -> dict:
    """Fly one condition on one stream's prepared schedule. Serially, the parent dispatch lock is held for the whole
    run and a second dispatch is refused; with concurrent, only setup and close-out hold it, so conditions can share
    one pair of model servers."""
    root = Path(root)
    if concurrent:
        return _run_arm(root, arm, profile, adviser_provider, max_steps, lambda: _parent_lock(root, wait=True), payload)
    with _parent_lock(root, wait=False):
        return _run_arm(root, arm, profile, adviser_provider, max_steps, nullcontext, payload)


def _run_arm(root: Path, arm: str, profile: dict | None, adviser_provider: Any, max_steps: int | None, lock,
             payload: str | None = None) -> dict:
    """Fly one condition. For C and F, SIGTERM and SIGINT are only recorded (and passed to the position worker) from
    setup to the final receipts, so a stop at any point still leaves a complete stopped state."""
    owned = _Owned()
    with _handling(owned) if arm in ("C", "C-full", "F") else nullcontext(owned):
        return _run_arm_held(root, arm, profile, adviser_provider, max_steps, lock, owned, payload)


def _run_arm_held(root: Path, arm: str, profile: dict | None, adviser_provider: Any, max_steps: int | None, lock,
                  owned: _Owned, payload: str | None = None) -> dict:
    if arm not in ARMS:
        raise ValueError("unknown condition")
    began_epoch, began = time.time(), time.monotonic()
    with lock():
        study, declared = _study(root)
        envelope = load(root)
        body = envelope["schedule"]
        for s in declared["streams"]:              # every declared stream is verified before anything flies
            if str(s) != root.name:
                load(study / str(s))
        if arm in PAUSED:
            pause_anchor(study, arm)
        rows = _rows_for(arm, body["rows"])
        record = None
        if arm in TELOHELM:
            if profile is None:
                raise ValueError("Telohelm conditions need the shared flight profile")
            record = _profile_record(profile)
        elif profile is not None:
            raise ValueError("the flight profile applies to Telohelm conditions only")
        _admit(body, record, max_steps)
        if record is not None:
            prior = root / "flight-profile.json"
            if prior.exists() and json.loads(prior.read_text()) != record:
                raise ValueError("Telohelm conditions must share one flight profile")
            _write(prior, record)
        if arm == "adviser" and adviser_provider is None:
            raise ValueError("the adviser needs an explicit provider")
        state_path = root / f"{arm}-state.json"
        with state_path.open("x") as f:
            f.write(json.dumps({"status": "running", "schedule_sha256": envelope["sha256"], "worker_pid": os.getpid(),
                                "payload": payload,
                                "positions": [{"episode": r["episode"], "status": "scheduled"} for r in rows]}))
    error, returned, signalled = None, False, False
    try:
        _fly(root, study, arm, body, rows, profile, adviser_provider, max_steps, owned, payload)
        returned = True
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        signalled = isinstance(exc, DispatchStopped)
        raise
    finally:
        with lock():
            positions = receipts(root, study, arm, rows, returned, signalled)
            state = {"status": "returned" if returned else "stopped" if signalled else "failed",
                     "schedule_sha256": envelope["sha256"],
                     "positions": positions, "error": error, "worker_pid": os.getpid(),
                     "execution_started_epoch_s": began_epoch, "execution_finished_epoch_s": time.time(),
                     "execution_wall_s": time.monotonic() - began, "stop": stop_record(study, arm),
                     "signals": list(owned.signals), "payload": payload}
            if arm == "adviser":
                state["counterfactual_stop"] = counterfactual_stop_record(study, arm)
            if any(p.get("input_matches_schedule") is False or p.get("copy_matches_snapshot") is False
                   for p in positions):
                state["status"] = "input_mismatch"
            if not signalled and any(p.get("process_exit") not in (None, 0) for p in positions):
                state["status"] = "position_failed"
            _write(state_path, state)
    return state


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    dec = sub.add_parser("declare")
    dec.add_argument("--study", type=Path, required=True)
    dec.add_argument("--streams", type=int, nargs="+", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True, help="STUDY/STREAM")
    prep.add_argument("--stream", type=int, required=True)
    prep.add_argument("--c-seed", type=int, required=True)
    prep.add_argument("--adviser-seed", type=int, required=True)
    prep.add_argument("--phases", type=int, nargs="+", default=list(PHASES))
    prep.add_argument("--freeze-record", type=Path, default=None, help="scored streams only: the frozen protocol")
    run = sub.add_parser("run")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--profile", type=Path, help="JSON: the shared Telohelm flight profile")
    run.add_argument("--adviser-url")
    run.add_argument("--adviser-model", default="Qwen/Qwen3.8-27B")
    run.add_argument("--max-steps", type=int)
    run.add_argument("--concurrent", action="store_true")
    run.add_argument("--payload", help="the payload root; every position worker then starts isolated from it")
    one = sub.add_parser("fly-one")
    one.add_argument("--root", type=Path, required=True)
    one.add_argument("--arm", choices=("C", "C-full", "F"), required=True)
    one.add_argument("--episode", required=True)
    one.add_argument("--max-steps", type=int)
    a = p.parse_args(argv)
    if a.command == "declare":
        print(json.dumps(declare_study(a.study, a.streams)))
        return
    if a.command == "prepare":
        freeze = json.loads(a.freeze_record.read_text()) if a.freeze_record else None
        env = prepare(a.out, a.stream, c_seed=a.c_seed, adviser_seed=a.adviser_seed, phases=tuple(a.phases),
                      freeze=freeze)
        print(json.dumps({"schedule_sha256": env["sha256"], "note": "inputs prepared; nothing flew"}))
        return
    if a.command == "fly-one":
        raise SystemExit(_fly_one(a.root, a.arm, a.episode, a.max_steps))
    provider = None
    if a.arm == "adviser":
        if not a.adviser_url:
            p.error("the adviser needs --adviser-url")
        import telohelm.local_model as local_model
        provider = local_model.provider(a.adviser_url, a.adviser_model, adviser.TIMEOUT_S, role="reasoning_model")
    profile = json.loads(a.profile.read_text()) if a.profile else None
    try:
        state = run_arm(a.root, a.arm, profile=profile, adviser_provider=provider, max_steps=a.max_steps,
                        concurrent=a.concurrent, payload=a.payload)
    except DispatchStopped as exc:
        print(json.dumps({"arm": a.arm, "status": "stopped", "signals": exc.signals}))
        raise SystemExit(128 + exc.signals[0])
    print(json.dumps({"arm": a.arm, "status": state["status"], "stop": state["stop"]}))


if __name__ == "__main__":
    main()
