"""Shared-serving study host: flies a declared, prepared study's conditions side by side inside one allocation.

Every condition of every declared stream gets its own dispatcher process (study_runner run --concurrent), so C and F
keep their recording signal handler on their own main thread and each owns its position workers. All of them read the
one profile file the host writes, and all share the allocation's model servers through the slots the launcher sets in
the environment. F starts once L's library snapshot after p2-e19 exists, and is skipped, with the reason kept, if L ends
without one.

Startup checks everything before anything starts: every declared stream loads with its receipts and phase 5 joins, every
pause anchor is complete, an existing flight profile matches, scored admission passes, and no condition has run already.

The host itself is one thread. SIGTERM and SIGINT only set a plain flag (no call, no lock), and the loop looks at it
between steps; every dispatcher is started through spawn(), so it is on record in host.jsonl before it can act. The
condition-wide pause needs nothing from the host: each dispatcher stops itself at the lasting stop record, and the host
logs that it appeared. A dispatcher that fails is recorded and the others go on.

The host holds an exclusive lock on the study for its whole run, so no other host flies the same study. On a stop
request, at the deadline or on any error in the host, every dispatcher group it started gets SIGTERM, then SIGKILL after
STOP_GRACE_S; at the end of every run, stopped or not, every such group is closed, so descendants left by a dispatcher
that has already exited are ended too. Signals are sent before anything is logged, and a failure to log never stops the
cleanup. A C or F dispatcher passes the stop to its position worker and records it; L and the comparators stop at once.
A position worker left by a killed dispatcher is then ended only if it is on record as that dispatcher's worker (the
dispatcher's worker record or the worker's own process record) and its live process group and command line still match.

The final receipts take each dispatcher's own state when it wrote a readable one, and otherwise rebuild its positions
from the records, marking unfinished ones cut off. A damaged record (a truncated line, unreadable JSON) is left in
place, listed with its digest, read only up to its last whole line, and gives an explicit damaged or unknown position;
nothing damaged counts as complete. A failure while rebuilding one dispatcher's receipt is kept in that receipt alone.
The host span is GPUs times the host's own wall time. It is not the whole allocation, which also covers the launcher's
preparation, server start and teardown; the launcher reconciles that once with the scheduler. The dispatchers' own spans
overlap and are not GPU-hours.

    python -m paper2.experiment2.study_host run --study STUDY --profile PROFILE.json --out HOST --deadline-epoch T
        [--arms L C F calculator adaptive_controller adviser] [--adviser-url U] [--max-steps N] [--gpus N]
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from . import study_runner as S
from .comparison import _sha, _write
from .spawn import spawn

STOP_GRACE_S = 60.0          # a dispatcher's time to stop after SIGTERM; longer than study_runner.GRACE_S
NAP_S = 0.5
DEFAULT_ARMS = ("L", "C", "F", "calculator", "adaptive_controller")


def plan(study: Path, arms: tuple[str, ...], profile: dict | None, max_steps: int | None,
         streams: tuple[int, ...] | None = None) -> dict:
    """Everything checked before anything starts. Returns the dispatchers to start, with each one's schedule rows.
    streams, if given, flies only those declared streams, so the scored streams can run one job at a time under the
    budget rule (protocol amendment 7, item 4); the pause anchor still spans every declared stream."""
    study = Path(study).resolve()
    if any(a not in S.ARMS for a in arms) or len(set(arms)) != len(arms):
        raise ValueError("unknown or repeated condition")
    telohelm = [a for a in arms if a in S.TELOHELM]
    if telohelm and profile is None:
        raise ValueError("Telohelm conditions need the shared flight profile")
    record = S._profile_record(profile) if telohelm else None
    declared = json.loads((study / "study.json").read_text())
    selected = list(declared["streams"]) if streams is None else list(streams)
    if not selected or len(set(selected)) != len(selected) or any(s not in declared["streams"] for s in selected):
        raise ValueError("selected streams must be distinct streams the study declares")
    dispatch = []
    for s in selected:
        root = study / str(s)
        body = S.load(root)["schedule"]
        S._admit(body, record, max_steps)
        prior = root / "flight-profile.json"
        if record is not None and prior.exists() and json.loads(prior.read_text()) != record:
            raise ValueError(f"stream {s} already has a different flight profile")
        for arm in arms:
            rows = S._rows_for(arm, body["rows"])
            if not rows:
                continue                              # F has no positions without phase 3
            if (root / f"{arm}-state.json").exists():
                raise ValueError(f"{arm} has already run on stream {s}")
            dispatch.append({"name": f"{arm}:{s}", "arm": arm, "stream": s, "root": str(root), "rows": rows})
    for arm in arms:
        if arm in S.PAUSED:
            S.pause_anchor(study, arm)
    return {"study": str(study), "streams": selected, "declared_streams": declared["streams"], "arms": list(arms),
            "dispatch": dispatch,
            "profile_sha256": None if record is None else _sha(record)}


def _process(pid: int) -> tuple[int, list[str]] | None:
    """The process group and command line of pid, or None if it is not running (a zombie counts as not running)."""
    done = subprocess.run(["ps", "-o", "pgid=,stat=,args=", "-p", str(pid)], capture_output=True, text=True)
    parts = done.stdout.split(None, 2)
    if done.returncode != 0 or len(parts) < 2 or "Z" in parts[1]:
        return None
    return int(parts[0]), (parts[2] if len(parts) > 2 else "").split()


class ObservationFailed(Exception):
    """The process listing failed or was incomplete, so nothing can be concluded from it."""


RELEASE_WAIT_S = 5.0         # how long collecting an exited or killed dispatcher's status may take


def _table() -> dict[int, tuple[int, str]]:
    """Every process's group and state, from one listing. A failed, unreadable or incomplete listing raises
    ObservationFailed; a complete one always includes this process."""
    try:
        done = subprocess.run(["ps", "-A", "-o", "pid=,pgid=,stat="], capture_output=True, text=True, timeout=30)
        table = {int(p[0]): (int(p[1]), p[2]) for p in (line.split() for line in done.stdout.splitlines())
                 if len(p) >= 3}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise ObservationFailed(f"ps: {type(exc).__name__}: {exc}") from None
    if done.returncode != 0 or os.getpid() not in table:
        raise ObservationFailed(f"ps exited {done.returncode} with {len(table)} rows")
    return table


def _group_members(pgid: int, table: dict | None = None) -> list[int]:
    """The running (not zombie) processes in a process group; raises ObservationFailed if they cannot be seen."""
    table = _table() if table is None else table
    return sorted(q for q, (g, stat) in table.items() if g == pgid and "Z" not in stat)


def _runs(tokens: list[str], want: list) -> bool:
    return any(tokens[i:i + len(want)] == want for i in range(len(tokens) - len(want) + 1))


def _carries(pid: int, token: str) -> bool:
    """Whether a running process has this worker's marker in its environment (/proc on Linux, ps elsewhere)."""
    marker = f"{S.WORKER_TOKEN}={token}"
    environ = Path("/proc") / str(pid) / "environ"
    if environ.parent.exists():
        try:
            return marker in environ.read_bytes().decode(errors="replace").split("\0")
        except OSError:
            return False
    shown = subprocess.run(["ps", "eww", "-o", "command=", "-p", str(pid)], capture_output=True, text=True).stdout
    return marker in shown.split()


def owned_workers(dispatchers: list[dict]) -> list[dict]:
    """Position worker groups left by these dispatchers, each with the action the host takes: 'kill' or 'report'.

    A record counts only if it names one of these dispatchers (the dispatcher's worker record, or the worker's own
    process record) and says the worker led its own group. A worker whose exit record exists was closed and reaped by
    its dispatcher, so its record is left out: any group with that number now belongs to a later lifetime. For the
    rest, a running leader must have the recorded group, this module's fly-one command for that root, condition and
    episode, and, when recorded, the worker's marker. A group whose leader has gone is killed only if a running member
    carries the worker's marker; otherwise ownership cannot be shown, and the group is reported and left alone."""
    found, seen = [], set()
    table = _table()
    for d in dispatchers:
        folder = Path(d["root"]) / "arms" / d["arm"]
        records = {}
        for path in list(folder.glob("*.worker.json")) + list(folder.glob("*/process.json")):
            try:
                r = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if r.get("dispatcher_pid", r.get("ppid")) == d["pid"] and isinstance(r.get("pid"), int):
                kept = records.setdefault(r["pid"], {})
                kept.update({k: v for k, v in r.items() if v is not None})
        for pid, r in records.items():
            episode = r.get("episode")
            if pid in seen or r.get("pgid", pid) != pid or (folder / f"{episode}.exit.json").exists():
                continue
            seen.add(pid)
            entry = {"pid": pid, "pgid": pid, "dispatcher": d["name"], "episode": episode}
            live = _process(pid) if pid in table and "Z" not in table[pid][1] else None
            if live is not None:
                want = ["paper2.experiment2.study_runner", "fly-one", "--root", d["root"], "--arm", d["arm"],
                        "--episode", episode]
                if live[0] != pid or not _runs(live[1], want):
                    continue                      # a different process now has this number
                marked = "token" not in r or _carries(pid, r["token"])
                found.append({**entry, "leader": "running", "action": "kill" if marked else "report",
                              **({} if marked else {"reason": "running leader without the worker's marker"})})
                continue
            members = [m for m in _group_members(pid, table) if m != pid]
            if not members:
                continue
            marked = [m for m in members if "token" in r and _carries(m, r["token"])]
            found.append({**entry, "leader": "exited", "members": members,
                          "action": "kill" if marked else "report",
                          **({"marked": marked} if marked else {"reason": "no member carries the worker's marker"})})
    return found


def _gone(pid: int, within: float) -> bool:
    end = time.monotonic() + within
    while time.monotonic() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:                     # (macOS) a zombie, already dead
            return True
        time.sleep(0.05)
    return False


def _boundary(timing: list[dict], episode: str) -> str:
    """L's learning boundary after an episode: closed, open (started, never closed) or none."""
    events = [r.get("event") for r in timing if r.get("episode") == episode and r.get("part") == "after_episode"]
    if not events:
        return "none"
    return "closed" if events.count("end") >= events.count("start") else "open"


def _damage(path: Path, **detail) -> dict:
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        digest = None
    return {"path": str(path), "sha256": digest, **detail}


def _jsonl(path: Path, damaged: list) -> list[dict]:
    """The rows of a JSON-lines record up to its last whole, readable line; a damaged one is listed and left as is."""
    if not path.is_file():
        return []
    rows = []
    for n, line in enumerate(path.read_bytes().split(b"\n")):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            damaged.append(_damage(path, valid_lines=len(rows), bad_line=n))
            break
    return rows


def _json(path: Path, damaged: list):
    """A JSON record, or None if absent; a damaged one is listed, left as is, and returned as the DAMAGED marker."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        damaged.append(_damage(path, error=type(exc).__name__))
        return DAMAGED


DAMAGED = object()


def _summary_status(summary: dict) -> str:
    return ("policy_failed" if summary.get("status") == "policy_failed" else
            "complete" if summary.get("complete") is True else "censored")


class Host:
    def __init__(self, a):
        self.a = a
        self.study = Path(a.study).resolve()
        self.out = Path(a.out).resolve()
        self.arms = tuple(a.arms)
        self.profile = json.loads(Path(a.profile).read_text()) if a.profile else None
        self.deadline = a.deadline_epoch
        self.procs: dict[str, subprocess.Popen] = {}
        self.ended: dict[str, int] = {}
        self.skips: dict[str, str] = {}
        self.pauses: set[str] = set()
        self.note_failures: list[str] = []
        self.released: set[str] = set()
        self.held: set[str] = set()
        self.observation_failures: list[str] = []
        self.release_failures: list[str] = []
        self.stop_flag, self.stop_reason = False, None

    def on_signal(self, signum, frame):
        """Set the stop flag and nothing else: two plain assignments, no call and no lock."""
        if not self.stop_flag:
            self.stop_reason = f"signal {signum}"
            self.stop_flag = True

    def nap(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while not self.stop_flag and time.monotonic() < end:
            time.sleep(max(0.0, min(0.1, end - time.monotonic())))

    def note(self, **row) -> None:
        """Log one event. Logging never stops the host: a failure is kept in memory and in the final receipt if that
        can still be written."""
        try:
            with open(self.out / "host.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.time(), **row}, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as exc:
            self.note_failures.append(f"{row.get('event')}: {type(exc).__name__}: {exc}")

    def argv(self, d: dict) -> list[str]:
        args = ["run", "--root", d["root"], "--arm", d["arm"], "--concurrent"]
        if d["arm"] in S.TELOHELM:
            args += ["--profile", str(self.out / "profile.json")]
        if d["arm"] == "adviser":
            args += ["--adviser-url", self.a.adviser_url, "--adviser-model", self.a.adviser_model]
        if self.a.max_steps:
            args += ["--max-steps", str(self.a.max_steps)]
        if self.a.payload:
            args += ["--payload", str(Path(self.a.payload).resolve())]
        return S.module_command(self.a.python, "paper2.experiment2.study_runner", args, self.a.payload)

    def launch(self, d: dict) -> None:
        argv = self.argv(d)
        env = dict(os.environ, TELOHELM_ARM=d['arm'], TELOHELM_STREAM=str(d['stream']))

        def register(p):
            self.procs[d["name"]] = p
            d["pid"] = p.pid
            self.note(event="start", name=d["name"], arm=d["arm"], stream=d["stream"],
                      pid=p.pid, pgid=p.pid, argv=argv)
        with open(self.out / "logs" / f"{d['arm']}-{d['stream']}.log", "x") as log:
            spawn(argv, register, stdout=log, stderr=subprocess.STDOUT, env=env)

    # A dispatcher's exit status is collected only once its group is empty. Until then its number stays its own, even
    # after it exits (it shows as a zombie), so no other group can take that number, and every member of the group is
    # this host's to signal. Once collected, the number is released, and the host never signals it again.
    def observe(self) -> dict | None:
        """The process table, or None if it cannot be read; each failure is kept for the receipt."""
        try:
            return _table()
        except ObservationFailed as exc:
            self.observation_failures.append(f"{time.time():.3f} {exc}")
            if len(self.observation_failures) % 50 == 1:          # the first, then every fiftieth
                self.note(event="observation_failed", error=str(exc), count=len(self.observation_failures))
            return None

    def exited(self, name: str, table: dict | None) -> bool:
        """Whether a dispatcher has exited. Without a valid observation that is unknown, so it is not treated as
        exited."""
        p = self.procs[name]
        if name in self.released:
            return True
        return table is not None and (p.pid not in table or "Z" in table[p.pid][1])

    def members(self, name: str, table: dict | None) -> list[int] | None:
        """Running members of a dispatcher's group other than itself, or None if they cannot be seen."""
        if table is None:
            return None
        pid = self.procs[name].pid
        return sorted(q for q, (g, stat) in table.items() if g == pid and q != pid and "Z" not in stat)

    def release(self, name: str) -> bool:
        """Collect a dispatcher's exit status, within RELEASE_WAIT_S. A dispatcher that does not finish in that time
        stays held, and the failure is kept for the receipt."""
        p = self.procs[name]
        try:
            p.wait(timeout=RELEASE_WAIT_S)
        except subprocess.TimeoutExpired:
            self.release_failures.append(f"{name}: not collected within {RELEASE_WAIT_S} s")
            self.note(event="release_failed", name=name)
            return False
        self.released.add(name)
        self.held.discard(name)
        self.ended[name] = p.returncode
        self.note(event="end", name=name, exit=p.returncode)
        return True

    def reap(self) -> None:
        """Release each exited dispatcher whose group is empty; keep holding one whose group still has members. With
        no valid observation, nothing is released."""
        table = self.observe()
        if table is None:
            return
        for name in self.procs:
            if name in self.released or not self.exited(name, table):
                continue
            if self.members(name, table):
                if name not in self.held:
                    self.held.add(name)
                    self.note(event="held", name=name, members=self.members(name, table))
            else:
                self.release(name)

    def watch_pauses(self) -> None:
        for arm in self.arms:
            if arm not in self.pauses and S.stop_record(self.study, arm) is not None:
                self.pauses.add(arm)
                self.note(event="pause", arm=arm, stop=S.stop_record(self.study, arm))

    def ready(self, d: dict) -> str | None:
        """'go', 'wait' or a skip reason. F waits for L's snapshot, which must be this stream's p2-e19 boundary with a
        library matching its manifest; once L has ended without a valid one, F is skipped with the reason."""
        if d["arm"] != "F":
            return "go"
        snapshot = Path(d["root"]) / "inputs" / "F-snapshot" / "snapshot.json"
        if not snapshot.is_file():
            name = f"L:{d['stream']}"
            l_planned = any(x["name"] == name for x in self.plan["dispatch"])
            if l_planned and (name not in self.procs or not self.exited(name, self.observe())):
                return "wait"
            if not snapshot.is_file():
                return "no snapshot from L"
        try:
            S.verify_snapshot(Path(d["root"]), d["stream"])
        except ValueError as exc:
            return f"invalid snapshot: {exc}"
        return "go"

    def close_groups(self, reason: str, stopping: bool) -> dict:
        """End every dispatcher group the host still holds, whether or not its leader has exited: SIGTERM, then SIGKILL
        after STOP_GRACE_S, then release every dispatcher within RELEASE_WAIT_S. A dispatcher not yet released is this
        host's unreaped child, so its group is signalled even when the process table cannot be read; a released
        dispatcher's number is never signalled. Every step is bounded, and signals go out before anything is logged."""
        table = self.observe()
        unreleased = sorted(n for n in self.procs if n not in self.released)

        def occupied(n, table):
            if table is None:
                return True                       # cannot be seen: assume it may still be running
            return not self.exited(n, table) or bool(self.members(n, table))
        live = [n for n in unreleased if occupied(n, table)]
        for n in live:
            S._signal_group(self.procs[n].pid, signal.SIGTERM)
        if stopping or live:
            self.note(event="stopping" if stopping else "closing", reason=reason, live=live)
        end = time.monotonic() + STOP_GRACE_S
        while time.monotonic() < end:
            table = self.observe()
            if table is not None and not any(occupied(n, table) for n in live):
                break
            time.sleep(0.25)
        table = self.observe()
        killed = [n for n in live if occupied(n, table)]
        for n in killed:
            S._signal_group(self.procs[n].pid, signal.SIGKILL)
        end = time.monotonic() + 5.0
        while killed and time.monotonic() < end:
            table = self.observe()
            if table is not None and not any(occupied(n, table) for n in killed):
                break
            time.sleep(0.05)
        for n in unreleased:
            self.release(n)
        if live:
            self.note(event="groups_closed", signalled=live, killed=killed)
        return {"signalled": live, "killed": killed, "unreleased": sorted(n for n in self.procs
                                                                          if n not in self.released)}

    def sweep(self) -> list[dict]:
        """End position workers left by this host's C and F dispatchers, by their records and live identity only."""
        dispatchers = [d for d in self.plan["dispatch"] if d["arm"] in ("C", "C-full", "F") and "pid" in d]
        try:
            left = owned_workers(dispatchers)
        except ObservationFailed as exc:
            left = [{"action": "report", "reason": f"observation failed: {exc}", "dispatchers": len(dispatchers)}]
        for o in left:
            if o["action"] == "kill":
                S._signal_group(o["pgid"], signal.SIGKILL)
                o["gone"] = all(_gone(pid, 5.0) for pid in [o["pid"], *o.get("members", [])])
        if left:
            self.note(event="swept", workers=left)
        return left

    def dispatch_receipt(self, d: dict, stopping: bool) -> dict:
        """One dispatcher's receipt. A failure while building it is kept here, with every position unknown, and does
        not stop the other receipts."""
        try:
            return self._dispatch_receipt(d, stopping)
        except Exception as exc:
            return {"name": d["name"], "arm": d["arm"], "stream": d["stream"], "launched": d["name"] in self.procs,
                    "exit": self.ended.get(d["name"]), "dispatch_status": "receipt_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "positions": [{"episode": r["episode"], "status": "unknown"} for r in d["rows"]],
                    "counts": {"unknown": len(d["rows"])}}

    def _dispatch_receipt(self, d: dict, stopping: bool) -> dict:
        root, arm, name = Path(d["root"]), d["arm"], d["name"]
        damaged: list = []
        item = {"name": name, "arm": arm, "stream": d["stream"], "launched": name in self.procs,
                "exit": self.ended.get(name), "skip": self.skips.get(name)}
        state = _json(root / f"{arm}-state.json", damaged)
        if isinstance(state, dict) and state.get("status") not in (None, "running") and \
                isinstance(state.get("positions"), list):
            positions, item["receipts_from"] = state["positions"], "dispatcher"
            item["state"], item["wall_s"] = state["status"], state.get("execution_wall_s")
            item["stop"] = state.get("stop")
        else:
            positions, item["receipts_from"] = self.rebuild(root, arm, d["rows"], stopping, damaged), "records"
            item["state"] = ("damaged" if state is DAMAGED else None if state is None
                             else state.get("status") if isinstance(state, dict) else "damaged")
        if arm == "L":
            timing = _jsonl(root / "arms" / "L" / "timing.jsonl", damaged)
            learning = self.profile is not None and self.profile.get("stand_in") is False
            for p in positions:
                p["boundary"] = _boundary(timing, p["episode"])
                if p["status"] == "complete" and (p["boundary"] != "closed" if learning else p["boundary"] == "open"):
                    p["status"] = "boundary_open"
        item["positions"] = positions
        item["counts"] = dict(Counter(p["status"] for p in positions))
        item["damaged"] = damaged
        if not item["launched"]:
            item["dispatch_status"] = "skipped" if item["skip"] else "not_started"
        elif item["receipts_from"] == "records":
            item["dispatch_status"] = "cut_off" if stopping else "failed"
        elif item["state"] == "returned" and item.get("stop"):
            item["dispatch_status"] = "paused"
        else:
            item["dispatch_status"] = item["state"]
        return item

    def rebuild(self, root: Path, arm: str, rows: list[dict], stopping: bool, damaged: list) -> list[dict]:
        """Positions from the records alone, for a dispatcher that left no readable final state. A damaged summary
        gives a damaged position; one that never started is not started; one cut short is cut off or failed."""
        inputs: dict = {}
        if arm in S.TELOHELM:
            for path in sorted((root / "arms" / arm).rglob("inputs.jsonl")):
                for r in _jsonl(path, damaged):
                    inputs.setdefault(r.get("episode"), set()).add(r.get("scenario_sha256"))
        out = []
        for row in rows:
            ep = row["episode"]
            item = {"episode": ep}
            summary_path = S._position_dir(root, arm, ep) / "summary.json"
            summary = _json(summary_path, damaged)
            exit_record = _json(root / "arms" / arm / f"{ep}.exit.json", damaged) if arm in ("C", "C-full", "F") else None
            if exit_record is DAMAGED:
                item["status"] = "damaged"
            elif isinstance(exit_record, dict) and exit_record.get("interrupted"):
                # As in the runner's own receipt, an interrupted worker overrides a summary it may already have saved;
                # the summary's status is kept as evidence.
                item.update(status="interrupted", interrupted=True)
                if summary is DAMAGED or isinstance(summary, dict):
                    item["summary_status"] = "damaged" if summary is DAMAGED else _summary_status(summary)
            elif summary is DAMAGED:
                item["status"] = "damaged"
            elif isinstance(summary, dict):
                item["status"] = _summary_status(summary)
                if arm in S.TELOHELM:
                    item["input_matches_schedule"] = inputs.get(ep) == {row["scenario_sha256"]}
                if arm == "F":
                    item["copy_matches_snapshot"] = S._f_copy_matches(root, ep)
            elif (root / "arms" / arm / ep).exists():
                item["status"] = "cut_off" if stopping else "failed"
            else:
                item["status"] = "not_started"
            if isinstance(exit_record, dict):
                item["process_exit"] = exit_record.get("exit")
            if isinstance(summary, dict):
                # The analysis reads a cut-off condition through this receipt, so it binds the summary it saw, as the
                # runner's own receipt does.
                item.update(summary=str(summary_path.relative_to(root)),
                            summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest())
            out.append(item)
        return out

    def run(self) -> int:
        self.out.mkdir(parents=True, exist_ok=False)
        (self.out / "logs").mkdir()
        started = time.time()
        gpus = self.a.gpus
        self.note(event="host_start", study=str(self.study), arms=list(self.arms), deadline=self.deadline,
                  gpus=gpus, job=os.environ.get("SLURM_JOB_ID"), pid=os.getpid())
        lock = open(self.study / ".host.lock", "a")
        try:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("another host is flying this study") from None
            self.plan = plan(self.study, self.arms, self.profile, self.a.max_steps,
                             tuple(self.a.streams) if self.a.streams else None)
        except Exception as exc:
            self.note(event="refused", error=f"{type(exc).__name__}: {exc}")
            lock.close()
            raise
        try:
            return self._fly(started, gpus)
        finally:
            lock.close()

    def _fly(self, started: float, gpus: int | None) -> int:
        if self.profile is not None:
            _write(self.out / "profile.json", self.profile)
        _write(self.out / "plan.json", {k: v for k, v in self.plan.items() if k != "dispatch"}
               | {"dispatch": [{k: v for k, v in d.items() if k != "rows"} for d in self.plan["dispatch"]]})
        pending, reason = list(self.plan["dispatch"]), "finished"
        try:
            while True:
                if self.stop_flag:
                    reason = self.stop_reason
                    break
                if time.time() >= self.deadline:
                    reason = "deadline"
                    break
                for d in list(pending):
                    if self.stop_flag:
                        break
                    go = self.ready(d)
                    if go == "wait":
                        continue
                    pending.remove(d)
                    if go == "go":
                        self.launch(d)
                    else:
                        self.skips[d["name"]] = go
                        self.note(event="skip", name=d["name"], reason=go)
                self.reap()
                self.watch_pauses()
                table = self.observe()
                if not pending and table is not None and all(self.exited(n, table) for n in self.procs):
                    break
                self.nap(NAP_S)
        except BaseException as exc:
            reason = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            stopping = reason != "finished"
            if stopping:
                for d in pending:
                    self.skips.setdefault(d["name"], f"not started: {reason}")
            closed = self.close_groups(reason, stopping)
            self.reap()
            try:
                self.watch_pauses()
            except Exception as exc:
                self.note_failures.append(f"pause check: {type(exc).__name__}: {exc}")
            swept = self.sweep()
            finished = time.time()
            dispatch = [self.dispatch_receipt(d, stopping) for d in self.plan["dispatch"]]
            verdict = {"complete": (not stopping and not swept and not self.pauses and not closed["signalled"]
                                    and not self.observation_failures and not self.release_failures
                                    and all(x["dispatch_status"] == "returned" and x["exit"] == 0 for x in dispatch)
                                    and all(set(x["counts"]) <= {"complete"} for x in dispatch)),
                       "stopped": None if not stopping else reason, "paused": sorted(self.pauses),
                       "dispatchers_not_returned": sorted(x["name"] for x in dispatch
                                                          if x["dispatch_status"] != "returned"),
                       "positions_not_complete": sum(n for x in dispatch for s, n in x["counts"].items()
                                                     if s != "complete"),
                       "workers_swept": sum(o["action"] == "kill" for o in swept),
                       "groups_reported_not_signalled": [o for o in swept if o["action"] == "report"], "damaged_records": sum(len(x.get("damaged", [])) for x in dispatch),
                       "groups_left_running_after_dispatchers": [] if stopping else closed["signalled"]}
            host_span = {"job": os.environ.get("SLURM_JOB_ID"), "gpus": gpus, "started_epoch": started,
                         "finished_epoch": finished, "wall_s": finished - started,
                         "gpu_hours": None if gpus is None else gpus * (finished - started) / 3600,
                         "note": ("GPUs times the host's own wall time; the whole allocation, with the launcher's "
                                  "preparation, server start and teardown, is reconciled once by the launcher with the "
                                  "scheduler; the dispatchers' spans overlap and are not GPU-hours")}
            _write(self.out / "host-receipts.json", {"reason": reason, "verdict": verdict, "host_span": host_span,
                                                     "dispatch": dispatch, "swept": swept, "groups": closed,
                                                     "note_failures": self.note_failures,
                                                     "observation_failures": self.observation_failures,
                                                     "release_failures": self.release_failures})
            self.note(event="host_end", reason=reason, verdict=verdict)
        return 0 if verdict["complete"] else 3


def _default_gpus() -> int | None:
    if os.environ.get("SLURM_GPUS_ON_NODE"):
        return int(os.environ["SLURM_GPUS_ON_NODE"])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return len([x for x in visible.split(",") if x.strip()]) if visible else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    run = sub.add_parser("run")
    run.add_argument("--study", type=Path, required=True)
    run.add_argument("--profile", type=Path, help="JSON: the shared Telohelm flight profile")
    run.add_argument("--out", type=Path, required=True, help="a new folder for the host's records")
    run.add_argument("--deadline-epoch", type=float, required=True)
    run.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS), choices=S.ARMS)
    run.add_argument("--adviser-url")
    run.add_argument("--adviser-model", default="Qwen/Qwen3.8-27B")
    run.add_argument("--max-steps", type=int)
    run.add_argument("--gpus", type=int, default=None, help="GPUs in the allocation (default: from Slurm or CUDA)")
    run.add_argument("--python", default=sys.executable)
    run.add_argument("--streams", type=int, nargs="+", help="fly only these declared streams (default: all)")
    check = sub.add_parser("check", help="run the host's startup checks and print the plan; start and write nothing")
    check.add_argument("--study", type=Path, required=True)
    check.add_argument("--profile", type=Path)
    check.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS), choices=S.ARMS)
    check.add_argument("--max-steps", type=int)
    check.add_argument("--streams", type=int, nargs="+", help="check only these declared streams (default: all)")
    run.add_argument("--payload", type=Path, help="the payload root; every dispatcher and worker starts isolated from it")
    a = ap.parse_args(argv)
    if a.mode == "check":
        try:
            profile = json.loads(a.profile.read_text()) if a.profile else None
            checked = plan(a.study, tuple(a.arms), profile, a.max_steps, tuple(a.streams) if a.streams else None)
        except Exception as exc:
            print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
            return 2
        print(json.dumps({"ok": True, "study": checked["study"], "streams": checked["streams"],
                          "arms": checked["arms"], "profile_sha256": checked["profile_sha256"],
                          "dispatch": [{"name": d["name"], "positions": len(d["rows"])} for d in checked["dispatch"]]}))
        return 0
    if "adviser" in a.arms and not a.adviser_url:
        ap.error("the adviser needs --adviser-url")
    if a.gpus is None:
        a.gpus = _default_gpus()
    host = Host(a)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, host.on_signal)
    return host.run()


if __name__ == "__main__":
    raise SystemExit(main())
