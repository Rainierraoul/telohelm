"""The shared-serving study host on development fixtures only (unreserved streams 4493 and 4495, C seed 6601, adviser
seed 6401) with stand-in models: startup refusals before anything starts, normal dispatch of every condition with F
after L's snapshot, the allocation accounted once, the condition-wide pause across streams, a failed dispatcher,
cancellation by signal and by deadline, and the sweep of position workers left behind. Each run starts the real host as
its own process. Nothing here draws a scored scenario, calls a model or touches the cluster."""
import ast
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

from paper2.experiment2 import study_host as H, study_runner as S

HERE = Path(__file__).resolve().parent
STREAM, OTHER, C_SEED, ADVISER_SEED = 4493, 4495, 6601, 6401
STAND_IN = {"stand_in": True, "local": None, "learn": None, "small_interval_s": 30, "watch_briefing": True,
            "model_report_control": True, "learn_candidates": 3, "learn_regression": 8}


def _prepare(base, phases, stream_list):
    study = base / "study"
    S.declare_study(study, list(stream_list))
    for s in stream_list:
        S.prepare(study / str(s), s, c_seed=C_SEED, adviser_seed=ADVISER_SEED, phases=phases)
    return study


@pytest.fixture(scope="module")
def three_phases(tmp_path_factory):
    return _prepare(tmp_path_factory.mktemp("p123"), (1, 2, 3), [STREAM])


@pytest.fixture(scope="module")
def two_streams(tmp_path_factory):
    return _prepare(tmp_path_factory.mktemp("two"), (1,), [STREAM, OTHER])


def _copy(study, tmp_path):
    shutil.copytree(study, tmp_path / "study")
    return tmp_path / "study"


def _host(tmp_path, study, *extra, deadline_s=600, max_steps=10, python=None, wait=True):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(STAND_IN))
    argv = [sys.executable, "-B", "-m", "paper2.experiment2.study_host", "run", "--study", str(study),
            "--profile", str(profile), "--out", str(tmp_path / "host"), "--deadline-epoch", str(time.time() + deadline_s),
            "--gpus", "2", *(["--max-steps", str(max_steps)] if max_steps else []),
            *(["--python", str(python)] if python else []), *extra]
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if not wait:
        return p
    out, _ = p.communicate(timeout=900)
    return p.returncode, out


def _receipts(tmp_path):
    return json.loads((tmp_path / "host" / "host-receipts.json").read_text())


def _events(tmp_path):
    return [json.loads(l) for l in (tmp_path / "host" / "host.jsonl").read_text().splitlines()]


def _by_name(receipts):
    return {d["name"]: d for d in receipts["dispatch"]}


def test_the_spawn_helper_is_the_accepted_m1_one():
    m1 = HERE.parents[1] / "development/m1-timing-rehearsal/spawn.py"
    def body(path):
        tree = ast.parse(path.read_text())
        return [ast.dump(n) for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    assert body(m1) == body(HERE.parent / "spawn.py")


def test_startup_refuses_before_anything_starts(tmp_path, two_streams):
    study = _copy(two_streams, tmp_path)
    with pytest.raises(ValueError, match="flight profile"):
        H.plan(study, ("C",), None, 10)
    (study / str(OTHER) / "flight-profile.json").write_text(json.dumps(S._profile_record({**STAND_IN,
                                                                                         "small_interval_s": 20})))
    with pytest.raises(ValueError, match="different flight profile"):
        H.plan(study, ("L", "C"), STAND_IN, 10)
    (study / str(OTHER) / "flight-profile.json").unlink()
    (study / str(STREAM) / "calculator-state.json").write_text("{}")
    with pytest.raises(ValueError, match="already run"):
        H.plan(study, ("calculator",), None, 10)
    (study / str(STREAM) / "calculator-state.json").unlink()
    unprepared = tmp_path / "unprepared"
    S.declare_study(unprepared, [STREAM, OTHER])
    shutil.copytree(study / str(STREAM), unprepared / str(STREAM))
    code, out = _host(tmp_path, unprepared)
    events = _events(tmp_path)
    assert code != 0 and [e["event"] for e in events] == ["host_start", "refused"]
    assert not list(unprepared.rglob("*-state.json")) and not (tmp_path / "host" / "host-receipts.json").exists()


def test_every_condition_flies_with_f_after_ls_snapshot_and_the_allocation_is_accounted_once(tmp_path, three_phases):
    study = _copy(three_phases, tmp_path)
    code, out = _host(tmp_path, study)
    r = _receipts(tmp_path)
    d = _by_name(r)
    assert code == 3 and r["reason"] == "finished"          # stand-in flights stop at their step limit: censored
    assert {n: (x["dispatch_status"], x["exit"]) for n, x in d.items()} == {
        f"{a}:{STREAM}": ("returned", 0) for a in ("L", "C", "F", "calculator", "adaptive_controller")}
    assert {n: sum(x["counts"].values()) for n, x in d.items()} == {
        f"L:{STREAM}": 60, f"C:{STREAM}": 15, f"F:{STREAM}": 10, f"calculator:{STREAM}": 60,
        f"adaptive_controller:{STREAM}": 60}
    assert all(x["receipts_from"] == "dispatcher" for x in d.values())
    starts = {e["name"]: e["t"] for e in _events(tmp_path) if e["event"] == "start"}
    snapshot = study / str(STREAM) / "inputs/F-snapshot/snapshot.json"
    assert snapshot.stat().st_mtime <= starts[f"F:{STREAM}"]
    assert min(starts.values()) == starts[f"L:{STREAM}"]
    a = r["host_span"]
    assert a["gpus"] == 2 and a["gpu_hours"] == pytest.approx(2 * a["wall_s"] / 3600) and "allocation" not in r
    assert r["swept"] == [] and r["verdict"]["workers_swept"] == 0 and r["groups"]["signalled"] == []
    assert all(x["damaged"] == [] for x in d.values()) and r["note_failures"] == []
    assert all(p["copy_matches_snapshot"] is True for p in d[f"F:{STREAM}"]["positions"])
    profile = json.loads((tmp_path / "host" / "profile.json").read_text())
    assert profile == STAND_IN and json.loads((study / str(STREAM) / "flight-profile.json").read_text()) == \
        S._profile_record(STAND_IN)


def test_a_pause_in_one_stream_stops_the_condition_in_both(tmp_path, two_streams):
    study = _copy(two_streams, tmp_path)
    anchor = S.pause_anchor(study, "C")
    stop = S.pause_authority(study, "C", STREAM, [{"kind": "reasoning_model", "scope": "running", "status": "timed_out",
                                                   "episode": anchor[0], "request_id": "r1", "recovers": None}])
    code, out = _host(tmp_path, study, "--arms", "L", "C", "calculator")
    d = _by_name(_receipts(tmp_path))
    assert code == 3 and stop["due"]
    for s in (STREAM, OTHER):
        assert d[f"C:{s}"]["dispatch_status"] == "paused" and d[f"C:{s}"]["counts"] == {"paused": 5}
        assert d[f"L:{s}"]["dispatch_status"] == "returned" and d[f"calculator:{s}"]["dispatch_status"] == "returned"
    assert _receipts(tmp_path)["verdict"]["paused"] == ["C"]
    assert [e["arm"] for e in _events(tmp_path) if e["event"] == "pause"] == ["C"]


def test_a_failed_dispatcher_is_recorded_f_is_skipped_without_ls_snapshot_and_the_rest_fly(tmp_path, three_phases):
    study = _copy(three_phases, tmp_path)
    wrapper = tmp_path / "python-failing-L"
    wrapper.write_text(f'#!/bin/sh\ncase " $* " in *" --arm L "*) exit 7;; esac\nexec "{sys.executable}" "$@"\n')
    wrapper.chmod(0o755)
    code, out = _host(tmp_path, study, "--arms", "L", "C", "F", "calculator", python=wrapper)
    d = _by_name(_receipts(tmp_path))
    assert code == 3
    assert d[f"L:{STREAM}"]["exit"] == 7 and d[f"L:{STREAM}"]["dispatch_status"] == "failed"
    assert d[f"L:{STREAM}"]["receipts_from"] == "records" and d[f"L:{STREAM}"]["counts"] == {"not_started": 60}
    assert d[f"F:{STREAM}"]["dispatch_status"] == "skipped" and d[f"F:{STREAM}"]["skip"] == "no snapshot from L"
    assert d[f"C:{STREAM}"]["dispatch_status"] == "returned" and sum(d[f"C:{STREAM}"]["counts"].values()) == 15
    assert d[f"calculator:{STREAM}"]["dispatch_status"] == "returned"


def _fly_one_workers(roots):
    """Every running fly-one worker for these roots, whoever owns it: the test's broad check that none is left."""
    listing = subprocess.run(["ps", "-A", "-o", "pid=,args="], capture_output=True, text=True).stdout
    return [int(line.split(None, 1)[0]) for line in listing.splitlines()
            if any(H._runs(line.split(), ["fly-one", "--root", r]) for r in roots)]


def _gone(pid, within=5.0):
    return H._gone(pid, within)


def _wait_for(predicate, within=120.0):
    end = time.monotonic() + within
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.2)
    return False


@pytest.fixture
def server():
    """A process outside every dispatcher group, standing in for a shared model server."""
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    yield p
    p.kill()
    p.wait()


@pytest.mark.parametrize("how", ["signal", "deadline"])
def test_a_stopped_host_stops_every_dispatcher_and_leaves_no_position_worker(tmp_path, two_streams, server, how):
    study = _copy(two_streams, tmp_path)
    roots = [str((study / str(s)).resolve()) for s in (STREAM, OTHER)]
    p = _host(tmp_path, study, "--arms", "L", "C", "calculator", max_steps=None, wait=False,
              deadline_s=600 if how == "signal" else 25)
    assert _wait_for(lambda: len(_fly_one_workers(roots)) >= 2), "no position worker started"
    if how == "signal":
        p.send_signal(signal.SIGTERM)
    out, _ = p.communicate(timeout=300)
    r = _receipts(tmp_path)
    d = _by_name(r)
    assert p.returncode == 3 and r["reason"] == ("signal 15" if how == "signal" else "deadline")
    for s in (STREAM, OTHER):
        c = d[f"C:{s}"]
        assert c["dispatch_status"] == "stopped" and c["receipts_from"] == "dispatcher"
        assert c["counts"].get("interrupted") == 1 and c["exit"] == 128 + signal.SIGTERM
        assert d[f"L:{s}"]["dispatch_status"] in ("cut_off", "returned")
    assert any(d[f"L:{s}"]["dispatch_status"] == "cut_off" and d[f"L:{s}"]["receipts_from"] == "records"
               and d[f"L:{s}"]["counts"].get("cut_off") for s in (STREAM, OTHER))
    assert _fly_one_workers(roots) == [] and server.poll() is None
    assert all(e["event"] != "killed" for e in _events(tmp_path))      # every dispatcher stopped within its grace


def _sleeper(*tokens, grandchild=False):
    code = ("import subprocess, sys, time\n"
            + ("subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n" if grandchild else "")
            + "time.sleep(120)")
    return subprocess.Popen([sys.executable, "-c", code, *tokens], start_new_session=True)


def _bare_host(tmp_path, study=None):
    host = H.Host.__new__(H.Host)
    host.out, host.study, host.profile = tmp_path, study, STAND_IN
    host.procs, host.ended, host.skips, host.pauses, host.note_failures = {}, {}, {}, set(), []
    host.released, host.held = set(), set()
    host.observation_failures, host.release_failures = [], []
    return host


def _worker_tokens(root, arm, episode):
    return ["paper2.experiment2.study_runner", "fly-one", "--root", root, "--arm", arm, "--episode", episode]


def test_the_sweep_ends_only_workers_on_record_as_this_hosts(tmp_path):
    root = str(tmp_path / "4493")
    folder = Path(root) / "arms" / "C"
    folder.mkdir(parents=True)
    ours, other_host, reused, regrouped = (_sleeper(*_worker_tokens(root, "C", "e1")),
                                           _sleeper(*_worker_tokens(root, "C", "e2")),
                                           _sleeper("something", "else"),
                                           _sleeper(*_worker_tokens(root, "C", "e4")))
    procs = (ours, other_host, reused, regrouped)
    try:
        dispatcher = os.getpid()
        records = {"e1": (ours.pid, ours.pid, dispatcher), "e2": (other_host.pid, other_host.pid, dispatcher + 1),
                   "e3": (reused.pid, reused.pid, dispatcher), "e4": (regrouped.pid, regrouped.pid + 1, dispatcher)}
        for ep, (pid, pgid, owner) in records.items():
            (folder / f"{ep}.worker.json").write_text(json.dumps({"pid": pid, "pgid": pgid, "dispatcher_pid": owner,
                                                                   "episode": ep}))
        assert _wait_for(lambda: H._process(ours.pid) is not None and H._runs(H._process(ours.pid)[1],
                                                                               _worker_tokens(root, "C", "e1")), 10)
        mine = [{"name": "C:4493", "arm": "C", "root": root, "pid": dispatcher}]
        assert [o["pid"] for o in H.owned_workers(mine)] == [ours.pid]
        assert H.owned_workers([{**mine[0], "arm": "F", "name": "F:4493"}]) == []   # an F-only host
        host = _bare_host(tmp_path)
        host.plan = {"dispatch": mine}
        assert [o["pid"] for o in host.sweep()] == [ours.pid] and ours.wait(timeout=10) == -signal.SIGKILL
        assert all(p.poll() is None for p in (other_host, reused, regrouped))
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
                p.wait()


def _exited_dispatcher():
    """A dispatcher that leads its own group, starts a sleeping child there and exits, not yet collected (a zombie
    still holding its number), and its child."""
    leader = subprocess.Popen([sys.executable, "-c", "import subprocess, sys\n"
                               "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
                               "print(p.pid, flush=True)"], start_new_session=True, stdout=subprocess.PIPE, text=True)
    child = int(leader.stdout.readline())
    assert _wait_for(lambda: "Z" in H._table().get(leader.pid, (0, "Z"))[1], 10)
    return leader, child


def test_cleanup_runs_when_logging_fails_and_reaches_a_group_whose_leader_has_exited(tmp_path):
    (tmp_path / "host.jsonl").mkdir()                 # every log write now fails
    host = _bare_host(tmp_path)
    exited, left_behind = _exited_dispatcher()
    running = _sleeper(grandchild=True)
    host.procs = {"L:4493": exited, "C:4493": running}
    closed = host.close_groups("signal 15", stopping=True)
    assert sorted(closed["signalled"]) == ["C:4493", "L:4493"] and closed["killed"] == []
    assert _gone(left_behind) and running.returncode is not None and host.released == {"C:4493", "L:4493"}
    assert host.ended == {"L:4493": 0, "C:4493": -signal.SIGTERM}
    assert host.note_failures and all("IsADirectoryError" in f for f in host.note_failures)


def test_a_finished_run_still_closes_descendants_its_dispatchers_left(tmp_path):
    host = _bare_host(tmp_path)
    exited, left_behind = _exited_dispatcher()
    host.procs = {"calculator:4493": exited}
    assert host.close_groups("finished", stopping=False)["signalled"] == ["calculator:4493"]
    assert _gone(left_behind)
    events = [json.loads(l) for l in (tmp_path / "host.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["closing", "end", "groups_closed"]


def test_damaged_crash_records_give_explicit_receipts_and_keep_the_rest(tmp_path, two_streams):
    study = _copy(two_streams, tmp_path)
    root = study / str(STREAM)
    rows = S.load(root)["schedule"]["rows"]
    first, second = rows[0]["episode"], rows[1]["episode"]
    (root / "L-state.json").write_text('{"status": "running", "positions": [')
    (root / "arms/L" / first).mkdir(parents=True)
    (root / "arms/L" / first / "summary.json").write_text('{"complete": tr')
    (root / "arms/L" / second).mkdir(parents=True)
    (root / "arms/L" / second / "summary.json").write_text(json.dumps({"complete": True}))
    (root / "arms/L/timing.jsonl").write_text(json.dumps({"event": "start", "episode": second, "part": "flight"})
                                              + "\n" + '{"event": "end", "epi')
    host = _bare_host(tmp_path, study)
    names = {f"L:{STREAM}": S._rows_for("L", rows), f"C:{STREAM}": S._rows_for("C", rows)}
    host.plan = {"dispatch": [{"name": n, "arm": n.split(":")[0], "stream": STREAM, "root": str(root), "rows": r}
                              for n, r in names.items()]}
    host.procs = {f"L:{STREAM}": None}
    host.ended = {f"L:{STREAM}": -9}
    real = host._dispatch_receipt

    def broken(d, stopping):
        if d["arm"] == "C":
            raise RuntimeError("for the test")
        return real(d, stopping)
    host._dispatch_receipt = broken
    l, c = (host.dispatch_receipt(d, stopping=True) for d in host.plan["dispatch"])
    assert l["receipts_from"] == "records" and l["state"] == "damaged" and l["dispatch_status"] == "cut_off"
    assert [p["status"] for p in l["positions"][:2]] == ["damaged", "complete"]
    assert l["positions"][1]["input_matches_schedule"] is False          # no inputs record: never counted a match
    assert l["counts"]["not_started"] == 18
    assert sorted(Path(x["path"]).name for x in l["damaged"]) == ["L-state.json", "summary.json", "timing.jsonl"]
    assert all(x["sha256"] for x in l["damaged"])
    assert (root / "L-state.json").read_text() == '{"status": "running", "positions": ['     # left as it was
    assert c["dispatch_status"] == "receipt_error" and c["counts"] == {"unknown": 5}


def test_f_waits_for_a_valid_snapshot_and_skips_an_invalid_one(tmp_path, three_phases):
    study = _copy(three_phases, tmp_path)
    root = study / str(STREAM)
    host = _bare_host(tmp_path, study)
    d = {"name": f"F:{STREAM}", "arm": "F", "stream": STREAM, "root": str(root)}
    host.plan = {"dispatch": [d]}
    assert host.ready(d) == "no snapshot from L"
    library = tmp_path / "library"
    library.mkdir()
    (library / "events.log").write_text("one\n")
    (library / "entry.json").write_text("{}")
    from paper2.experiment2 import run_stream as RS
    RS._snapshot_library(library, root / "inputs" / "F-snapshot", S.streams.episode_id(STREAM, 2, 19))
    assert host.ready(d) == "go"
    (root / "inputs/F-snapshot/library/entry.json").write_text('{"changed": true}')
    assert host.ready(d).startswith("invalid snapshot")


def test_a_second_host_on_the_same_study_is_refused(tmp_path, two_streams):
    import fcntl
    study = _copy(two_streams, tmp_path)
    with open(study / ".host.lock", "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code, out = _host(tmp_path, study, "--arms", "calculator")
    assert code != 0 and "another host" in out
    assert [e["event"] for e in _events(tmp_path)] == ["host_start", "refused"]


@pytest.mark.parametrize("arm", ["C", "F"])
@pytest.mark.parametrize("complete", [True, False], ids=["complete", "censored"])
def test_an_interrupted_exit_overrides_a_saved_summary_when_receipts_are_rebuilt(tmp_path, arm, complete):
    root, ep = tmp_path / str(STREAM), "e1"
    flight = root / "arms" / arm / ep / "flight" / ep
    flight.mkdir(parents=True)
    (flight / "summary.json").write_text(json.dumps({"complete": complete}))
    (root / "arms" / arm / f"{ep}.exit.json").write_text(json.dumps({"exit": -15, "interrupted": True,
                                                                     "signals": [15]}))
    host = _bare_host(tmp_path)
    [p] = host.rebuild(root, arm, [{"episode": ep, "scenario_sha256": "x"}], stopping=True, damaged=[])
    assert p["status"] == "interrupted" and p["interrupted"] is True and p["process_exit"] == -15
    assert p["summary_status"] == ("complete" if complete else "censored")


def test_a_rebuilt_receipt_binds_the_summary_it_read(tmp_path):
    """Codex's TURN303 review: the analysis reads a cut-off condition through the host's rebuilt receipt, so the
    receipt carries the summary's path and digest, as the runner's own receipt does."""
    import hashlib
    root, ep = tmp_path / str(STREAM), "e1"
    summary = root / "arms" / "L" / ep / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"complete": True, "goal_met": True}))
    [p] = _bare_host(tmp_path).rebuild(root, "L", [{"episode": ep, "scenario_sha256": "x"}], stopping=True, damaged=[])
    assert p["status"] == "complete" and p["summary"] == f"arms/L/{ep}/summary.json"
    assert p["summary_sha256"] == hashlib.sha256(summary.read_bytes()).hexdigest()


def _leaderless_group(tmp_path, folder, episode, dispatcher, marked=True):
    """A worker that leads its own group, starts a sleeping child there and exits, with its record. The child carries
    the recorded marker unless marked is False, which stands for an unrelated group that later took the number."""
    token = os.urandom(16).hex()
    env = {k: v for k, v in os.environ.items() if k != S.WORKER_TOKEN}
    if marked:
        env[S.WORKER_TOKEN] = token
    leader = subprocess.Popen([sys.executable, "-c", "import subprocess, sys\n"
                               "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
                               "print(p.pid, flush=True)"], start_new_session=True, stdout=subprocess.PIPE, text=True,
                              env=env)
    child = int(leader.stdout.readline())
    assert leader.wait(timeout=10) == 0
    (folder / f"{episode}.worker.json").write_text(json.dumps({"pid": leader.pid, "pgid": leader.pid,
                                                               "dispatcher_pid": dispatcher, "episode": episode,
                                                               "token": token}))
    return leader.pid, child


def test_a_worker_group_left_after_its_leader_exits_is_swept_only_when_it_is_ours(tmp_path):
    root = str(tmp_path / str(STREAM))
    folder = Path(root) / "arms" / "C"
    folder.mkdir(parents=True)
    me = os.getpid()
    ours, ours_child = _leaderless_group(tmp_path, folder, "e1", me)
    theirs, theirs_child = _leaderless_group(tmp_path, folder, "e2", me + 1)
    try:
        mine = [{"name": f"C:{STREAM}", "arm": "C", "root": root, "pid": me}]
        found = H.owned_workers(mine)
        assert [(o["pid"], o["leader"], o["members"]) for o in found] == [(ours, "exited", [ours_child])]
        host = _bare_host(tmp_path)
        host.plan = {"dispatch": mine}
        swept = host.sweep()
        assert [o["pid"] for o in swept] == [ours] and swept[0]["gone"] is True and _gone(ours_child)
        assert not _gone(theirs_child, 0.5)                    # another dispatcher's group is left alone
        (folder / "e1.worker.json").write_text(json.dumps({"pid": ours, "pgid": ours + 1, "dispatcher_pid": me,
                                                           "episode": "e1"}))
        assert H.owned_workers(mine) == []                     # a record whose worker did not lead its group
    finally:
        for pid in (ours_child, theirs_child):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _kill(*pids):
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_a_closed_workers_old_record_selects_nothing(tmp_path):
    root = str(tmp_path / str(STREAM))
    folder = Path(root) / "arms" / "C"
    folder.mkdir(parents=True)
    me = os.getpid()
    leader, child = _leaderless_group(tmp_path, folder, "e1", me)
    try:
        (folder / "e1.exit.json").write_text(json.dumps({"exit": 0, "group_left_running": False}))
        mine = [{"name": f"C:{STREAM}", "arm": "C", "root": root, "pid": me}]
        assert H.owned_workers(mine) == []                     # closed and reaped by its dispatcher: a later lifetime
        host = _bare_host(tmp_path)
        host.plan = {"dispatch": mine}
        assert host.sweep() == [] and not _gone(child, 0.5)
    finally:
        _kill(child)


def test_a_reused_number_with_unrelated_members_is_reported_and_left_alone(tmp_path):
    root = str(tmp_path / str(STREAM))
    folder = Path(root) / "arms" / "C"
    folder.mkdir(parents=True)
    me = os.getpid()
    leader, child = _leaderless_group(tmp_path, folder, "e1", me, marked=False)
    try:
        mine = [{"name": f"C:{STREAM}", "arm": "C", "root": root, "pid": me}]
        [found] = H.owned_workers(mine)
        assert found["action"] == "report" and found["members"] == [child] and "marker" in found["reason"]
        host = _bare_host(tmp_path)
        host.plan = {"dispatch": mine}
        [reported] = host.sweep()
        assert reported["action"] == "report" and "gone" not in reported and not _gone(child, 0.5)
    finally:
        _kill(child)


def test_a_running_leader_without_its_marker_is_reported(tmp_path):
    root = str(tmp_path / str(STREAM))
    folder = Path(root) / "arms" / "C"
    folder.mkdir(parents=True)
    worker = _sleeper(*_worker_tokens(root, "C", "e1"))
    try:
        (folder / "e1.worker.json").write_text(json.dumps({"pid": worker.pid, "pgid": worker.pid,
                                                           "dispatcher_pid": os.getpid(), "episode": "e1",
                                                           "token": "f" * 32}))
        assert _wait_for(lambda: H._process(worker.pid) is not None, 10)
        [found] = H.owned_workers([{"name": f"C:{STREAM}", "arm": "C", "root": root, "pid": os.getpid()}])
        assert found["action"] == "report" and found["leader"] == "running"
    finally:
        worker.kill()
        worker.wait()


# -- isolation and the startup check ----------------------------------------------------------------------------------
REPO = HERE.parents[2]
HOSTILE = "import pathlib\npathlib.Path({marker!r}).write_text('imported')\nraise SystemExit(99)\n"


def _payload(tmp_path):
    """A payload root holding the package and the product, as the launcher's payload will."""
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "paper2").symlink_to(REPO / "paper2")
    (payload / "telohelm").symlink_to(REPO / "paper2/telohelm/src/telohelm")
    return payload


def _hostile(tmp_path):
    """A working directory, also put on PYTHONPATH, whose package and standard-module look-alikes leave a marker if
    they are ever imported."""
    hostile, marker = tmp_path / "hostile", tmp_path / "HOSTILE-IMPORTED"
    for rel in ("paper2/__init__.py", "paper2/experiment2/__init__.py", "paper2/experiment2/study_runner.py",
                "paper2/experiment2/study_host.py", "telohelm/__init__.py", "json.py", "numpy.py"):
        f = hostile / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(HOSTILE.format(marker=str(marker)))
    env = {**{k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}, "PYTHONPATH": str(hostile),
           "PYTHONDONTWRITEBYTECODE": "1"}
    return hostile, marker, env


def test_the_hostile_folder_is_imported_by_an_unisolated_command(tmp_path, two_streams):
    hostile, marker, env = _hostile(tmp_path)
    argv = S.module_command(sys.executable, "paper2.experiment2.study_host", ["check", "--study", str(two_streams)])
    done = subprocess.run(argv, cwd=hostile, env=env, capture_output=True, text=True, timeout=120)
    assert marker.exists() and done.returncode != 0          # the trap works when the path is not isolated


def test_an_isolated_check_ignores_the_hostile_folder_and_writes_nothing(tmp_path, two_streams):
    study = _copy(two_streams, tmp_path)
    payload, (hostile, marker, env) = _payload(tmp_path), _hostile(tmp_path)
    before = sorted((str(p.relative_to(study)), p.stat().st_mtime_ns) for p in study.rglob("*"))
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(STAND_IN))
    argv = S.module_command(sys.executable, "paper2.experiment2.study_host",
                            ["check", "--study", str(study), "--profile", str(profile), "--arms", "L", "C"], payload)
    done = subprocess.run(argv, cwd=hostile, env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    result = json.loads(done.stdout)
    assert result["ok"] and [d["name"] for d in result["dispatch"]] == [f"L:{STREAM}", f"C:{STREAM}", f"L:{OTHER}",
                                                                         f"C:{OTHER}"]
    assert [d["positions"] for d in result["dispatch"]] == [20, 5, 20, 5] and not marker.exists()
    assert sorted((str(p.relative_to(study)), p.stat().st_mtime_ns) for p in study.rglob("*")) == before
    S.declare_study(tmp_path / "bare", [STREAM])
    refused = subprocess.run(S.module_command(sys.executable, "paper2.experiment2.study_host",
                                              ["check", "--study", str(tmp_path / "bare"), "--arms", "calculator"],
                                              payload), cwd=hostile, env=env, capture_output=True, text=True, timeout=120)
    assert refused.returncode == 2 and json.loads(refused.stdout)["ok"] is False and not marker.exists()


def test_an_isolated_host_its_dispatchers_and_their_workers_ignore_the_hostile_folder(tmp_path, two_streams):
    study = _copy(two_streams, tmp_path)
    payload, (hostile, marker, env) = _payload(tmp_path), _hostile(tmp_path)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(STAND_IN))
    argv = S.module_command(sys.executable, "paper2.experiment2.study_host",
                            ["run", "--study", str(study), "--profile", str(profile), "--out", str(tmp_path / "host"),
                             "--deadline-epoch", str(time.time() + 600), "--gpus", "2", "--max-steps", "10",
                             "--arms", "C", "calculator", "--payload", str(payload)], payload)
    done = subprocess.run(argv, cwd=hostile, env=env, capture_output=True, text=True, timeout=600)
    assert done.returncode == 3, done.stdout + done.stderr      # censored at the stand-in step limit
    assert not marker.exists()
    d = _by_name(_receipts(tmp_path))
    for s in (STREAM, OTHER):
        c = d[f"C:{s}"]
        assert c["dispatch_status"] == "returned" and c["counts"] == {"censored": 5}
        assert all(p["process_exit"] == 0 and p["input_matches_schedule"] for p in c["positions"])
        assert d[f"calculator:{s}"]["dispatch_status"] == "returned"
        state = json.loads((study / str(s) / "C-state.json").read_text())
        assert state["payload"] == str(payload.resolve())
    starts = [e for e in _events(tmp_path) if e["event"] == "start"]
    assert starts and all(e["argv"][1:4] == ["-I", "-B", "-c"] and e["argv"][5] == str(payload.resolve()) for e in starts)


def test_isolated_worker_commands_still_match_the_sweeps_ownership_check(tmp_path):
    argv = S._position_argv(tmp_path / "4493", "C", "e1", None, tmp_path / "payload")
    assert argv[1:3] == ["-I", "-B"] and argv[3] == "-c"
    assert H._runs(argv, ["paper2.experiment2.study_runner", "fly-one", "--root", str(tmp_path / "4493"),
                          "--arm", "C", "--episode", "e1"])
    assert S._position_argv(tmp_path / "4493", "C", "e1", None)[1:4] == ["-B", "-m", "paper2.experiment2.study_runner"]


def test_a_released_dispatchers_number_is_never_signalled_again(tmp_path):
    host = _bare_host(tmp_path)
    exited, unrelated = _exited_dispatcher()
    try:
        host.procs = {"calculator:4493": exited}
        host.release("calculator:4493")       # collected earlier; its number may now belong to another group
        assert host.close_groups("finished", stopping=False) == {"signalled": [], "killed": [], "unreleased": []}
        assert not _gone(unrelated, 0.5)
    finally:
        _kill(unrelated)


def test_an_exited_dispatcher_is_held_while_its_group_has_members_and_released_once_empty(tmp_path):
    host = _bare_host(tmp_path)
    exited, child = _exited_dispatcher()
    host.procs = {"L:4493": exited}
    host.reap()
    assert host.held == {"L:4493"} and not host.released and exited.returncode is None
    _kill(child)
    assert _wait_for(lambda: not host.members("L:4493", H._table()), 10)
    host.reap()
    assert host.released == {"L:4493"} and host.ended == {"L:4493": 0}


@pytest.mark.parametrize("profile", ["missing", "malformed", "unreadable"])
def test_check_refuses_a_bad_profile_with_a_structured_answer(tmp_path, two_streams, capsys, profile):
    path = tmp_path / "profile.json"
    if profile == "malformed":
        path.write_text('{"stand_in": tr')
    elif profile == "unreadable":
        path.mkdir()
    assert H.main(["check", "--study", str(two_streams), "--profile", str(path), "--arms", "C"]) == 2
    answer = json.loads(capsys.readouterr().out)
    assert answer["ok"] is False and answer["error"]


# -- when the process table cannot be read ------------------------------------------------------------------------------
@pytest.mark.parametrize('arm, stream', [('L', 4493), ('C', 4495)])
def test_dispatcher_labels_reach_children_and_keep_job_slots_and_cache(tmp_path, monkeypatch, arm, stream):
    """A short owned child reads only invented metadata; no study, scenario or model is started."""
    inherited = {'TELOHELM_ARM': 'old-arm', 'TELOHELM_STREAM': 'old-stream',
                 'TELOHELM_JOB_TOKEN': 'invented-job-token',
                 'TELOHELM_MODEL_SLOTS': '{"http://127.0.0.1:1":{"dir":"/invented/slots","n":1}}',
                 'HF_HOME': '/invented/offline-cache', 'HF_HUB_OFFLINE': '1'}
    for key, value in inherited.items():
        monkeypatch.setenv(key, value)
    host = H.Host.__new__(H.Host)
    host.out, host.procs = tmp_path, {}
    (tmp_path / 'logs').mkdir()
    seen = []
    host.note = lambda **row: seen.append(row)
    keys = list(inherited)
    code = 'import json,os;print(json.dumps({k:os.environ.get(k) for k in ' + repr(keys) + '}))'
    host.argv = lambda d: [sys.executable, '-I', '-B', '-c', code]
    d = {'name': f'{arm}:{stream}', 'arm': arm, 'stream': stream}
    host.launch(d)
    child = host.procs[d['name']]
    try:
        assert child.wait(timeout=10) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
    got = json.loads((tmp_path / 'logs' / f'{arm}-{stream}.log').read_text())
    assert got == {**inherited, 'TELOHELM_ARM': arm, 'TELOHELM_STREAM': str(stream)}
    assert {k: os.environ[k] for k in inherited} == inherited
    assert seen[0]['arm'] == arm and seen[0]['stream'] == stream and seen[0]['pid'] == child.pid


def _failing_ps(monkeypatch, returncode=1, stdout=""):
    real = subprocess.run

    def fake(argv, *args, **kwargs):
        if list(argv[:2]) == ["ps", "-A"]:
            return subprocess.CompletedProcess(argv, returncode, stdout, "")
        return real(argv, *args, **kwargs)
    monkeypatch.setattr(H.subprocess, "run", fake)


@pytest.mark.parametrize("returncode, stdout", [(1, ""), (0, ""), (0, "1 1 S\n2 2 S\n"), (0, "x y z\n")],
                         ids=["failed", "empty", "without this process", "unreadable"])
def test_a_failed_or_incomplete_listing_is_an_observation_failure(monkeypatch, returncode, stdout):
    _failing_ps(monkeypatch, returncode, stdout)
    with pytest.raises(H.ObservationFailed):
        H._table()


def test_a_live_dispatcher_is_not_released_when_the_table_cannot_be_read(tmp_path, monkeypatch):
    host = _bare_host(tmp_path)
    live = _sleeper()
    try:
        host.procs = {"calculator:4493": live}
        _failing_ps(monkeypatch)
        began = time.monotonic()
        host.reap()
        assert time.monotonic() - began < 2 and not host.released and live.poll() is None
        assert host.exited("calculator:4493", host.observe()) is False and len(host.observation_failures) == 2
    finally:
        live.kill()
        live.wait()


def test_cleanup_without_observations_still_ends_the_hosts_own_groups_within_bounds(tmp_path, monkeypatch):
    host = _bare_host(tmp_path)
    running = _sleeper(grandchild=True)
    exited, child = _exited_dispatcher()
    host.procs = {"C:4493": running, "L:4493": exited}
    monkeypatch.setattr(H, "STOP_GRACE_S", 1.0)
    _failing_ps(monkeypatch)
    began = time.monotonic()
    closed = host.close_groups("signal 15", stopping=True)
    elapsed = time.monotonic() - began
    monkeypatch.undo()
    assert elapsed < 1.0 + 5.0 + 2 * H.RELEASE_WAIT_S + 2
    assert closed["signalled"] == ["C:4493", "L:4493"] and closed["unreleased"] == []
    assert host.released == {"C:4493", "L:4493"} and host.observation_failures and _gone(child)
    assert not H.S._group_alive(running.pid)


@pytest.mark.parametrize("how", ["deadline", "stop"])
def test_the_host_stops_and_cleans_up_while_the_table_cannot_be_read(tmp_path, monkeypatch, how):
    import argparse
    out = tmp_path / "host"
    (out / "logs").mkdir(parents=True)
    a = argparse.Namespace(study=tmp_path, out=out, arms=["calculator"], profile=None,
                           deadline_epoch=time.time() + (2 if how == "deadline" else 600), python=sys.executable,
                           max_steps=None, payload=None, gpus=0)
    host = H.Host(a)
    host.plan = {"dispatch": [{"name": f"calculator:{STREAM}", "arm": "calculator", "stream": STREAM,
                               "root": str(tmp_path / str(STREAM)), "rows": []}]}
    started = []

    def launch(d):
        p = _sleeper(grandchild=True)
        host.procs[d["name"]], d["pid"] = p, p.pid
        started.append(p)
        if how == "stop":
            host.on_signal(signal.SIGTERM, None)
    monkeypatch.setattr(host, "launch", launch)
    monkeypatch.setattr(H, "STOP_GRACE_S", 1.0)
    _failing_ps(monkeypatch)
    began = time.monotonic()
    assert host._fly(time.time(), 0) == 3
    assert time.monotonic() - began < 30
    monkeypatch.undo()
    r = json.loads((out / "host-receipts.json").read_text())
    assert r["reason"] == ("deadline" if how == "deadline" else "signal 15") and r["verdict"]["complete"] is False
    assert r["observation_failures"] and r["release_failures"] == [] and r["groups"]["unreleased"] == []
    assert started[0].returncode is not None and not H.S._group_alive(started[0].pid)
    assert any(o["action"] == "report" and "observation failed" in o["reason"] for o in r["swept"])


def test_a_selected_stream_flies_alone_and_the_next_runs_later_with_the_anchor_on_both(tmp_path, two_streams):
    """Protocol amendment 7, item 4: scored streams run one job at a time; the pause anchor still spans every
    declared stream."""
    study = _copy(two_streams, tmp_path)
    one = H.plan(study, ("L", "C", "C-full"), STAND_IN, 10, (STREAM,))
    assert one["streams"] == [STREAM] and one["declared_streams"] == [STREAM, OTHER]
    assert sorted(d["name"] for d in one["dispatch"]) == sorted(f"{a}:{STREAM}" for a in ("L", "C", "C-full"))
    for bad in ((9999,), (STREAM, STREAM), ()):
        with pytest.raises(ValueError, match="distinct streams the study declares"):
            H.plan(study, ("C",), STAND_IN, 10, bad)
    anchor = S.pause_anchor(study, "C")
    assert {e.split("-")[0] for e in anchor} == {f"s{STREAM}", f"s{OTHER}"}
    code, out = _host(tmp_path, study, "--arms", "C", "--streams", str(STREAM))
    assert code == 3 and _receipts(tmp_path)["reason"] == "finished", out    # stand-in flights stop at their step limit
    assert sorted(_by_name(_receipts(tmp_path))) == [f"C:{STREAM}"]
    assert (study / str(STREAM) / "C-state.json").is_file() and not (study / str(OTHER) / "C-state.json").exists()
    later = tmp_path / "later"
    later.mkdir()
    code, out = _host(later, study, "--arms", "C", "--streams", str(OTHER))
    assert code == 3 and _receipts(later)["reason"] == "finished", out
    assert sorted(_by_name(_receipts(later))) == [f"C:{OTHER}"]
    assert (study / str(OTHER) / "C-state.json").is_file()
