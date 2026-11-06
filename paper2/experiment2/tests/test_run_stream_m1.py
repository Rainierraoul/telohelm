"""The stream runner's options for the M1 timing rehearsal (paper2/development/m1-timing-rehearsal/): a library snapshot
at a named episode, a start from an exact copy, learning off with the reasoning model kept for flight, layout v2 and
timing spans. Stand-in models only; nothing here measures learning or flight quality."""
import json
from pathlib import Path

import pytest

import paper2.experiment2.run_stream as RS
from paper2.experiment2 import streams as G

GUARDS = Path(RS.__file__).resolve().parents[1] / "telohelm/src/telohelm/taskpacks/tanker/guards"
DEAD = {"base_url": "http://127.0.0.1:9", "model": "unused", "timeout_s": 1.0}


def _files(root):
    return RS._files(root)


def test_a_snapshot_is_whole_and_a_copy_starts_from_it(tmp_path):
    first = tmp_path / "learning"
    snap = tmp_path / "snapshot"
    RS.run_stream(2, 1, first, max_steps=15, stream_seed=4491, phases=(1,), snapshot_after=("s4491-p1-e00", snap),
                  timing=True)
    meta = json.loads((snap / "snapshot.json").read_text())
    assert meta["after_episode"] == "s4491-p1-e00"
    assert meta["files"] == _files(snap / "library") and meta["files"]
    assert not snap.with_name("snapshot.partial").exists()
    rows = [json.loads(l) for l in (first / "timing.jsonl").read_text().splitlines()]
    assert [(r["event"], r["part"]) for r in rows] == [("start", "flight"), ("end", "flight"), ("start", "snapshot"),
                                                       ("end", "snapshot"), ("start", "flight"), ("end", "flight")]

    copy = tmp_path / "frozen"
    scenario = G.episode(4491, 1, 1).scenario
    RS.run_stream(1, 1, copy, max_steps=15, stream_seed=4491, prepared_scenarios=(scenario,), empty_library=True,
                  library_from=snap / "library")
    assert json.loads((copy / "library-from.json").read_text())["files"] == meta["files"]
    summary = json.loads((copy / "stream-summary.json").read_text())
    assert summary["library_from"] == str((snap / "library").resolve())


def test_learning_off_keeps_flight_reasoning_and_learns_nothing(tmp_path):
    out = tmp_path / "off"
    RS.run_stream(1, 1, out, max_steps=10, stream_seed=4491, phases=(1,), learn=DEAD, learning=False)
    summary = json.loads((out / "stream-summary.json").read_text())
    assert summary["learning"] == {"off": True, "note": "reasoning model for flight only; no learning between episodes"}
    assert not (out / "learning-events.json").exists()


def test_layout_v2_reaches_the_generation_record(tmp_path):
    out = tmp_path / "v2"
    RS.run_stream(1, 1, out, max_steps=5, stream_seed=4491, phases=(3,), layout="v2")
    row = json.loads((out / "generation.jsonl").read_text().splitlines()[0])
    assert row["episode"] == "s4491-p3-e00" and row["layout"] == "v2"
    assert json.loads((out / "stream-summary.json").read_text())["layout"] == "v2"


@pytest.mark.parametrize("kwargs, message", [
    ({"learning": False}, "learning off keeps the reasoning model"),
    ({"learning": False, "learn": DEAD, "learn_candidates": 3}, "learning options need learning"),
    ({"library_from": Path("/nonexistent")}, "a copied library is used as it is"),
    ({"layout": "v2"}, "layout v2 needs a stream seed"),
])
def test_inconsistent_settings_are_refused(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        RS.run_stream(1, 1, tmp_path / "x", max_steps=5, **kwargs)


def test_a_copy_must_be_a_saved_library_and_a_snapshot_never_overwrites(tmp_path):
    (tmp_path / "notalibrary").mkdir()
    with pytest.raises(ValueError, match="not a saved library"):
        RS.run_stream(1, 1, tmp_path / "y", max_steps=5, empty_library=True, library_from=tmp_path / "notalibrary")
    taken = tmp_path / "taken"
    taken.mkdir()
    with pytest.raises(FileExistsError):
        RS.run_stream(1, 1, tmp_path / "z", max_steps=5, stream_seed=4491, phases=(1,),
                      snapshot_after=("s4491-p1-e00", taken))


def test_inputs_are_recorded_before_each_flight(tmp_path):
    out = tmp_path / "inputs"
    RS.run_stream(1, 1, out, max_steps=5, stream_seed=4491, phases=(1,), record_inputs=True)
    from paper2.experiment2.comparators import scenario_fingerprint
    row = json.loads((out / "inputs.jsonl").read_text().splitlines()[0])
    assert row == {"episode": "s4491-p1-e00", "scenario_sha256": scenario_fingerprint(G.episode(4491, 1, 0).scenario)}
