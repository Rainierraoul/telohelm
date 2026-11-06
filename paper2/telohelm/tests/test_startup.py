"""The draft start-of-run evidence (TURN 93). Invented values; no real installation, server or machine is examined, except
that the imported-code capture hashes this process's own runtime source files."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import telohelm.runtime.startup as S
from telohelm.runtime.trusted_log import TrustedLog
from .test_exchange_join import checked

WHEN, PROTECTED = "invented capture time", "invented protected store"


def matching(config):
    want = S.expected(config)
    return S.record(config.fingerprint, "before the first model call",
                    {name: S.observed(want[name], f"invented {name} source", WHEN, PROTECTED) for name in S.ITEMS})


def test_a_matching_record_has_no_problems():
    c = checked()
    assert S.compare(matching(c), c) == []


@pytest.mark.parametrize("name", S.ITEMS)
def test_each_missing_item_is_a_problem(name):
    c = checked(); rec = matching(c)
    rec["items"].pop(name)
    assert S.compare(rec, c) == [f"{name}: missing"]


@pytest.mark.parametrize("name", S.ITEMS)
def test_each_unknown_item_is_a_problem_and_never_a_default(name):
    c = checked(); rec = matching(c)
    rec["items"][name] = S.unknown(f"invented {name} source", WHEN, PROTECTED)
    assert S.compare(rec, c) == [f"{name}: unknown"]


@pytest.mark.parametrize("name,value", [
    ("installed_code", {"transport.py": "4" * 64, "extra.py": "5" * 64}),     # all and only
    ("installed_code", {"transport.py": "5" * 64}),
    ("python", "3.11"), ("dependencies", {"invented": "1.0"}),
    ("weights", {"model-00001.safetensors": "b" * 64}),
    ("launch", ["--other"]), ("model_list", ["invented-small"]),
    ("effective_defaults", {"max_tokens": {"small_model": 1, "reasoning_model": 1}, "stop": [], "seed": None}),
    ("environment", {"hardware": "other", "precision": "bf16", "concurrency": 1, "batching": {}}),
])
def test_a_differing_item_is_a_problem(name, value):
    c = checked(); rec = matching(c)
    rec["items"][name] = S.observed(value, "invented source", WHEN, PROTECTED)
    assert S.compare(rec, c) == [f"{name}: differs from the configuration"]


def test_a_record_for_another_configuration_is_a_problem():
    c = checked(); other = checked(temperature=0.5)
    assert "another configuration" in S.compare(matching(other), c)[0]


@pytest.mark.parametrize("change", [
    lambda r: r["items"]["python"].pop("source"),
    lambda r: r["items"]["python"].update(captured_at=""),
    lambda r: r["items"]["python"].update(protection=" "),
    lambda r: r["items"]["python"].update(status="assumed"),
    lambda r: r["items"]["python"].update(status="unknown"),               # unknown but still carrying a value
    lambda r: r["items"].update(invented={}),
    lambda r: r.update(capture_point=""),
    lambda r: r.update(version=1.0),
    lambda r: r["items"]["python"].update(value=float("nan")),
])
def test_a_malformed_record_is_refused(change):
    c = checked(); rec = matching(c)
    change(rec)
    with pytest.raises(S.StartupError):
        S.compare(rec, c)


def test_imported_code_is_captured_from_the_files_actually_loaded():
    base = Path(S.__file__).resolve().parent
    mods = S.imported('telohelm.runtime')
    assert S in mods
    code = S.capture_code(mods, base)
    assert "startup.py" in code and "transport.py" in code and all(len(v) == 64 for v in code.values())
    with pytest.raises(S.StartupError):
        S.capture_code([json], base)                                    # outside the runtime package


def _log(tmp_path, *events):
    log = TrustedLog(tmp_path / "log", tmp_path / "heads", "invented")
    for e in events:
        if e[0] == "start":
            S.write(log, e[1])
        else:
            log.append({"kind": e[0]})
    return tmp_path / "log", tmp_path / "heads"


def test_a_start_record_before_the_first_call_admits(tmp_path):
    c = checked()
    assert S.admit_start(*_log(tmp_path, ("start", matching(c)), ("model_start",)), c) == []


def test_a_record_after_the_first_call_does_not_admit_it(tmp_path):
    c = checked()
    assert S.admit_start(*_log(tmp_path, ("model_start",), ("start", matching(c))), c) == [
        "a model call came before any start record"]


def test_no_record_or_a_mismatching_one_does_not_admit(tmp_path):
    c = checked()
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    assert S.admit_start(*_log(tmp_path / "a", ("model_end",)), c) == ["no start record"]
    bad = matching(c); bad["items"]["weights"]["value"] = {"x": "0" * 64}
    assert S.admit_start(*_log(tmp_path / "b", ("start", bad)), c) == ["weights: differs from the configuration"]


def test_a_changed_log_does_not_admit(tmp_path):
    c = checked()
    log, heads = _log(tmp_path, ("start", matching(c)))
    lines = log.read_text().splitlines()
    row = json.loads(lines[1]); row["payload"]["record"]["items"]["python"]["value"] = "3.11"
    log.write_text("\n".join([lines[0], json.dumps(row)] + lines[2:]) + "\n")
    assert "does not verify" in S.admit_start(log, heads, c)[0]


def test_expected_values_are_copies_of_the_configuration():
    c = checked()
    want = S.expected(c); want["weights"]["x"] = "0" * 64
    assert "x" not in S.expected(c)["weights"]
    assert copy.deepcopy(S.expected(c)) == S.expected(c)


# TURN 95 (Codex's TURN 94 review).

@pytest.mark.parametrize("a,b,equal", [(1, 1, True), (True, 1, False), (1.0, 1, False), (0, False, False),
                                       ({"a": [1, 2]}, {"a": [1, 2]}, True), ({"a": [1, 2]}, {"a": [1, 2.0]}, False),
                                       ([1, 2], [2, 1], False), ({"a": 1, "b": 2}, {"b": 2, "a": 1}, True),
                                       (None, None, True), (None, 0, False), ("1", 1, False)])
def test_values_are_compared_by_type_and_value(a, b, equal):
    assert S.same(a, b) is equal


def test_the_prompt_pins_must_be_observed_and_match():
    c = checked(); rec = matching(c)
    rec["items"]["prompts"]["value"]["parser_sha256"] = "9" * 64
    assert S.compare(rec, c) == ["prompts: differs from the configuration"]


def test_code_drift_finds_a_file_changed_or_removed_after_capture(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n"); (tmp_path / "b.py").write_text("y = 2\n")
    import importlib.util
    mods = []
    for name in ("a", "b"):
        spec = importlib.util.spec_from_file_location(f"turn95_{name}", tmp_path / f"{name}.py")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); mods.append(m)
    captured = S.capture_code(mods, tmp_path)
    assert S.code_drift(captured, tmp_path) == []
    (tmp_path / "a.py").write_text("x = 3\n"); (tmp_path / "b.py").unlink()
    assert S.code_drift(captured, tmp_path) == ["a.py", "b.py"]


# TURN 97: the start step tests use (an invented pre-import code evidence and a record made from the configuration).

def invented_code(config, phase="pre-import", problems=None, files=None):
    # TURN 99: the full typed schema (Codex's SG98.CODE.SCHEMA), with no bytecode and the pinned files loaded by the
    # source loader.
    files = dict(config.section("runtime")["code_sha256"]) if files is None else files
    return {"phase": phase, "captured_at": "invented capture before import", "protection": "invented read-only install",
            "base": "invented base", "files": files, "bytecode": {},
            "loaded": [{"module": f"invented.{p}", "file": p, "loader": "SourceFileLoader", "cached": None}
                       for p in files],
            "problems": [] if problems is None else problems}


def start_gateway(gw, provider, **kw):
    """Admit a store-backed gateway's start with invented evidence that matches its provider's configuration."""
    authority = S.RunAuthority(provider.config, invented_code(provider.config, **kw))
    return gw.record_start(matching(provider.config), authority)
