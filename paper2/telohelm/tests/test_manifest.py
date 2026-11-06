"""The draft configuration manifest (TURN 83). Invented contents; no section's value is chosen here."""
from __future__ import annotations

import copy
import hashlib

import pytest

import telohelm.runtime.manifest as F


def invented(origin="http://127.0.0.1:9", models=None, timeouts=None, temperature=0.0, max_response_bytes=1 << 20,
             served=None):
    """A complete invented manifest. TURN 85 (Codex's MF84.REQUIRED): every required field is present and typed, so a
    negative test changes one thing. The defaults match the adapters built in test_exchange_join.py."""
    models = models or {"small_model": "invented-small", "reasoning_model": "invented-reasoning"}
    timeouts = timeouts or {"small_model": 30.0, "reasoning_model": 600.0}
    served = served or {r: [m] for r, m in models.items()}
    return {
        "version": 1,
        "server": {"software": "invented-server", "version": "0.0", "launch_args": ["--invented"]},
        "weights": {"repository": "invented/model", "revision": "0" * 40,
                    "files": {"model-00001.safetensors": "a" * 64}},
        "names": {"served": served},
        "prompts": {"messages_builder_sha256": "1" * 64, "chat_template_sha256": "2" * 64, "parser_sha256": "3" * 64},
        "sampling": {"temperature": temperature, "max_output_tokens": {"small_model": 1024, "reasoning_model": 8192},
                     "seed": None},
        "transport": {"origin": origin, "max_response_bytes": max_response_bytes, "timeout_s": timeouts},
        "runtime": {"python": "3.12", "dependencies": {}, "code_sha256": {"transport.py": "4" * 64}},
        "roles": {"model": models, "modelled_latency_s": {"small_model": 2.4, "reasoning_model": 60.0},
                  "reasoning": {"small_model": {}, "reasoning_model": {}}},
        "generation": {"max_tokens": {"small_model": 1024, "reasoning_model": 8192}, "stop": [], "seed": None},
        "environment": {"hardware": "invented", "precision": "bf16", "concurrency": 1, "batching": {}},
    }


def frozen(tmp_path, value):
    p = tmp_path / "expected"
    p.write_text(value)
    return p


def test_the_fingerprint_is_canonical_and_checked_against_the_frozen_one(tmp_path):
    m = invented()
    fp = F.fingerprint(m)
    reordered = dict(reversed(list(m.items())))
    assert F.fingerprint(reordered) == fp
    assert F.check(m, frozen(tmp_path, fp + "\n")) == fp


@pytest.mark.parametrize("change", [
    lambda m: m["weights"]["files"].update({"model-00001.safetensors": "b" * 64}),
    lambda m: m["generation"].update(temperature=0.1),
    lambda m: m["names"].update(extra="alias"),
    lambda m: m["environment"].update(hardware="other"),
])
def test_any_change_fails_the_check(tmp_path, change):
    m = invented()
    path = frozen(tmp_path, F.fingerprint(m))
    changed = copy.deepcopy(m)
    change(changed)
    with pytest.raises(F.ManifestError):
        F.check(changed, path)


@pytest.mark.parametrize("text", [None, "", "a" * 63, "A" * 64, "a" * 64 + "\n\n", " " + "a" * 64, "g" * 64])
def test_a_missing_or_malformed_expected_fingerprint_fails_closed(tmp_path, text):
    path = tmp_path / "expected" if text is None else frozen(tmp_path, text)
    with pytest.raises(F.ManifestError):
        F.check(invented(), path)


@pytest.mark.parametrize("change", [
    lambda m: m.pop("environment"),
    lambda m: m.update(extra={"x": 1}),
    lambda m: m.update(version=2),
    lambda m: m.update(version=True),
    lambda m: m.update(roles={}),
    lambda m: m.update(roles=["x"]),
    lambda m: m["generation"].update(temperature=float("nan")),
    lambda m: m["generation"].update(temperature=float("inf")),
    lambda m: m["prompts"].update(text="\ud800"),
    lambda m: m["prompts"].update({"\ud800": "x"}),
    lambda m: m["server"].update(args=("a", "b")),
    lambda m: m["server"].update({1: "x"}),
    lambda m: m.update(version=1.0),                                  # TURN 85 (MF84.REQUIRED): typed version
    lambda m: m["weights"].update(files={"x": "not a digest"}),
    lambda m: m["weights"].pop("revision"),
    lambda m: m["prompts"].update(parser_sha256=None),
    lambda m: m["sampling"].update(max_output_tokens={"small_model": 1024}),
    lambda m: m["sampling"].update(temperature=True),
    lambda m: m["transport"].update(timeout_s={"small_model": 0, "reasoning_model": 1}),
    lambda m: m["names"].update(served={"small_model": [], "reasoning_model": ["x"]}),
    lambda m: m["environment"].update(concurrency=0),
    lambda m: m.update({s: {"unknown": None} for s in F.SECTIONS}),
])
def test_a_malformed_manifest_is_refused(change):
    m = invented()
    change(m)
    with pytest.raises(F.ManifestError):
        F.fingerprint(m)


def test_file_digests(tmp_path):
    (tmp_path / "a.py").write_bytes(b"invented")
    assert F.file_digests([tmp_path / "a.py"]) == {str(tmp_path / "a.py"): hashlib.sha256(b"invented").hexdigest()}
    with pytest.raises(F.ManifestError):
        F.file_digests([tmp_path / "missing.py"])


def test_a_checked_configuration_is_made_only_by_the_check(tmp_path):
    m = invented()
    c = F.checked(m, frozen(tmp_path, F.fingerprint(m)))
    assert c.fingerprint == F.fingerprint(m) and c.section("transport")["origin"] == "http://127.0.0.1:9"
    m["transport"]["origin"] = "changed later"
    assert c.section("transport")["origin"] == "http://127.0.0.1:9"     # a private copy
    with pytest.raises(F.ManifestError):
        F.CheckedConfig(m, "0" * 64)


def test_the_roles_agree_with_the_gateway():
    import telohelm.runtime.models as M
    assert F.ROLES == M.ROLES
