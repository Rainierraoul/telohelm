"""Code provenance before import (TURN 97, Codex's SG96.5 and SG96.6), with harmless temporary packages only."""
from __future__ import annotations

import importlib
import itertools
import sys
from pathlib import Path

import pytest

import telohelm.codecapture as C

_names = itertools.count()


@pytest.fixture
def package(tmp_path):
    """A fresh temporary package on sys.path, removed again afterwards."""
    name = f"turn97_pkg{next(_names)}"
    root = tmp_path / "site"
    pkg = root / name
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "mod.py").write_text("def answer():\n    return 1\n")
    sys.path.insert(0, str(root))
    old_flag = sys.dont_write_bytecode
    yield name, pkg
    sys.dont_write_bytecode = old_flag
    sys.path.remove(str(root))
    for m in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
        del sys.modules[m]


def load(name, module, bytecode=False):
    sys.dont_write_bytecode = not bytecode
    importlib.invalidate_caches()
    return importlib.import_module(f"{name}.{module}")


def test_a_catalog_before_import_verifies_the_loaded_modules(package):
    name, pkg = package
    cat = C.catalog(pkg, name, "invented time", "invented read-only install")
    assert cat["phase"] == "pre-import" and cat["already_loaded"] == [] and "mod.py" in cat["files"]
    m = load(name, "mod")
    assert C.verify_loaded(cat, [m]) == [] and C.drift(cat) == []
    assert C.evidence(cat, [m])["problems"] == []


def test_a_catalog_after_import_is_marked_post_import(package):
    name, pkg = package
    load(name, "mod")
    cat = C.catalog(pkg, name, "invented time", "invented protection")
    assert cat["phase"] == "post-import" and f"{name}.mod" in cat["already_loaded"]


def test_a_late_module_not_in_the_catalog_is_found(package):
    name, pkg = package
    cat = C.catalog(pkg, name, "invented time", "invented protection")
    (pkg / "late.py").write_text("x = 1\n")
    m = load(name, "late")
    assert C.verify_loaded(cat, [m]) == [f"{name}.late: late.py is not in the catalog"]


def test_source_drift_is_found_after_import(package):
    name, pkg = package
    cat = C.catalog(pkg, name, "invented time", "invented protection")
    m = load(name, "mod")
    (pkg / "mod.py").write_text("def answer():\n    return 2\n")
    assert C.verify_loaded(cat, [m]) == [f"{name}.mod: mod.py differs from the catalog"]
    assert C.drift(cat) == ["mod.py"]
    assert m.answer() == 1                                              # the loaded code is still the old code


def test_uncatalogued_bytecode_beside_a_source_is_found(package):
    name, pkg = package
    cat = C.catalog(pkg, name, "invented time", "invented protection")  # no .pyc exists yet
    m = load(name, "mod", bytecode=True)                                # importing writes one
    assert any("cached bytecode is not in the catalog" in p for p in C.verify_loaded(cat, [m]))


def test_changed_catalogued_bytecode_is_found(package):
    name, pkg = package
    first = load(name, "mod", bytecode=True)
    cached = Path(first.__spec__.cached)
    del sys.modules[f"{name}.mod"]
    cat = C.catalog(pkg, name, "invented time", "invented protection")
    assert any(k.endswith(".pyc") for k in cat["bytecode"])
    cached.write_bytes(cached.read_bytes() + b"\0")
    m = load(name, "mod", bytecode=False)
    problems = C.verify_loaded(cat, [m])
    assert any("cached bytecode differs from the catalog" in p for p in problems)
    assert any(r.endswith(".pyc") for r in C.drift(cat))


def test_a_module_from_outside_the_package_is_found(package, tmp_path):
    name, pkg = package
    cat = C.catalog(pkg, name, "invented time", "invented protection")
    assert C.verify_loaded(cat, [importlib.import_module("json")]) == [
        "json: loaded from outside the catalogued package"]


def test_a_change_reverted_between_checks_is_not_found(package):
    """A stated limit, kept as a probe: endpoint checks cannot see a transient change."""
    name, pkg = package
    cat = C.catalog(pkg, name, "invented time", "invented protection")
    original = (pkg / "mod.py").read_bytes()
    (pkg / "mod.py").write_text("def answer():\n    return 99\n")
    (pkg / "mod.py").write_bytes(original)
    assert C.drift(cat) == []


def test_a_catalog_must_say_when_and_how_it_was_protected(package):
    name, pkg = package
    with pytest.raises(C.CaptureError):
        C.catalog(pkg, name, "", "invented protection")
    with pytest.raises(C.CaptureError):
        C.catalog(pkg / "missing", name, "t", "p")


def test_a_real_pre_import_catalog_admits_a_matching_start(package, tmp_path):
    """End to end with invented values: the catalog of a temporary package, a manifest pinning those files, and a
    gateway whose start is admitted only with that evidence."""
    import telohelm.runtime.exchange as X; import telohelm.runtime.manifest as F; import telohelm.runtime.models as M; import telohelm.runtime.startup as S
    from telohelm.runtime.ledger import Ledger
    from telohelm.runtime.trusted_log import TrustedLog
    from .test_exchange_join import Wire
    from .test_local_adapter import ROLES; from .test_local_adapter import TIMEOUTS
    from .test_manifest import invented
    name, pkg = package
    cat = C.catalog(pkg, name, "invented time", "invented read-only install")
    m = load(name, "mod")
    manifest = invented(); manifest["runtime"]["code_sha256"] = dict(cat["files"])
    (tmp_path / "expected").write_text(F.fingerprint(manifest))
    config = F.checked(manifest, tmp_path / "expected")
    want = S.expected(config)
    rec = S.record(config.fingerprint, "before the first model call",
                   {k: S.observed(want[k], "invented", "t", "p") for k in S.ITEMS})
    a = M.LocalChatProvider("http://127.0.0.1:9", ROLES, Wire(), TIMEOUTS, exchanges=X.ExchangeStore(tmp_path / "x"),
                            config=config)
    gw = M.ModelGateway("e", "c", "s", a, TrustedLog(tmp_path / "log", tmp_path / "heads", "r"), Ledger(),
                        tmp_path / "j")
    assert gw.record_start(rec, S.RunAuthority(config, C.evidence(cat, [m]))) == []
    assert gw.submit("small_model", [("user", "q")], 256, 1, 10, 10, 2.4)["status"] == "answered"
