"""A cold bootstrap and gateway factory for a run (TURN 105, revised TURN 107; a draft for cross-review against Codex's
SG96.1 to 7 and CB106.1 to 5). It lives outside the runtime package and imports nothing from it until a preflight has
authorised every file the import would execute. Nothing in the study calls it yet, and it has been run only on a
temporary copy of the runtime with invented values for what needs the cluster.

cold_start():
1. refuses if any module of the runtime package or its ancestors is already imported (the cold-load premise);
2. preflight, before any application code runs (TURN 107, Codex's BOOT106 and CB106): checks the manifest against the
   frozen expected fingerprint with this module's own canonical encoding, not the application's; resolves the runtime
   package and each ancestor package by path, without executing them; requires each ancestor's __init__.py to match the
   manifest's runtime.ancestor_sha256 and to have no cached bytecode; requires the runtime package to resolve to the
   catalogued directory; and requires the catalogued source and bytecode to be exactly the pinned ones. A catalog alone
   is an observation, not an authorisation: later refusal cannot undo code that already ran;
3. imports the runtime with bytecode writing off, checks the manifest again with the application's validator
   (manifest.checked), and corroborates that each ancestor was loaded from the file the preflight authorised;
4. restores the protected roots from the registry file, which must exist (an empty file for a first run);
5. builds the code evidence from the catalog and the modules actually loaded;
6. builds the start record from an observer, which returns the nine items.
build_gateway() then makes a store-backed provider whose arguments all come from the checked configuration, a gateway
with its own trusted log, publishes that log as a root, and records the start.

The observer decides what is evidence. cpu_observer() observes, on this machine, the captured code files, the Python
version and the versions of named installed distributions. Everything else (prompts, weights, launch, model list,
server defaults, environment) is unknown unless the caller supplies an entry, and an unknown item refuses the start.
Entries made from the configuration itself are fixtures, not observations.

Premises, not checked here: the runtime is installed where the run cannot write, and nothing changes the files between
the preflight and the import (a time-of-check to time-of-use premise); this module, codecapture, the study2 package
initializer, the Python interpreter, its standard library, its import hooks and its search path are trusted and pinned
by other means (they have already run when cold_start is called, so this code cannot authorise them); the registry and
expected fingerprint files are in protected storage. The preflight resolves packages with the standard path finder; an
installed import hook that resolves them differently is outside this check, and the corroboration after import is
what would notice it. A run that stops between finding a problem and writing anything leaves no evidence; that needs an
independent checkpoint.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import json
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

try:
    from . import codecapture as C
except ImportError:
    # Run outside the study2 package (coldlaunch.py, TURN 109): use the capture helper the launcher checked and ran
    # from its source bytes just before this module (TURN 111, Codex's BOOT110). Without the launcher, the helper beside
    # this file is loaded by path instead, and is then trusted by premise.
    C = sys.modules.get("telohelm_codecapture")
    if C is None:
        import importlib.util as _util
        _spec = _util.spec_from_file_location("telohelm_codecapture", Path(__file__).with_name("codecapture.py"))
        C = _util.module_from_spec(_spec)
        sys.modules["telohelm_codecapture"] = C
        _spec.loader.exec_module(C)


class BootstrapError(RuntimeError):
    """The bootstrap cannot be made as its premises require."""


class Bootstrap:
    """What a cold start produced: the runtime modules it loaded, the checked configuration, the restored roots, the run
    authority and the start record. Read-only after it is made."""
    __slots__ = ("modules", "config", "roots", "authority", "record", "catalog")

    def __init__(self, **kw):
        for k in self.__slots__:
            object.__setattr__(self, k, kw[k])

    def __setattr__(self, name, value):
        raise AttributeError("a bootstrap cannot be changed")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def cpu_observer(dependencies: list[str] | None = None, supplied: dict[str, dict] | None = None,
                 protection: str = "the running interpreter and its installed distributions") -> Callable:
    """An observer for what a CPU machine can see itself; anything else is unknown unless supplied."""
    def observe(S: Any, config: Any, evidence: dict) -> dict:
        when = _now()
        deps = {}
        for name in (dependencies or []):
            try:
                deps[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                deps[name] = None
        items = {
            "installed_code": S.observed(dict(evidence["files"]), "codecapture catalog before import", when,
                                         evidence["protection"]),
            "python": S.observed(platform.python_version(), "platform.python_version()", when, protection),
            "dependencies": (S.observed(deps, "importlib.metadata", when, protection) if None not in deps.values()
                             else S.unknown("importlib.metadata: a distribution is not installed", when, protection)),
        }
        for name in S.ITEMS:
            if name not in items:
                items[name] = (supplied or {}).get(name) or S.unknown("not observable on this machine", when,
                                                                       "none: needs the cluster")
        return items
    return observe


def _canonical(obj: Any) -> bytes:
    """The manifest's canonical encoding, computed here so the preflight does not run application code."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(runtime_dir: Path, runtime_package: str, manifest: dict, expected_path: Path, captured_at: str,
              protection: str) -> dict:
    """Authorise, before any application import, every file the import of the runtime package would execute
    (CB106.1 to 3). Returns the catalog and the resolved ancestor origins; raises BootstrapError otherwise. Nothing in
    the application package runs here."""
    try:
        expected = Path(expected_path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as e:
        raise BootstrapError(f"no readable expected fingerprint at {expected_path}") from e
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise BootstrapError("the expected fingerprint is not one SHA-256")
    try:
        actual = hashlib.sha256(_canonical(manifest)).hexdigest()
    except (TypeError, ValueError) as e:
        raise BootstrapError("the manifest is not finite JSON") from e
    if actual != expected:
        raise BootstrapError("the manifest is not the frozen one")
    runtime = manifest.get("runtime") if isinstance(manifest, dict) else None
    if not isinstance(runtime, dict) or not isinstance(runtime.get("code_sha256"), dict):
        raise BootstrapError("the frozen manifest pins no runtime code")
    ancestors_pinned = runtime.get("ancestor_sha256")
    names = runtime_package.split(".")
    if not isinstance(ancestors_pinned, dict) or set(ancestors_pinned) != {".".join(names[:i]) for i in range(1, len(names))}:
        raise BootstrapError("the frozen manifest must pin exactly the runtime package's ancestors")
    # Resolve each package by path with the standard path finder, which locates files without executing them.
    search, origins = None, {}
    for i in range(1, len(names) + 1):
        name = ".".join(names[:i])
        spec = importlib.machinery.PathFinder.find_spec(name, search)
        if spec is None or not spec.origin or not spec.submodule_search_locations:
            raise BootstrapError(f"{name} does not resolve to a package on the search path")
        origin = Path(spec.origin).resolve()
        if i < len(names):
            if origin.name != "__init__.py" or _sha(origin) != ancestors_pinned[name]:
                raise BootstrapError(f"the ancestor {name} is not the pinned file")
            cached = list((origin.parent / "__pycache__").glob("__init__.*.pyc"))
            if cached:
                raise BootstrapError(f"the ancestor {name} has cached bytecode beside it, which could run instead")
            origins[name] = str(origin)
        elif origin.parent != Path(runtime_dir).resolve():
            raise BootstrapError(f"{name} resolves to {origin.parent}, not the catalogued {Path(runtime_dir).resolve()}")
        search = list(spec.submodule_search_locations)
    cat = C.catalog(Path(runtime_dir), runtime_package, captured_at, protection)
    if cat["phase"] != "pre-import":
        raise BootstrapError("the catalog was not taken before import")
    if cat["files"] != runtime["code_sha256"]:
        raise BootstrapError("the runtime source is not exactly the pinned source; refused before import")
    if cat["bytecode"] != runtime.get("bytecode_sha256", {}):
        raise BootstrapError("the runtime's cached bytecode is not exactly the pinned bytecode; refused before import")
    return {"catalog": cat, "ancestors": origins}


def cold_start(runtime_dir: Path, runtime_package: str, manifest: dict, expected_path: Path, registry: Path,
               observe: Callable, captured_at: str, protection: str) -> Bootstrap:
    names = runtime_package.split(".")
    family = {".".join(names[:i]) for i in range(1, len(names))}
    loaded_before = [n for n in sys.modules if n == runtime_package or n.startswith(runtime_package + ".") or n in family]
    if loaded_before:
        raise BootstrapError(f"the runtime or its ancestors are already imported ({sorted(loaded_before)[0]}); a cold "
                             "start needs a fresh interpreter")
    pre = preflight(runtime_dir, runtime_package, manifest, expected_path, captured_at, protection)
    cat = pre["catalog"]
    sys.dont_write_bytecode = True
    F = importlib.import_module(f"{runtime_package}.manifest")
    S = importlib.import_module(f"{runtime_package}.startup")
    M = importlib.import_module(f"{runtime_package}.models")
    X = importlib.import_module(f"{runtime_package}.exchange")
    TL = importlib.import_module(f"{runtime_package}.trusted_log")
    config = F.checked(manifest, Path(expected_path))
    for name, origin in pre["ancestors"].items():      # corroborate after loading (CB106.4)
        f = getattr(sys.modules.get(name), "__file__", None)
        if f is None or str(Path(f).resolve()) != origin:
            raise BootstrapError(f"the ancestor {name} was loaded from somewhere other than the authorised file")
    if not Path(registry).exists():
        raise BootstrapError(f"no protected roots registry at {registry}; a first run needs an empty one")
    roots = M.ProtectedRoots.restore(Path(registry))
    modules = [sys.modules[n] for n in sorted(sys.modules)
               if (n == runtime_package or n.startswith(runtime_package + ".")) and getattr(sys.modules[n], "__file__", None)]
    evidence = C.evidence(cat, modules)
    record = S.record(config.fingerprint, "cold bootstrap, before the first model call", observe(S, config, evidence))
    authority = S.RunAuthority(config, evidence, check_drift=True)
    return Bootstrap(modules={"manifest": F, "startup": S, "models": M, "exchange": X, "trusted_log": TL},
                     config=config, roots=roots, authority=authority, record=record, catalog=cat)


def build_gateway(boot: Bootstrap, episode: str, condition: str, stream: str, transport: Any, store: Path, log: Path,
                  heads: Path, journal: Path, registry: Path, ledger: Any = None, reportable: bool = True,
                  run_id: str = "telohelm run") -> tuple[Any, list[str]]:
    """A started gateway whose provider takes every setting from the checked configuration. Returns the gateway and
    its start problems; with problems the gateway refuses every fresh call."""
    M, X, TL = boot.modules["models"], boot.modules["exchange"], boot.modules["trusted_log"]
    c = boot.config
    t, roles, names, sampling = c.section("transport"), c.section("roles"), c.section("names"), c.section("sampling")
    provider = M.LocalChatProvider(t["origin"], roles["model"], transport, t["timeout_s"],
                                   temperature=sampling["temperature"], reportable=reportable,
                                   served_names_by_role={r: tuple(v) for r, v in names["served"].items()},
                                   exchanges=X.ExchangeStore(Path(store)), config=c, roots=boot.roots)
    if ledger is None:
        ledger = importlib.import_module(f"{M.__name__.rsplit('.', 1)[0]}.ledger").Ledger()
    gw = M.ModelGateway(episode, condition, stream, provider, TL.TrustedLog(Path(log), Path(heads), run_id), ledger,
                        Path(journal))
    boot.roots.register(Path(log), Path(heads))
    M.ProtectedRoots.record(Path(registry), Path(log), Path(heads))
    return gw, gw.record_start(boot.record, boot.authority)
