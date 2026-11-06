"""The cold-loading entry point (TURN 109, a draft for cross-review against Codex's CB106 and its TURN 108 boundary).

Importing study2.bootstrap loads the study2 package, which is an ancestor of study2.runtime, so the bootstrap could not
cold-load the runtime it was written for. This launcher avoids that. It is run as a script, or loaded by file path, from
a helper directory installed separately from the target application, and it loads bootstrap.py and codecapture.py from
that directory by file path. The study2 package initializer therefore runs only after the bootstrap's preflight has
authorised it as a pinned ancestor of the runtime.

load() checks the helper files against pinned SHA-256 digests before executing them, when the caller supplies the pins
from an independently protected record. Without pins the helpers are trusted by premise. This launcher itself, the
interpreter, its standard library, import hooks and search path are trusted by premise: they have run before any check
here could.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import ModuleType

HELPERS = ("bootstrap.py", "codecapture.py")


class LaunchError(RuntimeError):
    """The helpers cannot be loaded as pinned."""


def _exec_source(name: str, path: Path, source: bytes) -> ModuleType:
    """Run exactly these source bytes as a new module (BOOT110.HELPER.CACHE): no loader, so no cached bytecode is ever
    consulted, and the bytes run are the bytes that were checked. On any failure the half-made module is removed, so
    it cannot be reused silently."""
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(source, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException as e:
        sys.modules.pop(name, None)
        raise LaunchError(f"{path.name} failed to initialise: {type(e).__name__}") from e
    return module


def load(directory: Path | None = None, pins: dict[str, str] | None = None) -> ModuleType:
    """The bootstrap module, run from the helper source bytes in directory (default: this file's directory) after
    checking them against pins, if given. Each helper's bytes are read once, checked, compiled and run; no cached
    bytecode is used. The loaded generation is recorded: asking again for the same helpers returns the same module,
    and asking for different ones in the same process is refused (BOOT110.HELPER.LOADED). The bootstrap is bound to
    the capture helper loaded here. Whether the target is still cold is checked by bootstrap.cold_start."""
    d = (Path(directory) if directory is not None else Path(__file__).parent).resolve()
    sources = {name: (d / name).read_bytes() for name in HELPERS}
    digests = {name: hashlib.sha256(b).hexdigest() for name, b in sources.items()}
    if pins is not None:
        if set(pins) != set(HELPERS):
            raise LaunchError(f"pins must name exactly {HELPERS}")
        for name in HELPERS:
            if digests[name] != pins[name]:
                raise LaunchError(f"{name} is not the pinned helper")
    identity = {"directory": str(d), "digests": digests}
    existing = sys.modules.get("telohelm_bootstrap")
    if existing is not None:
        if getattr(existing, "__telohelm_identity__", None) == identity:
            return existing
        raise LaunchError("a different generation of the helpers is already loaded in this process")
    if "telohelm_codecapture" in sys.modules:
        raise LaunchError("a capture helper is already loaded without its bootstrap; start a fresh process")
    capture = _exec_source("telohelm_codecapture", d / "codecapture.py", sources["codecapture.py"])
    try:
        boot = _exec_source("telohelm_bootstrap", d / "bootstrap.py", sources["bootstrap.py"])
    except LaunchError:
        sys.modules.pop("telohelm_codecapture", None)
        raise
    if getattr(boot, "C", None) is not capture:
        for n in ("telohelm_bootstrap", "telohelm_codecapture"):
            sys.modules.pop(n, None)
        raise LaunchError("the bootstrap is not bound to the capture helper loaded with it")
    boot.__telohelm_identity__ = identity
    return boot
