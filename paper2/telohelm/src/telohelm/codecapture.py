"""Code provenance captured before the runtime is imported (Codex's SG96.5 and SG96.6). CPU draft for cross-review,
TURN 97; tested only with harmless temporary modules. It imports nothing from study2.runtime, so a trusted bootstrap can
run it before the application is loaded.

catalog() hashes every source file under a package directory, and every cached bytecode file Python could load instead,
and notes which modules of that package were already imported when it ran. If any were, the capture is post-import and
weaker: it shows what is on disk now, not what was loaded. verify_loaded() then checks the modules a process actually
imported against the catalog: each must come from a catalogued source file whose content is unchanged, and any cached
bytecode beside it must be catalogued and unchanged too, since Python may load a .pyc instead of the source. drift()
re-hashes the catalogued files.

What this cannot show. Endpoint checks miss a change made and then reverted between them (a transient change), and
nothing here stops a hostile change made inside the same process. Hashes of files corroborate which code was installed;
a claim about executed code also needs the installation to be read-only to the run (a premise, not checked here), the
catalog to be taken before import, and the loaded modules and bytecode to be verified against it.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

PHASES = ("pre-import", "post-import")


class CaptureError(ValueError):
    """The capture cannot be made as asked."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def catalog(package_dir: Path, package: str, captured_at: str, protection: str) -> dict[str, Any]:
    """Hash every .py file and every cached .pyc under package_dir, and record whether any module of the package was
    already imported. Keys are paths relative to package_dir."""
    base = Path(package_dir).resolve()
    if not base.is_dir():
        raise CaptureError(f"{base} is not a directory")
    if not captured_at.strip() or not protection.strip():
        raise CaptureError("a catalog must say when it was captured and how its source is protected")
    loaded = sorted(n for n, m in sys.modules.items()
                    if m is not None and (n == package or n.startswith(package + ".")))
    files = {str(p.relative_to(base)): _sha(p) for p in sorted(base.rglob("*.py"))}
    bytecode = {str(p.relative_to(base)): _sha(p) for p in sorted(base.rglob("*.pyc"))}
    return {"package": package, "base": str(base), "phase": "post-import" if loaded else "pre-import",
            "already_loaded": loaded, "files": files, "bytecode": bytecode, "captured_at": captured_at,
            "protection": protection}


def verify_loaded(cat: dict, modules: list[ModuleType]) -> list[str]:
    """Problems with the modules a process imported, against the catalog. Empty means every module came from a
    catalogued, unchanged source file with no uncatalogued or changed bytecode beside it."""
    base = Path(cat["base"])
    problems = []
    for m in modules:
        f = getattr(m, "__file__", None)
        if not f:
            problems.append(f"{m.__name__}: no source file")
            continue
        path = Path(f).resolve()
        try:
            rel = str(path.relative_to(base))
        except ValueError:
            problems.append(f"{m.__name__}: loaded from outside the catalogued package")
            continue
        if rel not in cat["files"]:
            problems.append(f"{m.__name__}: {rel} is not in the catalog")
        elif not path.exists() or _sha(path) != cat["files"][rel]:
            problems.append(f"{m.__name__}: {rel} differs from the catalog")
        cached = getattr(getattr(m, "__spec__", None), "cached", None)
        if cached and Path(cached).exists():
            try:
                crel = str(Path(cached).resolve().relative_to(base))
            except ValueError:
                crel = None
            if crel is None or crel not in cat["bytecode"]:
                problems.append(f"{m.__name__}: cached bytecode is not in the catalog")
            elif _sha(Path(cached)) != cat["bytecode"][crel]:
                problems.append(f"{m.__name__}: cached bytecode differs from the catalog")
    return problems


def drift(cat: dict) -> list[str]:
    """Catalogued files whose content has changed, or that have gone, since the catalog was taken. A change followed
    by its reversal between two checks is not found."""
    base = Path(cat["base"])
    out = []
    for kind in ("files", "bytecode"):
        for rel, digest in sorted(cat[kind].items()):
            p = base / rel
            if not p.exists() or _sha(p) != digest:
                out.append(rel)
    return out


def loaded(cat: dict, modules: list[ModuleType]) -> list[dict]:
    """How each module was loaded: its catalogued source path, its loader's class and any cached bytecode path beside it
    (TURN 99, Codex's SG98.CODE.BYTECODE). A timestamp-valid .pyc is executed instead of the source, so whether one
    existed, and which, is part of what ran."""
    base = Path(cat["base"])
    out = []
    for m in modules:
        def rel(p):
            try:
                return str(Path(p).resolve().relative_to(base))
            except (TypeError, ValueError):
                return None
        spec = getattr(m, "__spec__", None)
        cached = getattr(spec, "cached", None)
        out.append({"module": m.__name__, "file": rel(getattr(m, "__file__", None)),
                    "loader": type(getattr(spec, "loader", None)).__name__,
                    "cached": rel(cached) if cached and Path(cached).exists() else None})
    return out


def evidence(cat: dict, modules: list[ModuleType]) -> dict[str, Any]:
    """The code evidence a start record carries: the catalog's phase, when and how it was captured, the package base,
    the source and bytecode hashes, how each module was loaded, and the problems found when the loaded modules are
    checked against the catalog. Admission needs a pre-import phase, no problems, source and bytecode equal to what the
    frozen configuration pins, and every module loaded from a pinned source with no unpinned bytecode."""
    return {"phase": cat["phase"], "captured_at": cat["captured_at"], "protection": cat["protection"],
            "base": cat["base"], "files": dict(cat["files"]), "bytecode": dict(cat["bytecode"]),
            "loaded": loaded(cat, modules),
            "problems": verify_loaded(cat, modules) + [f"drift: {r}" for r in drift(cat)]}
