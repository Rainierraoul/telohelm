"""Make the scored freeze record (protocol amendment 7, item 6) and verify it the way the runner will.

    python -I -B make_freeze.py --repo REPO --profile PROFILE.json --out FREEZE.json

PROFILE is the shared flight profile to be frozen, as in a plan's "profile", with each model server named by its role
({small_model}, {reasoning_model}) because addresses change with every job. The record holds exactly the runner's
five keys: the protocol file's path and digest, the generator digest and the profile digest, with frozen true. The
source commit and the analysis code it binds go in the companion record beside it, because the runner refuses any
other key. Nothing is drawn, prepared or flown.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROTOCOL = "paper2/protocol/EXPERIMENT-2-PROTOCOL.md"


def make(repo: Path, profile: dict) -> tuple[dict, dict]:
    # The generator digest covers Telohelm's world modules, so they must be the committed source, never an installed
    # copy of another version; the scored job's wheel is built from the same commit and has the same bytes.
    sys.path[:0] = [str(repo / "paper2/telohelm/src"), str(repo)]
    import telohelm
    from paper2.experiment2 import study_runner as S
    if not Path(telohelm.__file__).resolve().is_relative_to(repo / "paper2/telohelm/src"):
        raise ValueError(f"telohelm was imported from {telohelm.__file__}, not from the repository's source")
    record = S._profile_record(profile, executable=False)
    freeze = {"frozen": True, "protocol_path": PROTOCOL,
              "protocol_sha256": hashlib.sha256((repo / PROTOCOL).read_bytes()).hexdigest(),
              "generator_sha256": S.generator_sha256(), "profile_sha256": S.profile_digest(record)}
    S.verify_freeze(freeze, repo)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--", "paper2"], text=True)
    if dirty.strip():
        raise ValueError("commit paper2 first: the freeze binds committed source only")
    analysis = repo / "paper2/experiment2/analysis.py"
    companion = {"source_commit": commit, "freeze_sha256": hashlib.sha256(
                     (json.dumps(freeze, indent=1, sort_keys=True) + "\n").encode()).hexdigest(),
                 "analysis_sha256": hashlib.sha256(analysis.read_bytes()).hexdigest(),
                 "profile": record, "scope": "scored streams 4301 to 4304, conditions L, C, F and C-full (amendment 7)"}
    return freeze, companion


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    freeze, companion = make(a.repo.resolve(), json.loads(a.profile.read_text()))
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.write_text(json.dumps(freeze, indent=1, sort_keys=True) + "\n")
    a.out.with_name(a.out.stem + "-companion.json").write_text(json.dumps(companion, indent=1, sort_keys=True) + "\n")
    print(json.dumps(freeze, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
