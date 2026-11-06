"""The launcher's CPU stand-in rehearsal: prepare its development study and write its plan, then summarise a run.

    python rehearse_cpu.py prepare --study DIR        (declares and prepares streams 4493 and 4495, phases 1 to 3)
    python rehearse_cpu.py plan --study DIR           (writes plan-cpu-rehearsal.json binding that study's manifest)
    python rehearse_cpu.py summarise --out OUT        (the launcher's result and the host's receipts, in brief)

The study uses unreserved development streams and seeds, so it can be rebuilt exactly. The rehearsal flies stand-in
models only and measures nothing about real models."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STREAMS, C_SEED, ADVISER_SEED, PHASES = (4493, 4495), 6601, 6401, (1, 2, 3)
STAND_IN = {"stand_in": True, "local": None, "learn": None, "small_interval_s": 30, "watch_briefing": True,
            "model_report_control": True, "learn_candidates": 3, "learn_regression": 8}


def study_manifest(study):
    """The same manifest as job.study_manifest: every file by relative path with its SHA-256; links are refused."""
    files = {}
    for path in sorted(Path(study).rglob("*")):
        if path.is_symlink():
            raise ValueError("The study holds no links")
        if path.is_file():
            files[str(path.relative_to(study))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def manifest_sha(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare(study):
    from paper2.experiment2 import study_runner as S
    S.declare_study(study, list(STREAMS))
    for s in STREAMS:
        S.prepare(Path(study) / str(s), s, c_seed=C_SEED, adviser_seed=ADVISER_SEED, phases=PHASES)
    return manifest_sha(study_manifest(study))


def plan(study):
    template = json.loads((HERE / "plan-template.json").read_text())
    body = {**template, "scope": "CPU stand-in rehearsal of the study launcher on development streams 4493 and 4495, "
                                  "phases 1 to 3; software only, measuring nothing about real models",
            "status": "cpu rehearsal", "arms": ["L", "C", "F", "calculator", "adaptive_controller"],
            "study_manifest_sha256": manifest_sha(study_manifest(study)), "profile": None,
            "rehearsal_profile": STAND_IN, "rehearsal_max_steps": 10, "host_hours": 1,
            "limits": {**template["limits"], "dispatchers": len(STREAMS) * 5}}
    (HERE / "plan-cpu-rehearsal.json").write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
    return body["study_manifest_sha256"]


def summarise(out):
    out = Path(out)
    result = json.loads((out / "result.json").read_text())
    receipts = json.loads((out / "host" / "host-receipts.json").read_text())
    return {"status": result["status"], "mode": result["mode"], "stages": result["stages"],
            "skipped": result["skipped"], "host_exit": result.get("host_exit"), "finalisation": result["finalisation"],
            "dispatcher_sweep": result["dispatcher_sweep"], "worker_sweep_exit": result["worker_sweep"]["exit"],
            "host_reason": receipts["reason"], "verdict": receipts["verdict"],
            "dispatch": {d["name"]: {"status": d["dispatch_status"], "exit": d["exit"], "counts": d["counts"]}
                         for d in receipts["dispatch"]},
            "checksums_complete": json.loads((out / "checksums.json").read_text())["complete"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("prepare", "plan"):
        sub.add_parser(name).add_argument("--study", type=Path, required=True)
    sub.add_parser("summarise").add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps(prepare(a.study) if a.mode == "prepare" else plan(a.study) if a.mode == "plan"
                     else summarise(a.out), indent=1, default=str))
