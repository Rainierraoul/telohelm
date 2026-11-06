"""Reproduce paper 2's result files from the downloaded data archive.

    python paper2/release/reproduce_tables.py --data <folder with the unzipped data archive> --out <new folder> [--check]

It rebuilds four files: the two result-table files of the four scored runs, and the two files of the condition in which
the reasoning model flies alone (the adviser): the output of paper2/experiment2/analysis_adviser.py and the output of
the adviser timing reader.

It sits at paper2/release/ in the public export and in the private repository alike, and finds the readers at
paper2/reproduction/ beside it, so it needs no flat-layout assumption.

The two readers, results-tables-claude-20261103.py and flight-records-claude-20261103.py, are not changed. They find
their inputs from their own location: each takes the folder two levels above itself as the root and reads
`cluster-records/scored-4304-20261102/terminal/...` below it. This script builds a small temporary root with that
layout and runs the readers from there:

- paper2/reproduction/ holds copies of the two readers and the four terminal audit reports. They are copies, not
  links, because Python resolves a link before the readers compute their root.
- the study folder and the frozen analysis output are linked to the downloaded data, so nothing large is copied.

Each reader writes its JSON file beside itself in the temporary root. This script copies the two files to --out and
deletes the temporary root. The flight-record reader reads the table file that the first reader has just written, as
it does in the private repository.

--data can be the folder where all data ZIPs were unzipped together (it holds scored-4301-4304-20261029/ and
frozen-analysis/), or a private `terminal/` record folder (it holds originals/scored-4301-4304-20261029/).
The frozen analysis output is read from the copy kept beside the readers (byte-identical to the original, with the
SHA-256 that the readers check), and from the data folder only if that copy is absent. If the data archive was built
with path substitution, its copy differs from the original and the readers would refuse it.

The adviser's records unzip to jobs/4546313/ (its study files, the job folder run/ and the rest of the preserved job).
analysis_adviser.py reads one study root that holds both the four runs and the adviser, so this script builds that root
from links: folders found on one side only are linked whole, and a file found on both sides must be byte-identical (30
are). It then runs `python -m paper2.experiment2.analysis_adviser` from the repository root, unchanged. The timing reader
records the relative paths it was given, so it runs from a temporary root where the adviser's study/ and run/ sit at
the paths it was first run from. --skip-adviser leaves the adviser out; --adviser names its folder if it is elsewhere.

--check compares each output with the committed JSON beside the readers, byte for byte, and exits 1 if any differs.
Linking needs symbolic links, which Windows allows only with developer mode or administrator rights.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
TABLES = "results-tables-claude-20261103"
FLIGHTS = "flight-records-claude-20261103"
AUDIT_REPORTS = ("scored-4339375-terminal-codex-20261030.json", "scored-4403902-terminal-codex-20261101.json",
                 "scored-4499845-terminal-codex-20261102.json", "scored-4503929-terminal-codex-20261103.json")
FROZEN_COPY = "frozen-analysis-output-20261103.json"
STUDY = "scored-4301-4304-20261029"
PRIVATE_TERMINAL = Path("cluster-records/scored-4304-20261102/terminal")
CODE_ROOT = HERE.parent.parent  # the repository root, which holds paper2/experiment2/
ADVISER_JOB = "4546313"
ADVISER_ANALYSIS = "adviser-analysis-output-codex-20261105.json"
ADVISER_TIMING = "adviser-slot-timing-codex-20261105"
# Where the timing reader was first run from, relative to the private repository root.
ADVISER_ORIGINALS = Path("cluster-records/adviser-20261103/amendment2/scored-terminal-20261105/originals")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_readers(given=None):
    for folder in ([Path(given)] if given else [HERE, HERE.parent / "reproduction"]):
        if (folder / f"{TABLES}.py").is_file() and (folder / f"{FLIGHTS}.py").is_file():
            return folder.resolve()
    raise SystemExit("The two readers were not found. Pass --readers <folder holding them>.")


def find_inputs(data, readers):
    data = Path(data).resolve()
    for base in (data, data / "originals"):
        if (base / STUDY).is_dir():
            study = base / STUDY
            break
    else:
        raise SystemExit(f"No {STUDY}/ folder in {data} or {data / 'originals'}. Unzip all data archives into one folder.")
    # The readers check the output against a pinned SHA-256. The copy beside the readers is byte-identical to the
    # original. The data archive's copy is too unless the archive was built with path substitution, which they refuse.
    analysis = readers / FROZEN_COPY
    if not analysis.is_file():
        analysis = data / "frozen-analysis" / "analysis.json"
    if not analysis.is_file():
        raise SystemExit("The frozen analysis output is in neither the readers' folder nor the data folder.")
    return study, analysis


def run_reader(root, name):
    script = root / "paper2/reproduction" / f"{name}.py"
    done = subprocess.run([sys.executable, "-I", "-B", str(script)], cwd=root, capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f"{name}.py failed with exit code {done.returncode}\n{done.stdout}{done.stderr}")


def reproduce(data, out, readers=None, check=False, expected=None, keep_work=False, quiet=False):
    readers = find_readers(readers)
    study, analysis = find_inputs(data, readers)
    out = Path(out).resolve()
    names = [f"{TABLES}.json", f"{FLIGHTS}.json"]
    if any((out / n).exists() for n in names):
        raise SystemExit(f"{out} already holds an output file; choose a new folder.")
    root = Path(tempfile.mkdtemp(prefix="telohelm-tables-")).resolve()
    try:
        repro = root / "paper2/reproduction"
        repro.mkdir(parents=True)
        for name in (f"{TABLES}.py", f"{FLIGHTS}.py", *AUDIT_REPORTS):
            shutil.copyfile(readers / name, repro / name)
        terminal = root / PRIVATE_TERMINAL
        (terminal / "originals").mkdir(parents=True)
        (terminal / "frozen-analysis").mkdir()
        (terminal / "originals" / STUDY).symlink_to(study)
        (terminal / "frozen-analysis" / "analysis.json").symlink_to(analysis)
        run_reader(root, TABLES)
        run_reader(root, FLIGHTS)
        out.mkdir(parents=True, exist_ok=True)
        results = {}
        for name in names:
            shutil.copyfile(repro / name, out / name)
            results[name] = {"sha256": sha256(out / name), "bytes": (out / name).stat().st_size}
        results["readers"] = {n: sha256(readers / n) for n in (f"{TABLES}.py", f"{FLIGHTS}.py")}
        if check:
            committed = Path(expected).resolve() if expected else readers
            for name in names:
                want = committed / name
                results[name]["committed_sha256"] = sha256(want)
                results[name]["identical_to_committed"] = (out / name).read_bytes() == want.read_bytes()
        if not quiet:
            for name in names:
                line = f"{name}  {results[name]['sha256']}"
                if check:
                    line += "  identical to committed" if results[name]["identical_to_committed"] else \
                        f"  DIFFERS from committed ({results[name]['committed_sha256']})"
                print(line)
        return results
    finally:
        if keep_work:
            print(f"Temporary root kept at {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)


def find_adviser(data, given=None):
    data = Path(data).resolve()
    for base in ([Path(given)] if given else [data / "jobs" / ADVISER_JOB, data / ADVISER_ORIGINALS]):
        if (base / "study").is_dir() and (base / "run").is_dir():
            return base.resolve()
    raise SystemExit(f"No adviser records (study/ and run/) in {data / 'jobs' / ADVISER_JOB}. Unzip the adviser ZIP with "
                     "the others, pass --adviser <folder>, or pass --skip-adviser.")


def merge_links(target, first, second):
    """A folder of links that holds both trees. A file on both sides must be byte-identical."""
    target.mkdir()
    for name in sorted({p.name for p in first.iterdir()} | {p.name for p in second.iterdir()}):
        a, b = first / name, second / name
        if a.exists() and b.exists():
            if a.is_dir() and b.is_dir():
                merge_links(target / name, a, b)
            elif a.is_file() and b.is_file() and a.read_bytes() == b.read_bytes():
                (target / name).symlink_to(a)
            else:
                raise SystemExit(f"The four runs and the adviser records differ at {a.relative_to(first.parent)}")
        else:
            (target / name).symlink_to(a if a.exists() else b)


def run_command(command, cwd, what):
    done = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f"{what} failed with exit code {done.returncode}\n{done.stdout}{done.stderr}")


def reproduce_adviser(data, out, readers=None, adviser=None, check=False, expected=None, keep_work=False, quiet=False,
                      code_root=CODE_ROOT):
    readers = find_readers(readers)
    study, _ = find_inputs(data, readers)
    base = find_adviser(data, adviser)
    out = Path(out).resolve()
    names = [ADVISER_ANALYSIS, f"{ADVISER_TIMING}.json"]
    if any((out / n).exists() for n in names):
        raise SystemExit(f"{out} already holds an output file; choose a new folder.")
    root = Path(tempfile.mkdtemp(prefix="telohelm-adviser-")).resolve()
    try:
        merged = root / "merged-study"
        merge_links(merged, study, base / "study")
        run_command([sys.executable, "-E", "-B", "-m", "paper2.experiment2.analysis_adviser", "--study", str(merged),
                     "--out", str(root / "analysis")], Path(code_root).resolve(), "analysis_adviser")
        originals = root / ADVISER_ORIGINALS
        originals.mkdir(parents=True)
        (originals / "study").symlink_to(base / "study")
        (originals / "run").symlink_to(base / "run")
        reader = root / "paper2/reproduction" / f"{ADVISER_TIMING}.py"
        reader.parent.mkdir(parents=True)
        shutil.copyfile(readers / f"{ADVISER_TIMING}.py", reader)
        run_command([sys.executable, "-I", "-B", str(reader), "--study", str(ADVISER_ORIGINALS / "study"),
                     "--run", str(ADVISER_ORIGINALS / "run"), "--out", str(root / "timing.json")], root,
                    f"{ADVISER_TIMING}.py")
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / "analysis" / "analysis-adviser.json", out / ADVISER_ANALYSIS)
        shutil.copyfile(root / "timing.json", out / f"{ADVISER_TIMING}.json")
        results = {name: {"sha256": sha256(out / name), "bytes": (out / name).stat().st_size} for name in names}
        results["adviser_records"] = str(base)
        if check:
            committed = Path(expected).resolve() if expected else readers
            for name in names:
                want = committed / name
                results[name]["committed_sha256"] = sha256(want)
                results[name]["identical_to_committed"] = (out / name).read_bytes() == want.read_bytes()
        if not quiet:
            for name in names:
                line = f"{name}  {results[name]['sha256']}"
                if check:
                    line += "  identical to committed" if results[name]["identical_to_committed"] else \
                        f"  DIFFERS from committed ({results[name]['committed_sha256']})"
                print(line)
        return results
    finally:
        if keep_work:
            print(f"Temporary root kept at {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="Folder with the unzipped data archive")
    parser.add_argument("--out", type=Path, help="Folder for the two reproduced JSON files (created if absent)")
    parser.add_argument("--check", action="store_true", help="Compare each output with the committed JSON byte for byte")
    parser.add_argument("--readers", type=Path, help="Folder holding the readers, audit reports and committed JSON")
    parser.add_argument("--expected", type=Path, help="Folder holding the committed JSON, if not beside the readers")
    parser.add_argument("--keep-work", action="store_true", help="Keep the temporary root for inspection")
    parser.add_argument("--adviser", type=Path, help="Folder holding the adviser job's study/ and run/, if not jobs/4546313")
    parser.add_argument("--skip-adviser", action="store_true", help="Reproduce only the two table files")
    args = parser.parse_args()
    scratch = None
    if args.out is None:
        if not args.check:
            parser.error("--out is required unless --check is given")
        scratch = tempfile.mkdtemp(prefix="telohelm-tables-out-")
        args.out = Path(scratch) / "out"
    try:
        results = reproduce(args.data, args.out, args.readers, args.check, args.expected, args.keep_work)
        names = [f"{TABLES}.json", f"{FLIGHTS}.json"]
        if not args.skip_adviser:
            results.update(reproduce_adviser(args.data, args.out, args.readers, args.adviser, args.check, args.expected,
                                             args.keep_work))
            names += [ADVISER_ANALYSIS, f"{ADVISER_TIMING}.json"]
    finally:
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)
    if args.check and not all(results[n]["identical_to_committed"] for n in names):
        sys.exit(1)


if __name__ == "__main__":
    main()
