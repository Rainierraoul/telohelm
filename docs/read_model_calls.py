"""Read four published Telohelm summaries. Supply the original shared ZIP from Zenodo."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


ARCHIVE_SHA256 = "b915f74748a44486991a57489d814e87c87d7549263bb98da5c08db6fdde0560"
ANALYSIS_SHA256 = "415fd80d33371dd5bdecffe69b6bd7c63cbf274f32d8a01a7f1ed072e10ba818"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    path = parser.parse_args().archive
    with path.open("rb") as source:
        if hashlib.sha256(source.read()).hexdigest() != ARCHIVE_SHA256:
            raise SystemExit("The ZIP differs from the published archive.")
        source.seek(0)
        with ZipFile(source) as archive:
            raw = archive.read("frozen-analysis/analysis.json")
    if hashlib.sha256(raw).hexdigest() != ANALYSIS_SHA256:
        raise SystemExit("The analysis differs from the published digest.")
    study = json.loads(raw)
    rows = []
    for stream in study["per_stream"]:
        phases = stream["rq1"]["L"]["by_phase"].values()
        rows.append((stream["stream"], sum(p["positions"] for p in phases),
                     sum(p["cost"]["model_calls"] for p in phases),
                     sum(p["outcomes"].get("met_alone", 0) for p in phases)))
    columns = ((2, "flight model calls"), (3, "goals met alone"))
    if [r[0] for r in rows] != study["analysed_streams"]:
        raise SystemExit("The stream order differs from the summary.")
    for index, label in columns:
        summary = study["across"][f"RQ1 L {label} on matched positions"]
        if [r[index] for r in rows] != summary["per_stream"]:
            raise SystemExit(f"The phase counts disagree with the summary for {label}.")
    print("stream  matched positions  flight model calls  goals met alone")
    for stream, positions, calls, alone in rows:
        print(f"{stream:6}  {positions:17}  {calls:18}  {alone:15}")


if __name__ == "__main__":
    main()
