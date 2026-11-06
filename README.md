# Telohelm

This repository holds the software behind 'Telohelm: replacing repeated language-model decisions with tested procedures in a changing task' by Simon Vajda, Applied Artificial Intelligence Institute, Deakin University, and a guide to its study records, which are archived separately. Paper status: in preparation. It builds on an earlier paper by the author (Vajda, 2026; [preprint](https://doi.org/10.21203/rs.3.rs-11312211/v1), [code and data](https://doi.org/10.5281/zenodo.23225358)).

Telohelm is a self-adaptive agent for tasks in which the same decisions come up again and again. A reasoning model can make each decision, but each call costs tokens and a wait. Telohelm therefore asks the reasoning model, between episodes, to write a candidate procedure for one decision. An episode is one simulated run of the task, and a procedure is a short Python program that makes one decision from what the agent can observe.

A separate program, the monitor, replays recorded episodes with the candidate and makes no model calls, and it adopts the candidate only if the candidate meets its part of the goal in every replay. Adopted procedures form the library and decide with no model call, and where none applies Telohelm asks a small model, then the reasoning model and last a person. A cycle of observing, orienting, deciding and acting (OODA) runs inside each episode, and a MAPE-K loop runs between episodes.

The study task is a simulated air-to-air refuelling join, in which the aircraft makes two decisions: its heading and its radio report.

## Status

The four scored runs (A to D in the paper) are complete. So is the adviser condition, in which the reasoning model flies alone. It flew 100 episodes, 25 from each of the four streams. The paper's results come from these runs of two decisions in one simulated task, and the scored study is frozen. A stream is a generated schedule of scenarios, and new runs use development streams, not the four scored ones.

## Try it without a model

You need Python 3.12, and the commands run from the repository root with no GPU or model download.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r paper2/telohelm/requirements-tested.txt -e './paper2/telohelm[test]'
.venv/bin/python -m paper2.experiment2.run_stream --help
.venv/bin/python -m paper2.experiment2.run_stream --episodes 1 --max-steps 30 --out .local-runs/setup-check
```

The first two commands install the agent with its tested dependencies, and the third prints the options of the study runner. The fourth flies one episode of at most 30 steps with stand-ins in place of the models and writes its records to `.local-runs/setup-check`. It checks that the parts are wired together and measures nothing about model learning. The full study needs a GPU cluster and the two models below. Its job script is `paper2/experiment2/launch/run.sbatch` and its launcher is `paper2/experiment2/launch/job.py`.

To run the tests, use `.venv/bin/python -m pytest paper2/telohelm/tests paper2/experiment2/tests` from the repository root. The file `pytest.ini` there leaves out the tests that read files this repository does not include, and says why.

## What is in the repository

```text
docs/                       guide to the study records
paper2/telohelm/            agent, tanker task pack and simulator, tests
paper2/experiment2/         study runner, conditions, frozen analysis, cluster launch
paper2/protocol/            protocol, freeze records, two adviser amendments
paper2/reproduction/        frozen analysis output, adviser analysis output, table readers, audit readers
paper2/manuscript/figures/  scripts and sources for the two figures
paper2/release/             reproduce_tables.py
```

| To | Read |
|---|---|
| Learn how the agent works | Section 3 of the paper, then `paper2/telohelm/README.md` |
| Inspect the study design | `paper2/protocol/EXPERIMENT-2-PROTOCOL.md` and the files beside it |
| See how a run is made | `paper2/experiment2/launch/README.md`, and `paper2/experiment2/analysis.py` for the analysis |
| Open the study records, or check a run against its audit | [docs/DATA.md](docs/DATA.md) and the `scored-` and `adviser-` files in `paper2/reproduction/` |

## Reproduce the paper's tables

Download the six data ZIPs from the data archive (https://doi.org/10.5281/zenodo.23993339) and unzip all of them into one folder. That folder then holds `scored-4301-4304-20261029/`, `frozen-analysis/` and `jobs/`. After the setup above, run this from the repository root, with a new folder for `--out`:

```sh
.venv/bin/python paper2/release/reproduce_tables.py --data <that folder> --out <a new folder> --check
```

The script rebuilds four files from the records: the paper's two result-table files, and the two adviser files, which are the output of `paper2/experiment2/analysis_adviser.py` and the timing of each of the adviser's reasoning-model calls. With `--check` it compares all four byte for byte with the copies in `paper2/reproduction/` (`results-tables-claude-20261103.json`, `flight-records-claude-20261103.json`, `adviser-analysis-output-codex-20261105.json` and `adviser-slot-timing-codex-20261105.json`), and on the author's machine all four matched exactly. A match shows that the downloaded records give the four files exactly, and no flight is repeated. The run takes about 25 seconds and needs no model, GPU or simulation. The script uses symbolic links, which Windows restricts, and it builds from them the single study folder that `analysis_adviser.py` reads. The option `--skip-adviser` rebuilds only the two table files.

## The study records

The records of the four scored runs and the adviser condition are in a separate data archive at https://doi.org/10.5281/zenodo.23993339. They come as six ZIPs, about 2.9 GB compressed and 13.3 GB unzipped. The guide in [docs/DATA.md](docs/DATA.md) covers the run labels, the ZIPs and checksum files, the material held back from the agent during the study, the checks made before packaging and the audit readers.

## Models

The study used Qwen/Qwen3-4B-Instruct-2507 at revision `cdbee75f17c01a7cc42f958dc650907174af0554` as the small model and Qwen/Qwen3.8-27B at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` as the reasoning model, both served by vLLM. The weights are not included and come from the publishers, and three tokenizer files of the reasoning model are bundled in the repository.

## Licences

Code is released under the [MIT licence](LICENSE). Documentation, the protocol, figures, derived tables and the study records are released under [CC BY 4.0](LICENSE-data), and the three tokenizer files keep their Apache 2.0 licence ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) and [NOTICE](NOTICE)). The model replies in the study records were produced by the two Qwen models, which their publishers license separately.

## How to cite

GitHub's 'Cite this repository' button reads [CITATION.cff](CITATION.cff). The software is archived at https://doi.org/10.5281/zenodo.23993336 and the study records at https://doi.org/10.5281/zenodo.23993339. Please cite the paper as well.

## Use of AI

Claude Code (Anthropic) and Codex (OpenAI) generated and revised the software, protocols, analyses and documentation under the author's direction, and the author set the aims and is responsible for the claims. The Qwen models inside Telohelm are the experiment's subjects and are separate from these tools.
