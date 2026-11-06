# Study records

This guide describes the study records of the four scored runs, and of the adviser condition, behind 'Telohelm: replacing repeated language-model decisions with tested procedures in a changing task' (Simon Vajda). The records are in a data archive at https://doi.org/10.5281/zenodo.23993339 and are released under [CC BY 4.0](../LICENSE-data). The [README](../README.md) describes the software that reads them.

## Runs and ZIPs

The paper labels the four scored runs A to D. Each run flew 100 episodes from one stream, a generated schedule of scenarios, and each run was one cluster job. The adviser condition, in which the reasoning model flies alone, flew 100 episodes, 25 from each of the four streams, as one cluster job. Its design is in two adviser amendments in `paper2/protocol/`: `EXPERIMENT-2-AMENDMENT-ADVISER-20261103.md` defines the condition and `EXPERIMENT-2-AMENDMENT-ADVISER-2-20261104.md` changes it. The condition is coded in `paper2/experiment2/adviser.py`, its analysis in `paper2/experiment2/analysis_adviser.py` and its plan in `paper2/experiment2/launch/plan-scored-adviser2-20261104.json`. The records come as six ZIPs.

| ZIP | Run | Stream | Job | Files | Unzipped size |
|---|---|---|---|---|---|
| `run-A-stream-4301-job-4339375.zip` | A | 4301 | 4339375 | 5,067 | about 3.8 GB |
| `run-B-stream-4302-job-4403902.zip` | B | 4302 | 4403902 | 3,841 | about 2.8 GB |
| `run-C-stream-4303-job-4499845.zip` | C | 4303 | 4499845 | 3,920 | about 2.9 GB |
| `run-D-stream-4304-job-4503929.zip` | D | 4304 | 4503929 | 4,031 | about 3.7 GB |
| `adviser-job-4546313.zip` | adviser | 4301 to 4304 | 4546313 | 828 | about 53 MB |
| `shared-study-files-and-frozen-analysis.zip` | all | 4301 to 4304 | all four | 19 | about 19 MB |

Together that is 17,706 files, about 13.3 GB unzipped and about 2.9 GB compressed. The sizes in the table are rounded, so they do not add to the total exactly.

## What the ZIPs hold

Each run ZIP holds that stream's folder of the final study, with its schedule and the prepared scenarios and inputs. For each condition, meaning one way of flying the same scenarios, the folder has the episode records: observations, model requests and replies with token usage, actions, candidate procedures, the monitor's replays, adoptions and outcomes. The ZIP also holds the job's own provenance (installation, serving, native checks and checksums) and the source that was staged for the job. The shared ZIP holds `study.json`, the pause records and the frozen analysis output.

The adviser ZIP unzips to `jobs/4546313/`, which holds the job's records as they were preserved. `study/` has the adviser's episodes in the four stream folders and 30 study files that repeat files of the four runs byte for byte. `run/` is the job's own folder, with installation, serving, native checks, checksums and the model-slot records. `payload/` is the source staged for the job. The folder also has `handoff/`, `logs/`, `submission/` and `preserved-manifest.json`, which gives the SHA-256 and size of every other file in the folder.

## Unzipping

Unzip all six ZIPs into one folder, which then holds three folders. `scored-4301-4304-20261029/` is the final study, with the four stream folders, `study.json` and the pause records. `jobs/` holds each job's provenance and staged source, including `jobs/4546313/`, and `frozen-analysis/` holds the frozen analysis output. The script `paper2/release/reproduce_tables.py` takes this folder as its `--data` argument, and the README gives the command.

## Checksum files

Four files sit beside the ZIPs: this archive's README and three checksum files. `SHA256SUMS` gives the SHA-256 of the six ZIPs, the README, the manifest and the contents list, and `CONTENTS-SHA256SUMS.txt` gives it for every member of every ZIP. `DATA-MANIFEST.json` lists every file with its size, its SHA-256 and its treatment in packaging. Check the ZIPs against `SHA256SUMS` before unzipping, and the unzipped files against the other two afterwards.

## Byte-identical records

The records are published exactly as preserved: every file is byte-identical to the original, so every hash recorded in the records can still be checked. We edited nothing, because an edit would break the hashes recorded against the file.

Some files contain the author's user name in cluster folder paths, but a pattern scan found no credentials. The ten model-server logs, two for each of the five jobs, are left out of the ZIPs, and `DATA-MANIFEST.json` lists each with its size and SHA-256.

## Held back from the agent and published after the study

Three kinds of material in each stream's folder were held back from the agent during the study and are published now.

- `private/alternative-world/` holds the second world of each help case. A help case is built from a pair of worlds that look identical to the agent until help arrives, and the agent never sees the second one.
- `phase5-witness/` holds the reference pilot's runs with help. The reference pilot is the fixed controller that the designers supply.
- `inputs/F-snapshot/` holds the starting copy of the frozen library, which is the library of the learning agent kept fixed for comparison.

The code that labels the hidden set is published as well, in `paper2/experiment2/hidden_set/`, and it holds no cases.

## Checks made before packaging

We checked the preserved originals before packaging. The SHA-256 and size of every file in the final record set match the cluster's manifest, whose digest is `8f3eaee7b8799a460492583a7cea150536bd46774b8cc27a8b87a06153d92c29`, and each job's own checksums match as well. For the adviser, every file matched `preserved-manifest.json`, the digest of that file is `515fe4c0325dad761f198548f708094b9b528400878115ae6c948d9b044e1598`, and the job's own checksums and its staged source's manifest matched too.

## Frozen analysis output

The frozen analysis output, which has SHA-256 `415fd80d33371dd5bdecffe69b6bd7c63cbf274f32d8a01a7f1ed072e10ba818`, is in the shared ZIP and, byte-identical, in `paper2/reproduction/frozen-analysis-output-20261103.json`.

## Audit readers

Each scored job has a terminal audit reader and a report in `paper2/reproduction/`, in files whose names start with `scored-` and the job number. The readers for jobs 4403902, 4499845 and 4503929 run from the public layout, and they give the same output on the public records as on the originals. On a Mac they report many floating-point context differences, which their own documentation explains, so their committed terminal reports come from native Linux runs.

The reader for job 4339375 reads the private repository's history with `git show`, so it cannot run from this layout and is evidence only.

The adviser has its own files in `paper2/reproduction/`, with names that start with `adviser-`. `adviser-4546313-terminal-codex-20261105.json` is the terminal audit report of job 4546313. `adviser-analysis-output-codex-20261105.json` is the output of `analysis_adviser.py`, with SHA-256 `a7a07f268a59853d14ce5d84eaafcfa043e964024b89c66ea928945c0274f58a`. `adviser-slot-timing-codex-20261105.py` is the reader that times each of the adviser's reasoning-model calls, and `adviser-slot-timing-codex-20261105.json` is its output, with SHA-256 `8e8d1097499b3dc1912e6f5864f2e80376aeb78103f15867212229c262972957`. `paper2/release/reproduce_tables.py` rebuilds the analysis output and the timing output from the unzipped records, and with `--check` compares both with these copies.
