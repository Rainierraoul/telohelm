# Workspace map

Some scripts and tests find the repository root by this file. It also shows where things are.

At the top level are `README.md`, the licence files (`LICENSE`, `LICENSE-data`, `LICENSE-APACHE-2.0` and `NOTICE`), `CITATION.cff` and two folders.

- `docs/` holds `DATA.md`, the guide to the study records.
- `paper2/` holds everything else, in the folders below.

Inside `paper2/`:

- `telohelm/` is the installable agent, the tanker task pack and simulator, and their tests.
- `experiment2/` is the study around the agent: scenario generation, the runner, the comparison conditions, the frozen analysis and the cluster launch scripts.
- `protocol/` holds the Experiment 2 protocol, its freeze records and the adviser amendment.
- `reproduction/` holds the frozen analysis output, the readers that build the paper's result tables and the audit readers with their reports.
- `manuscript/figures/` holds the scripts and sources for the paper's two figures.
- `release/` holds `reproduce_tables.py`.

The study records are not in this repository. They are in the data archive that `docs/DATA.md` describes.
