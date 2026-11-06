# Paper 2 Experiment 2

This folder supplies the study around Telohelm: episode streams, comparison conditions, private cases, reference labels, the planting benchmark and analysis. The reusable agent is installed from `../telohelm`.

Run commands from the repository root after following [the installation guide](../../README.md).

```sh
.venv/bin/python -m paper2.experiment2.run_stream --help
.venv/bin/python -m paper2.experiment2.run_stream \
  --episodes 1 --max-steps 30 --out .local-runs/setup-check
```

Use a fresh output folder. This small stand-in run checks setup and logging; it is not a learned result. Inspect its `stream-summary.json` and episode folders for goals, calls, costs, adopted sources and action records.

Real inference needs separately configured model servers. Development streams are distinct from reserved scored streams. The scored protocol and system must be frozen before scored evidence is collected; these commands do not bypass those requirements or submit a cluster job.

The prior source namespace was `study2`. Historical records retain their original source commits and paths. Current commands use `paper2.experiment2` and the installed `telohelm` package.
