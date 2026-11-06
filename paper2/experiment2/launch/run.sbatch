#!/bin/bash
# Unsubmitted study job. A reviewed plan, Codex's exact-source review, the stated gates and, for a reservation around
# 100 GPU-hours or more, Simon's yes precede any submission. The time limit is set from a reviewed projection and the
# verified queue rules; the value below is a placeholder that the plan's review replaces.
#SBATCH --job-name=telohelm-study
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --gpus=rtxp6000:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=20:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/%u/dcarr/logs/%x-%j.out
#SBATCH --error=/scratch/%u/dcarr/logs/%x-%j.err
set -euo pipefail
umask 077
: "${CODE:?a separately reviewed minimal payload is required}"
: "${STUDY:?a prepared study whose manifest the plan binds is required}"
: "${SLURM_JOB_ID:?run only inside the reviewed Slurm job}"
TASK_ROOT="/scratch/$USER/dcarr"
export HF_HOME="$TASK_ROOT/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1
unset PYTHONPATH PYTHONHOME TELOHELM_MODEL_SLOTS
rc=0
"$TASK_ROOT/envs/study2/bin/python" -I -B -c \
  'import sys,runpy;sys.path.insert(0,sys.argv.pop(1));runpy.run_path(sys.argv.pop(1),run_name="__main__")' \
  "$CODE" "$CODE/job.py" --payload "$CODE" --study "$STUDY" --out "$TASK_ROOT/runs/study-$SLURM_JOB_ID" \
  --hf "$HF_HOME" --vllm "$TASK_ROOT/envs/vllm/bin/vllm" || rc=$?
set +e
exit "$rc"
