# Study launcher

One Slurm job runs a prepared study's conditions side by side on one pair of owned model servers. Nothing here has been submitted. A reviewed plan, Codex's exact-source review and the stated gates come first, and a reservation of around 100 GPU-hours or more also needs Simon's yes.

## Files

- [job.py](job.py) is the job. It reuses the native stages of the M1 rehearsal's launcher, accepted at df0e7ab, and the native primitives reviewed at 81a5af3, around the study host.
  - It checks the payload and the prepared study's manifest.
  - It installs the product into a fresh environment, isolated, and checks the plan's load limits against the allocation and the installed sandbox constants.
  - It runs the native containment controls, then the study host's startup check isolated in the installed environment, before any server starts.
  - It starts and admits two owned servers, checks token parity, sets the shared slots, samples its own GPUs and runs the host.
  - The job sets a fresh random marker, `TELOHELM_JOB_TOKEN`, in its own environment before it starts anything, so every process it starts inherits it: the host and its dispatchers and workers, both servers and their engine processes, the sampler, and every short command.
  - Close-out runs each stage on its own, so a failure in one never skips the next, and every wait is bounded. It stops the host, each server and the sampler. It then ends everything still carrying the marker, renewing that evidence before every signal and signalling marked processes in its own group one by one, and confirms that nothing is left. Then it closes each log handle and writes checksums. A stage that failed, a process whose environment could not be read, or a sweep that could not see makes the checksum record incomplete.
  - Before anything starts, the plan's checks run. The host duration must be finite, positive and at most 240 hours. There must be exactly two GPUs, with whole-number tensor parallelism of 1. The serving settings, slots and profile models are also checked. The startup check's dispatchers must match the plan's count and conditions.
  - `--cpu-rehearsal` runs every stage that needs no GPU or model server, with stand-in models, and lists each skipped stage by name.
- [build_payload.py](build_payload.py) builds the payload from one commit. The product wheel must match that commit's product source file for file. The payload holds the experiment modules the host and runner import, the native tests and a committed plan.
- [run.sbatch](run.sbatch) is the unsubmitted Slurm script. Its time limit is a placeholder until a reviewed projection and the verified queue rules set it.
- [plan-template.json](plan-template.json) shows the fields a plan needs. Its profile names the servers as `{small_model}` and `{reasoning_model}`, which the job fills in with its own servers' addresses. The watch and report-control settings stay unset until the tanker diagnostic fixes them, and the startup check refuses them while they are unset.
- [plan-cpu-rehearsal.json](plan-cpu-rehearsal.json), [rehearse_cpu.py](rehearse_cpu.py) and [rehearsal-cpu-20261026.json](rehearsal-cpu-20261026.json) are the CPU rehearsal: its plan, its helper and its record.

## The CPU rehearsal

It used development streams 4493 and 4495, phases 1 to 3, all five main conditions, stand-in models and a 10-step limit. The payload was built from commit 781e12e, and the launcher was run through the isolated `-I` command the Slurm script uses. Every stage that needs no GPU or model server passed. All ten dispatchers returned with exit 0, and all 410 positions stopped at the step limit. Every cleanup stage completed, the sweep closed with nothing left running, and the checksum record is complete. The product the dispatchers ran was the installed one, and every dispatcher command was isolated. It took about 80 seconds; the record has the exact times.

It is a software rehearsal. It runs no native containment, model server or real model, and says nothing about throughput or cost.

## Not settled

- A scored payload carries the frozen protocol at the path its freeze record binds, and the build refuses if the digest differs. The freeze record must still bind a profile digest that leaves out the servers' addresses, since those change with each job's ports. That needs a small change to the runner's admission check, which has not been made.
- The M1 rehearsal's migration onto this launcher needs its own reviewed design. Its C seeds are outside the runner's development range, its pause is off, and the host's default conditions differ from its own.
- The wall time, the queue and the full five-condition load on four streams are decided from measurements.
