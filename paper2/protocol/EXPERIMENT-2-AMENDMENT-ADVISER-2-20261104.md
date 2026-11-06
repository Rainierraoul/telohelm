# Experiment 2 amendment 2: the adviser outside the condition-wide pause, with a reservation for all 100 episodes

Status: approved 4 November 2026. Simon approved 80 GPU-hours for the adviser job within 280 total, and Codex's agreement is recorded in coordination/TURN.json. Times are Australian Eastern Daylight Time (AEDT). The second native smoke and exact admission must pass before the scored flight.

Codex started the scored adviser job 4546270 under the first amendment at 06:54:41, before it had seen this proposal. The job was cancelled at 06:56:02, after 81 seconds, while its model servers were still starting. No adviser episode started: the prepared study scored-4301-4304-20261029 has no adviser folder in any stream. The job's launch records are kept. No scored adviser outcome exists.

## What happened

The first adviser amendment (commit 03099734) asked for a development smoke before the scored flight. In job 4522838, on stream 4493, one of the four reasoning calls in the first episode timed out, the protocol's pause rule fired, and the automatic handoff stopped without starting the scored job. Codex then corrected the handoff check to accept a verified protocol pause (commit afd3f09d) and started the scored job, which was cancelled before any episode, as the status above records. The details are in paper2/notes/ADVISER-SMOKE-4522838-CLAUDE-20261104.md.

## The change and its reason

The adviser is exempt from the condition-wide pause. The pause is the rule at line 240 of EXPERIMENT-2-PROTOCOL.md. It looks at the first 20 scheduled positions of a condition and stops the condition when 10 times the timeout-or-truncation calls is greater than the started calls. The runner checks it after every episode.

The pause exists to stop a condition when the model service is failing, to protect the budget and the validity of the results. The adviser's frozen design already says what a timeout does in flight: the aircraft holds its last accepted heading and the adviser is called again. Every timeout is an outcome under the same frozen 600 s setting, so it is recorded and reported and never retried or replaced. With four calls an episode, one timeout in either of the first two episodes would stop the comparison before it had a result (4 or 8 calls started, and 10 is more than 8).

The runner still works out the stop the pause would have made. It records that as a counterfactual stop, meaning a stop that the runner computed and wrote down and then left alone, as the protocol's development --no-pause option already does. Every timeout is also reported per stream and phase, with its call position, beside the answered calls.

## Budget and host limit

Compute is counted in GPU-hours, where one graphics processing unit (GPU) used for one hour is one GPU-hour. The first amendment set a host limit of 18 hours, Slurm time of 20 hours and a reservation of at most 40 GPU-hours. Its estimate of 8 to 15 host hours was too low, for the reason given in the smoke note.

A dispatcher is the worker that flies one stream's episodes, and the plan has four. A slot is the share of a model server that the launcher gives the host's workers. The plan sets one reasoning-model slot, so the dispatchers take turns and reasoning calls run one at a time. At the smoke's 1,155 s an episode, 100 episodes need about 32 host hours, and the old 18-hour limit would fly roughly 56. That is one episode, a weak sample, and the second smoke below gives a better one.

To fly all 100 pre-drawn positions, the host limit becomes 38 hours, the Slurm time 40 hours and the reservation at most 80 GPU-hours (2 GPUs for 40 hours). Spent so far is 186.0 GPU-hours plus 0.77 for the smoke, which is 186.8, plus whatever job 4546270 uses before it is cancelled. Adding 80 gives about 267, inside the 280 freeze threshold and inside the 80 GPU-hours per reservation that Simon approved on 28 October. The job is over 20 GPU-hours, so Codex's agreement in TURN.json is still required.

## What does not change

Nothing else changes: adviser.py and its prompt, the 600 s timeout, 16,384 output tokens, medium effort, the sampling settings, one reasoning slot, the 100 pre-drawn episodes and the reporting rules in analysis_adviser.py. Raising the timeout or the slots is rejected because either would change call times, and so which calls time out.

## Implementation and the second smoke

The adviser loop in the runner calls pause_authority after every episode and has no switch to turn that off (study_runner.py, lines 794 to 799). The change is therefore in the runner. The launcher code is untouched, and the scored plan changes only its host hours and Slurm time. Codex implements it with tests and records the new runner digest here before the scored job.

Codex then runs one more development smoke on a development stream, in a freshly prepared study because the first smoke's study holds a stop record: five episodes, under 5 GPU-hours, with the pause exempt and the counterfactual stop recorded. It measures pace and timeouts over more than one episode. If the pace implies more than 38 host hours for 100 episodes, Codex adjusts the reservation inside 280 or records the expected share of unflown episodes before launch.

## Reporting

Paper 2 says that the adviser flew under this second amendment, why, and when, alongside the first amendment's statements. It adds that the change was decided after a development smoke and before any scored adviser outcome existed. It reports the counterfactual stop and the timeouts as set out above. Episodes the host limit leaves unflown are reported as unflown and never replaced.

## Reviewed implementation before the second native flight

Codex implemented the adviser-only exception without changing job.py or adviser.py. The first due decision is published atomically at counterfactual-stops/adviser.json and recorded in adviser state; the other conditions retain their enforced stop. Runner SHA-256: afb86defbf3af77e9ce0ac8fcc42a1be31508395024ee87d3b6890100338c7b1. Adviser SHA-256 remains 05cb81a128ab90c7b1008ec3de3b797f0849dc5b07e142df5cc9bd6574e3664a. The updated reader exposes the counterfactual event and episode/phase/call-position timeout details, labelling incomplete call records. Its SHA-256 is b86895252dd9f5f776f988589d7dad74af7d56d5cc8e11b3a30dff3a57afe48b.

The reviewed smoke has a 1.9-hour host limit and two-hour scheduler limit, at most four GPU-hours. Including the first smoke's actual 0.770555556, both smokes remain below five. The complete conservative budget, including the 0.045-hour cancelled setup, is at most 270.812777778. The scored cap remains 80 even if the smoke suggests more than 38 host hours; report expected unflown positions instead of increasing the approved cap. The source/integration checks passed 197 tests and the latest reader/gate checks passed 23. Review details are in paper2/notes/ADVISER-AMENDMENT-2-REVIEW-CODEX-20261104.md.
