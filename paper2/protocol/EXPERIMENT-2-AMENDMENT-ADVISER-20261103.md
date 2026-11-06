# Experiment 2 amendment: the reasoning model alone (optional adviser), flown after the four scored streams

Status: proposed 3 November 2026 for Codex's review; nothing flies until Codex agrees in coordination/TURN.json.

## Decision and reason

Simon Vajda decided on 3 November 2026 to fly this condition, saying 'go on then, lets do the condition run'. Adversarial reviews of Paper 2 (Fable twice, Codex, Sol and Astra) all named the missing comparison. Earlier that day he had dropped the comparator ('this is making more work').

In the four scored runs the no-learning conditions never asked the reasoning model about the heading, because the small model's in-range replies pass the format checks. In the learning condition the heading came from procedures the reasoning model wrote. The comparison of learning with no learning therefore mixes learning with which model sets the heading. Stream 4301's records showed this on 31 October (24,557 heading requests, all to the small model). Paper 2 reports it as its main limit.

## What flies (condition, episodes, budget match)

The frozen protocol's optional condition: 'Optional: the reasoning model alone, as a budgeted adviser (decision D3), on 5 chosen episodes per phase per stream, 100 in all' (EXPERIMENT-2-PROTOCOL.md, protocol digest 02539acd…). The reasoning model alone flies the agent, with no library, no small model and no learning.

Each reply gives a heading and optionally a radio-report trigger and one help request, plus the time of its next call. An episode has at most 4 calls, the first at t = 0. Others come when a reply asks for one, on a goal change, on a refused action, on an unusable reply or when the goal tanker's newest radar report turns stale. Replies arrive 60 simulated seconds later, and the aircraft holds its last accepted heading between them. Settings are the frozen reasoning-model flight settings: 16,384 output tokens, 600 s timeout, medium effort, temperature 1.0, top-p 0.95 and top-k 20.

The episodes are 25 in each of streams 4301 to 4304 (5 per phase, phases 1 to 5, including the help cases), drawn on 26 October from seeds 6001 to 6004 before any condition flew. Each is flagged `adviser` in its stream's schedule.json of the prepared study scored-4301-4304-20261029, and a check on 3 November confirmed 25 per stream and 5 per phase. Twenty also appear among the 80 episodes that No learning and Unlimited calls flew (5, 5, 6 and 4 per stream). Learning flew all 100 of each stream, so every adviser episode has a Learning partner on the same prepared start.

Learning and every scored condition have four reasoning-model calls an episode: Learning at most one in flight and up to three for learning, the adviser four in flight. The budget per episode is matched; the use in flight is not, and the paper will say so.

## Code and settings fixed before the flight (with the digests)

The SHA-256 of adviser.py is 05cb81a128ab90c7b1008ec3de3b797f0849dc5b07e142df5cc9bd6574e3664a, byte-identical to the frozen source commit 4b607c89 (29 October). Its prompt is the text the file held at the freeze, whatever its docstring calls it, and stays so whatever the condition does. Also unchanged since the freeze are study_host.py (a6c893ba…), the runner, the frozen analysis.py (febebcad…) and the Telohelm package. No setting is tuned, before or after the flight.

The launcher job.py (now 44688c7b…) gains one function that passes the reasoning-model server's address to the host when the plan flies the adviser. No other plan changes.

The new module analysis_adviser.py (d9b937d8…) reads the frozen analysis. Per stream and pooled, by phase, it reports goals met without a help request, met with one and missed, with model calls and tokens. It does so for the adviser and for Learning on its episodes, with Learning's charged share of learning cost and its full learning cost beside them. Its digest is recorded here before the first scored adviser flight; a later change and its reason are recorded before it runs on scored output.

The branch stayed off main until all four streams had flown, so their payload stayed identical to the audited one. It merged on 3 November (merge ef87e7a9), when 112 tests passed in test_adviser, test_analysis_adviser, test_launch and test_study_host. One development episode (stream 4493) flew end to end on the Mac against a stand-in server that always answers heading 045, making 4 calls with 240 seconds held pending. This is a development check only and measures nothing.

## Smoke before the scored flight

Codex runs a native smoke through the established launcher on a development stream (not a reserved scored stream), adviser arm only, under 5 GPU-hours. It must show the job's own model servers on their own ports and adviser calls reaching the reasoning model with the 60-second hold. It must also show complete episode records and host receipts that analysis_adviser.py can read. Fixes may touch only the launcher path, never adviser.py or the prompt. Any change is recorded with its reason before the scored flight.

## The scored flight (job, budget, stopping, unflown episodes)

One job flies the adviser arm only, on streams 4301 to 4304 of the same prepared study as the scored runs (study manifest c241bae2c9fc93d63e75e408f7c25ce149d04e86051b219ac06754f0fbde4c38 in the scored plans, which Codex checks against the remote study before launch). It uses two RTX PRO 6000 GPUs (the launcher requires two; the small-model server is idle), with a host limit of 18 hours, Slurm time of 20 hours and a reservation of at most 40 GPU-hours.

Spent so far is 186.0 of the 280 GPU-hour freeze threshold (300 ceiling, 20 reserve), so 226 at most after this job. The estimate is 8 to 15 host hours and 16 to 30 GPU-hours, from 4301's flight reasoning calls (median 37 s, mean 135 s); it rests on a different prompt.

Episodes the host limit leaves unflown are reported as unflown, never replaced. A rerun after a software fault is allowed only before any adviser outcome is read. Codex's agreement in TURN.json is required, because the job is over 20 GPU-hours.

## Reporting

Every adviser outcome is reported, whatever it shows, beside Learning on the same episodes and beside No learning and Unlimited calls on the 20 shared episodes. Paper 2 states that the condition was flown after the four streams' results had been read, at the author's decision, because reviewers named the missing comparison. It also states that its design, code and episodes were fixed before any scored flight and that its per-episode budget matches Learning's while its use in flight differs. If the analysis runs on a local merged copy (the Mac's originals plus the adviser's returned records) because the remote study is gone, that is recorded.

## Not changed by this amendment

Paper 1's calculator arm exists in the host but has an unresolved qualification note (amendment 7). It does not fly. No other condition, setting or episode changes.
