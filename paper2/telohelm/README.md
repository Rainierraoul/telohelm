# Telohelm

Telohelm is a research prototype that turns experience into reusable Python procedures. It uses those procedures for routine decisions and calls a language model when a suitable procedure is unavailable or a configured check detects trouble.

The current task is a simulated aircraft joining a tanker and making a radio report. A task pack supplies the goal, observation fields, permitted actions and tools. The host connects the simulator to the runtime and applies the selected actions.

After an episode, the learner receives feedback and examples from its own records. It proposes a procedure, which runs in a separate process under execution limits. A protected monitor tests the candidate on completed episodes and checks whether earlier successes are retained before adoption. Later decisions record which adopted source they used.

The runtime also records model requests, replies, trial outcomes and costs. These records let a researcher follow a procedure from its proposal to later use.

## Install and check setup

From the research repository root with Python 3.12:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r paper2/telohelm/requirements-tested.txt -e './paper2/telohelm[test]'
.venv/bin/python -m paper2.experiment2.run_stream --episodes 1 --max-steps 30 --out .local-runs/setup-check
```

Use a fresh output folder. This short run checks installation, simulation and records with stand-ins. It ends at its step limit and provides no learning or efficacy evidence. Inspect `.local-runs/setup-check/stream-summary.json` and the episode folders. Real learning needs configured model servers; installation does not download weights or submit jobs.

The agent installs independently of the paper experiments. The command above uses the study's host to connect it to the tanker simulation. Both the saved-reply learning example and the distinct CartPole example below are packaged with the agent.

## Current limits

Development runs demonstrate procedure adoption and reuse in the tanker task. Whether the new handover mechanism improves recovery remains unresolved. Learned checks are implemented but have not been adopted in these runs.

The task interfaces also run the small CartPole demonstration below through the same core. Its replies are authored software controls. This prototype has been evaluated in simulation; physical control and broader task generality require separate studies.

## Where to inspect the implementation

| Location under `src/telohelm/` | Responsibility |
|---|---|
| `contracts/` | Task decisions, observation and output fields, goals and tools. |
| `runtime/` | Receive observations, choose a method, execute an action and record its source and cost. |
| `learning.py`, `learning_monitor.py` | Propose a candidate from experience, test it and decide adoption. |
| `qualification/` | Protected storage used by the monitor. |
| `taskpacks/tanker/` | Tanker-specific observation, goal, tool and simulator adapters. |
| `model_files/` | Exact tokenizer and template assets; model weights are separate. |

`tests/` holds component checks. Study schedules, reference labels, comparison conditions and analysis live in `../experiment2/`. Earlier workers are in `../../archive/telohelm_earlier/`, outside the installed package. The root [workspace guide](../../WORKSPACE.md) records compatibility paths and backup scope.

## Inspect a saved-reply learning cycle

From the research checkout after installation:

```sh
.venv/bin/python -m telohelm.demo --out .local-runs/learning-demo
```

The installed module also runs outside the checkout when called with the Python environment that contains Telohelm.

Use a new output folder. This is an offline software replay: it repeats a short authored tanker scenario and feeds the learner two unchanged model-written programs saved from development job 4063098. The initial malformed reply is deliberately retained. No model server, weights or cluster connection is used.

The ordinary gateway records the replay calls. The ordinary monitor runs each candidate in contained trial episodes before adoption. Later episodes record which exact adopted source acted. Inspect `summary.json`, `episodes.json`, `learning/` and `library/` in the output directory. Fixed public format guards and the scenario are designer-supplied; the programs are saved model replies. Model token demand is unknown in this replay.

This demonstrates the software path on one repeated case. It does not establish new model competence, cost savings, unseen-case performance or in-episode recovery.

## Try another simulator

Install the optional headless [Gymnasium CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) connection, then run:

```sh
python -m pip install -e './paper2/telohelm[cartpole]'
python -m telohelm.demo_cartpole --out .local-runs/cartpole-demo
```

The goal is to balance for 128 simulator steps. A constant-left baseline is followed by two **authored replies**: constant left and a simple angle/velocity controller. The unchanged core tests each candidate before adoption, then uses any adopted procedure on a second fixed starting seed. A final host-requested withdrawal is a separate negative control. The command returns failure if any expected check fails; it preserves all outcomes.

Read `summary.json`, the native step records in each `outcome.json`, `learning/trial-evidence/` and `withdrawal.json`. The four simulator readings are named without derived control features. No model is called. This demonstrates a task connection and the software learning cycle, not model discovery, a CartPole benchmark result or a cost benefit. One tick is one 0.02-second simulator step; real-time model control is untested.

## Use your own compatible server

The explicit server option sends one learning request, then tests any adopted procedure on two fixed later starts:

```sh
python -m telohelm.demo_cartpole --out .local-runs/cartpole-server \
  --server-url http://127.0.0.1:8001 --context-window 262144
```

Start your server separately. This client currently supports the bundled `Qwen/Qwen3.8-27B` tokenizer and template, served on a numeric loopback address with `/tokenize` and `/v1/chat/completions`. A chat-shaped endpoint alone is insufficient. Each exact request must have the same token count locally and on the server, and fit the context capacity you declare. Refused probes are saved and send no chat request. Model-name replies are checked too. Matching these checks does not verify the server's weights, hardware or capacity; a scientific run still needs its own serving evidence.

Defaults are one request, up to 131,072 output tokens, 16,384 input tokens and four hours, with `xhigh` effort. Set `--max-output-tokens`, `--input-cap`, `--timeout-s` and `--effort` explicitly to change them. There are no retries or authored-response fallback. The server receives the task rules, fixed format guard and baseline failure feedback; it receives neither the authored controller nor the later seeds.

Inspect `run-config.json`, `learning/token-count-gates.jsonl`, the gateway records and `learning_charges` in the summary. Delivery, qualification, adoption and later use are separate records. This is an unreportable development connection; it does not establish fresh model learning merely by returning HTTP success. The project's real-model test remains subject to independent review before execution.
