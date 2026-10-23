# Orbit

A local language-agent team that calls tools under a schema and refuses any claim those tools cannot prove.

[![check](https://github.com/shmorsi/orbit-multi-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/shmorsi/orbit-multi-agent/actions/workflows/ci.yml)
[![MIT license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue)](https://www.python.org/)
[![Live replay](https://img.shields.io/badge/replay-live-blue)](https://shmorsi.github.io/orbit/)

**Live replay:** [shmorsi.github.io/orbit](https://shmorsi.github.io/orbit/) · [Why this matters](#why-this-matters) · [Key results](#key-results) · [Six roles](#v2-six-roles) · [Quickstart](#quickstart) · [Integrity](#reproducibility-and-integrity)

```mermaid
flowchart LR
  task[Task] --> commander[Commander]
  commander -->|assign| scout[Scout]
  commander -->|assign| engineer[Engineer]
  commander -->|assign| analyst[Analyst]
  scout --> tools[Tools]
  engineer --> tools
  analyst --> tools
  tools --> analyst
  analyst -->|draft| critic[Critic]
  critic -->|revise| analyst
  critic -->|accept| auditor[Auditor]
  auditor --> trace[JSONL trace]
```

Six seats share one gate. The commander assigns a step, specialists call tools, the critic can send a draft back, and the auditor refuses any claim the tool results do not support. The measured runs used the three-role team in [Architecture](#architecture). This six-role team has not been scored.

## Why this matters

Hiring teams want an agent that uses tools and stops when the evidence runs out. Orbit is a small, fully inspectable version of that problem. Every seat returns one JSON object. The tools are sandboxed and nothing in that path opens a socket. A critic can send a claim back. A gate keeps a claim only when the cited tool result contains it, a python call printed it, or it equals a count of matching lines in the cited text.

The frozen test is hashed, tagged, and kept out of training. Each model is scored once, with a paired bootstrap, and a negative result stays on the page. The measured 1.5B QLoRA run learned the answer format. On the frozen 12-task test it did not produce a task-success gain this sample can support. The paired interval for task success includes zero. That is the result.

## Key results

This block is written by `python -m orbit_agents.eval.readme` from the files in `results/`. API scores and team distill are pending. A pending run is a row that says not scored, with no number filled in.

<!-- RESULTS_START -->
Measured once on the frozen test `team-eval-v1`, Qwen2.5-1.5B-Instruct, QLoRA, 307 train dialogues and 71 tune dialogues. Files: `results/v2_qwen1.5b/`. Logged loss went from 1.152004337310791 at step 10 to 0.05106291174888611 at step 150. Task success did not clear zero.

| Run | Successes | Task-success rate | Schema-valid rate | Trap refusals |
| --- | --- | --- | --- | --- |
| Single base | 2 of 12 | 0.16666666666666666 | 0.75 | 2 of 2 |
| Team base | 2 of 12 | 0.16666666666666666 | 0.6071428571428571 | 2 of 2 |
| Team plus adapter | 3 of 12 | 0.25 | 0.7797619047619048 | 1 of 2 |
| Team distill | not scored | | | |

The adapter's two answerable successes sit next to a missed trap: correct refusals went from 2 to 1. The paired bootstrap for task success, adapter minus team base, is -0.16666666666666666 to 0.3333333333333333 (point 0.08333333333333333). That interval includes zero, so the task-success gain is not statistically significant. The same pair's schema-valid interval is 0.0013888888888889117 to 0.33888888888888885 (point 0.17261904761904764). Schema-valid rate moved. Task success did not, on this 12-task set.

The earlier 0.5B run is in `results/v1_qwen0.5b/`. Successes were 2 of 12, 2 of 12, then 1 of 12, with schema-valid rates 0.0, 0.07083333333333335, and 0.6131944444444445. Logged loss went from 1.3191939353942872 to 0.20024461746215821. Its task-success interval also failed to clear zero. Fine-tuning taught the format.

The placeholder files at the top of `results/` stay pending. A harness file in `results/` is a system check of the scripted tools, not a model score. Team distill is a fourth config. It has not been scored.

The harness changed in v3. Search now lists tables and logs, python can read a bundled file by path, a count printed by python or equal to the matching lines in the cited text stays grounded, and the critic must return one JSON object. The v2 numbers above stay archived in `results/v2_qwen1.5b/`. single_base, team_base, team_lora, and team_distill must be scored again on the frozen test under v3 before any comparison. Those v3 scores are not in yet.
<!-- RESULTS_END -->

![Mission Control replaying a saved trace, with the frozen-test flight data underneath](docs/mission-board.png)

![Flight-data board for the frozen 1.5B runs: task success, schema-valid rate, the paired bootstrap, training loss, and trap outcomes, with team distill empty](docs/mission-flight-data.png)

<video src="docs/mission-control.mp4" controls muted playsinline aria-label="Mission Control replaying a saved trace, stepping through planner, worker, tool, and critic"></video>

![Mission Control replaying a saved engine trace](docs/mission-control.gif)

![Mission Control at the moment the critic accepts a grounded claim about printer jobs](docs/mission-replay.png)

## Architecture

Orbit runs over a fictional makerspace corpus. The worker calls tools. The critic sends a bad claim back. A gate keeps a claim when the cited tool result contains it, when a successful python call printed the count, or when the count equals the number of lines in the cited text that contain a token from the question. Otherwise the gate refuses the claim. Mission Control replays the JSONL trace, including a task you start on your own machine.

```mermaid
flowchart LR
  task[Task] --> planner[Planner]
  planner --> bus[Message bus]
  bus --> worker[Worker]
  worker --> tools[Tools]
  tools --> worker
  worker --> critic[Critic]
  critic -->|revise| worker
  critic -->|accept| gate[Groundedness gate]
  gate --> trace[JSONL trace]
  trace --> viewer[Mission Control]
```

The teacher pass sits beside that loop. It never sees the frozen test.

```mermaid
flowchart LR
  pool[Train and tune tasks] --> teacher[Claude or Gemini teacher]
  teacher --> sandbox[Local tool sandbox]
  sandbox --> teacher
  teacher --> keep[Checker and schema filter]
  keep --> dialogs[Distilled dialogs]
  dialogs --> student[QLoRA student]
  block[Frozen test blocklist] -.-> teacher
  block -.-> keep
  student --> score[One frozen score]
```

The tools are a sandboxed Python cell that reads a bundled table or file by path, a calculator, corpus file read, a per-run file write, keyword retrieval over the bundled notes, tables, and logs, and an offline web cache. Nothing in the tool path opens a socket. The teacher script is the only piece that calls a model API (Claude by default, or Gemini), and only with a train or tune prompt.

Flight rules, from the code the replay reads:

- The planner returns at most four steps, or an empty plan when the corpus cannot answer.
- The worker emits one tool call or a final answer. A claim cites a tool result that contains it, a python call that printed the count, or text with that many matching lines.
- The critic replies with one JSON object. It accepts a grounded claim, including a count printed by python and a count of matching lines in the cited text. Otherwise it revises, at most 2 times, and then the gate refuses unsupported claims.
- The team stops after 8 turns. A broken schema gets 2 tries on that turn.

## v2: six roles

`--team v2` runs the same tasks, tools, and grounding gate with six seats instead of three. `--team v1` stays the default, so every number on this page is a v1 team result.

```mermaid
flowchart LR
  task[Task] --> commander[Commander]
  commander -->|assign| scout[Scout]
  commander -->|assign| engineer[Engineer]
  commander -->|assign| analyst[Analyst]
  scout --> tools[Tools]
  engineer --> tools
  analyst --> tools
  tools --> analyst
  analyst -->|draft| critic[Critic]
  critic -->|revise| analyst
  critic -->|accept| auditor[Auditor]
  auditor --> trace[JSONL trace]
```

| Role | Tools | Can answer |
| --- | --- | --- |
| Commander | none; assigns each step to one specialist | no |
| Scout | retrieve, read_file, web_search | no |
| Engineer | python, write_file | no |
| Analyst | calculator | yes, the only seat that drafts the final |
| Critic | none | no; accepts or sends the draft back, at most 2 times |
| Auditor | none; read-only copy of the trace | no; refuses any claim or number the tool results do not support |

- Every turn is one JSON object checked against that role's schema. A tool outside the role's list fails validation and never runs.
- The auditor is code, not a model turn. Its verdict is the v1 grounding gate, and it writes one finding per claim and per number into the trace.
- The limits are the v1 limits: 8 turns, 2 revise passes, and 2 schema tries on a turn.
- Every seat calls one interface, `LLMClient.complete(messages, schema)`, in `src/orbit_agents/policy.py`. Backends: scripted, transformers, ollama, llama.cpp, and anthropic. Adding a provider is one class; `OpenAICompatiblePolicy` is the documented stub.
- The policy sees only the role prompts and the task. It never sees the checker contract the distillation teacher gets, or any frozen-test content.

Offline, with the scripted harness (a harness check, not a model score):

```bash
python -m orbit_agents.eval.run --team v2 --split test --out traces/v2_check.json
make traces-v2   # rewrites mission/traces/v2: accept, revise, and an auditor refusal
```

With Claude as the policy for both teams, so v1 and v2 are compared on the same model:

```bash
make score-dry-run MODEL=claude-haiku-5-5          # checks the frozen test hash and tag, prints the request bound
export ORBIT_ANTHROPIC_API_KEY=your-key
make score-v1 MODEL=claude-haiku-5-5
make score-v2 MODEL=claude-haiku-5-5
```

Each target scores the frozen test once and writes `results/v3_api/<model>/team_v1.json` or `team_v2.json`. It refuses a config that is already measured. The request cap and the 5 USD cost stop are the distillation script's. When both files are measured, the paired bootstrap of v2 minus v1 is written next to them, and the README generator adds the rows below. The v2 team has not been scored yet.

If `ORBIT_ANTHROPIC_API_KEY` is unset, `make score-v1` and `make score-v2` exit before any request and name that variable. Nothing is read from a file. `make score-dry-run` does not read a key.

## Quickstart

Three commands from a fresh clone. The first creates the environment and installs the package. The second runs the suite. The third opens Mission Control.

```bash
python3 -m venv .venv && source .venv/bin/activate && make setup
make test
make demo
```

`make doctor` times the scripted harness on your machine. Platform notes are in [Run it locally](#run-it-locally). The development notes, including what not to edit, are in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

## Reproducibility and integrity

- The frozen test is `eval/test.jsonl`, tagged `team-eval-v1`. Its digest is stored in `eval/manifest.json`. `make score-dry-run` checks the file and the tag, reads no API key, and writes nothing.
- `python finetune/leakage.py` fails if a frozen task, a near-duplicate, or an oracle call appears in train or tune.
- `python -m orbit_agents.eval.readme --check` fails if the Key results block disagrees with `results/`. The test suite runs that check.
- A config is scored once. `eval.run` and `eval.compare` refuse archived files, harness checks, the pending placeholders, and a second score of a measured frozen-test file.
- Files at the top of `results/` with status pending are not model scores. `results/harness_single.json` and `results/harness_team.json` replay the scripted harness. They are system checks.
- The manifest also stores a prompt digest from the protocol freeze. Later harness work changed the role prompts. The scorer checks the frozen test and the tag. This page does not treat that prompt digest as a current hash of the prompts.
- Cite the project with [CITATION.cff](CITATION.cff).

## What I'd do next

Future work, not a claim about a result:

- Score the three-role team and the six-role team once on the same API model, with `make score-v1` and `make score-v2`, and let the generator add the paired interval.
- Run the distill notebook once on a Colab T4, score team distill under harness v3, and commit only `results/v3_qwen1.5b/*.json` and `README.md`.
- Keep the archived v2 numbers in `results/v2_qwen1.5b/` and do not compare them with a v3 score.
- Train more than one seed before talking about a distribution. This page has one training run per config.

## Limits

- The frozen test is 12 tasks, 2 of them traps. One task moves a rate by a twelfth, so only large effects can clear zero.
- Each model was trained once and scored once. There is no seed sweep.
- The adapter that gained a success also missed a trap: correct refusals went from 2 to 1.
- The harness changed in v3. The v2 numbers are archived and are not compared with v3 runs. No v3 model score is in yet.
- Team distill has not been scored. The notebook below is the path to that one score.
- The v2 team (6 roles) has not been scored. Its traces in `mission/traces/v2/` come from the scripted harness, which follows each task's oracle; they show the protocol, not model skill.
- The free-T4 wall clock for training plus scoring has not been measured.

## Distillation

The templated 307 dialogs taught the JSON shape and not the counting. `finetune/distill_teacher.py` asks a teacher model to act as planner, worker, and critic on train and tune tasks only. Each tool call runs in Orbit's sandbox. A trajectory is kept only when the task checker passes and every turn matches the role schema. Trap tasks stay in the pool so a correct refusal can be kept. The generator adds count, sum, threshold, log, quotient, write, and refusal variants on top of the 69 templated train tasks. Gemini and Claude write the same dialog files. Each kept dialog records the teacher model in its metadata, and `stats.json` records it too.

The frozen test is a local blocklist. Its ids, questions, near-duplicates, distinctive fragments, and oracle calls are not placed in a request and are not written as rows. `--dry-run` prints the pool and does not read an API key.

Cost controls:

- The hard cap defaults to 800 requests for this run. The checkpoint still records the running total. A restart is not blocked by that total unless you also set `--max-total-requests`.
- With `--provider gemini`, the client waits so the pace stays near 4 requests per minute. Each request log prints that configured rate, the effective rate since the run started, and the count per status code.
- The first request is a preflight with no task text. A client error there exits before any task is sent.
- HTTP 503 waits 20 seconds, then 40, then 80. Each of those retries counts toward the cap. After that the process stops.
- A per-minute HTTP 429 whose stated retry delay is 120 seconds or less waits that long once and continues. A per-day quota, a longer delay, or a second consecutive 429 stops the process and writes the checkpoint.
- Any other client error stops the process on that response. The run does not send another request.
- Another server error is tried up to 3 times, then the process stops.
- A model whose name starts with gemma does not receive a system instruction field. That text is placed in the first user turn.
- Token counts and the request total are printed as JSON lines.
- If the Google project has billing turned on, successful calls can still incur charges up to the cap. Confirm billing is off before you export the key.

```bash
python finetune/distill_teacher.py --dry-run
export ORBIT_ANTHROPIC_API_KEY=your-key
python finetune/distill_teacher.py \
  --provider anthropic \
  --model claude-haiku-5-5 \
  --max-requests 800 \
  --requests-per-minute 20 \
  --max-cost-usd 5 \
  --attempts 3 \
  --output finetune/data/distill
```

The default provider is Anthropic and the default model is `claude-haiku-5-5`. A name that starts with `claude` selects Anthropic even without `--provider`. `--model` also accepts the Sonnet and Opus ids in the price table. The Anthropic key is read from `ORBIT_ANTHROPIC_API_KEY`. Requests omit temperature unless you pass `--temperature`. The reply is taken from text blocks only. The teacher prompt includes the final-answer JSON and the tool-call JSON the checker scores, including the claim names for that task. Pass `--families` and `--limit-tasks` to try a few tasks, and `--debug-dir` to keep every attempt. The pace defaults to 20 requests per minute. Request logs include input tokens, output tokens, and a running cost estimate in USD. The estimate uses the price table in `finetune/distill_teacher.py`. The same table is recorded in `finetune/configs/qlora_distill.json`. An unknown model uses the fallback price, 15 dollars per million input tokens and 75 dollars per million output tokens, so the cost limit cannot be under-counted. At 5 USD the run stops and writes the checkpoint. A response that says the credit balance is too low, or any other billing error, stops the same way. HTTP 429 and overloaded or other server errors wait 20 seconds, then 40, then 80, and each retry counts toward the cap. Any other client error stops on that response.

Gemini is still available. Pass `--provider gemini` and set `ORBIT_TEACHER_API_KEY`. That path keeps the free-tier pace of 4 requests per minute and does not apply the dollar limit. A stopped run resumes from `finetune/data/distill/checkpoint.json`. The kept dialogs are `train.jsonl` and `tune.jsonl` in that directory, in the same message format as the templated set, plus `stats.json` with attempts and pass rate per task family.

Train the student with `finetune/configs/qlora_distill.json`. In Colab, set `ORBIT_DISTILL=1` before the distillation cell in `finetune/colab_train.ipynb`. That cell trains the distill config and scores `team_distill` once. The harness is now v3, so a fair comparison needs a fresh score of single_base, team_base, team_lora, and team_distill on the frozen test. Set `ORBIT_RESCORE=1` before the rescore cell. That cell does not train. It writes the four scores once under `results/v3_qwen1.5b/`. The v2 numbers stay archived in `results/v2_qwen1.5b/`. The comparison adds the distill pairs only when that file is present. Do not score the frozen test in a loop.

## Run it locally

Python 3.11 or newer. The demo target is an 8 GB machine with the 0.5B model. The 1.5B model is slow on that machine but can run in fp16. QLoRA and bitsandbytes are CUDA-only. `make setup` does not install them, and the demo never imports them.

`make doctor` prints seconds for the scripted harness on the machine you run it on. `make doctor-live` downloads Qwen2.5-0.5B-Instruct and prints seconds for one real task. This page does not publish those readings.

### M1 Mac

```bash
python3 -m venv .venv
source .venv/bin/activate
make setup && make doctor
make doctor-live
make demo
```

`make doctor` should print `mps` when PyTorch can see the GPU, and `cpu` when it cannot. Open the localhost URL the server prints. Pick a bundled task id with the scripted harness, or type a question and choose Transformers.

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
make setup && make doctor
make doctor-live
make demo
```

Without CUDA or MPS the device line is `cpu`. The same Mission Control page streams the run.

### Windows

`make` may be missing. From a Python 3.11 venv:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
py -3.11 -m pip install -e ".[dev]" -r requirements-infer.txt
py -3.11 -m orbit_agents.doctor
py -3.11 -m orbit_agents.doctor --live
py -3.11 -m orbit_agents.serve
```

The Windows device is `cpu`. The Python tool still dies after one second; the limit is a child process.

### What you should see

Mission Control shows the planner, the worker, a tool call in the telemetry strip, and the critic. The phase rail steps Planner, Worker, Tool, Critic. The Trace menu also lists the six-role traces under Team v2; choosing one switches the ring to those six seats and the rail to Commander, Specialist, Tool, Critic, and Auditor. Under the replay, the flight-data wall charts the frozen 1.5B runs with bootstrap intervals, the loss curve, the trap outcomes, and an empty slot for team distill. The T+ column is a replay clock, not measured latency.

`mission/index.html` is static. GitHub Pages can host the `mission` directory, and the same files open under `make demo`. The run form needs the local server. `prefers-reduced-motion` turns the pulses off.

Record the clip again with `python demo/record_replay.py`. It needs Chrome and ffmpeg.

Troubleshooting:

- `torch is not installed` means `make setup` has not been run. QLoRA is still not required.
- `mps` missing on an M1 means the demo is on CPU. The 1.5B model is slow there; use the 0.5B model or Ollama if you want a shorter wait.
- A free-form question with the scripted harness returns an error. That backend only runs a bundled task id.
- Out of memory: stay on the 0.5B default. Do not install bitsandbytes on a Mac to make QLoRA work. That stack is the Colab path.

Pins for the demo live in `requirements-infer.txt`. The Colab training pins live in `finetune/requirements-colab.txt` and do not reinstall torch.

## Fine-tune

On a free T4, open `finetune/colab_train.ipynb` and run it from the top.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shmorsi/orbit-multi-agent/blob/main/finetune/colab_train.ipynb)

The default cell trains `finetune/configs/qlora_medium.json` (Qwen2.5-1.5B-Instruct, QLoRA). It targets a free T4 with 16 GB and about 60 minutes for training plus the three evals. That wall clock has not been measured here.

- Optional: set `ORBIT_MOUNT_DRIVE=1` before the first code cell to mount Drive. Otherwise checkpoints stay on the runtime disk under `/content/orbit-checkpoints`.
- Upload the project zip to `/content/orbit.zip` to train that copy. With no zip, the notebook clones https://github.com/shmorsi/orbit-multi-agent.
- The install cell uninstalls torchao, installs `finetune/requirements-colab.txt` with `--upgrade-strategy only-if-needed`, then `pip install -e . --no-deps`, and runs the leakage check.
- Score the frozen test once into `results/v3_qwen1.5b/`: single base, team base, team plus the adapter, then the comparison. A file that is already measured is skipped. These cells do not replace the pending files at the top of `results/`. Set `ORBIT_DISTILL=1` only when the distilled JSONL is in the zip. That adds `team_distill` under the same directory and does not repeat the three scores.
- The notebook then runs `python -m orbit_agents.eval.readme`, which rewrites the results block from those files. Zip the run to `/content/outputs.zip` and download it. Commit `results/v3_qwen1.5b/*.json` and `README.md` only.

`finetune/configs/qlora_small.json` is the 0.5B config from the measured v1 run. `--method lora` is the MPS or CPU training path. On an M1, prefer that 0.5B config or Ollama.

Do not score the frozen test in a loop. A new model gets one frozen score. Iterate on the tune split with `python -m orbit_agents.eval.run --split tune`.

### Distill and score in one notebook

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shmorsi/orbit-multi-agent/blob/main/notebooks/run_distill_and_score.ipynb)

[`notebooks/run_distill_and_score.ipynb`](notebooks/run_distill_and_score.ipynb) runs the team-distill path end to end on a free T4: the Claude teacher (`claude-haiku-5-5` by default, `claude-sonnet-5-5` as an option) under the same request cap and cost stop, QLoRA with `finetune/configs/qlora_distill.json`, one frozen score into `results/v3_qwen1.5b/team_distill.json`, the v3 baselines it needs (each once, skipped if present), the paired comparison, and a regenerated README results block. Add the key as a Colab secret named `ORBIT_ANTHROPIC_API_KEY` before running. It is read from secrets or the environment and never printed. If the secret and the environment variable are both missing, the key cell exits and names `ORBIT_ANTHROPIC_API_KEY`. The download keeps `results/v3_qwen1.5b/*.json` and `README.md` at those paths. Commit only those files.

The results block above is written by `python -m orbit_agents.eval.readme` from the files in `results/`. `--check` fails when the README and the files disagree, and the test suite runs that check.
