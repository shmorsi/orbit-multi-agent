# Model card

Orbit is a research demo of a tool-using team that refuses claims its tools cannot prove. It is not a production assistant and it is not a general chatbot.

The measured models are Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct, fine-tuned with QLoRA on templated team dialogues. The archived scores live in `results/v1_qwen0.5b/` and `results/v2_qwen1.5b/`; the current harness-v3 scores live in `results/v3_qwen1.5b/`. Figures on the README are written by `python -m orbit_agents.eval.readme` from those files. This card does not restate them.

## What was trained

The student is the three-role team: planner, worker, and critic. Training dialogues come from the bundled fictional makerspace corpus and from the generators described in `finetune/PROTOCOL.md`. The corpus is in `data/corpus/`. Dialogue counts and the training config are in `results/v2_qwen1.5b/RUN.json` and `finetune/configs/`.

A distillation path asks a teacher (Claude by default, Gemini as an option) to produce trajectories on train and tune tasks only. That score is `results/v3_qwen1.5b/team_distill.json`. The distill config sets `max_seq_length` to 1536. The trainer refuses a shorter cap, because truncation drops the assistant turn. The notebook is `notebooks/run_distill_and_score.ipynb`.

## Orbit Code and the boardroom

Orbit Code is a planner, coder, test runner, and reviewer. The reviewer is code: it rejects a patch whose tests fail or whose diff touches the tests. The split is `data/scenarios/code_v2/`, protocol `code-eval-v2`. The measured files are `results/code_v2/claude-haiku-5-5/`. This card does not restate those numbers. The README does, and a pass@1 interval that includes zero is not a significant difference. The local Qwen row stays pending until that file is measured. `python -m orbit_agents.eval.score_code --dry-run` prints the request bound and the cost cap and writes nothing. `results/code/claude-haiku-5-5/` is the superseded `code-eval-v1` record. The team harness used for that run stopped when the planner reply failed the schema, and a critic accept stopped a red build before the coder could revise. Those files are not the current score.

The boardroom is a scripted demo on fictional company files. The auditor refuses a claim the cited file does not support. It is not a model score.

## API-policy teams

These teams were not trained. The three-role team and the six-role team have each been scored once as an API policy. The files are under `results/v3_api/`. The round table has not been scored. `make score-v3` is the command that would score it once, with `ANTHROPIC_API_KEY` or `ORBIT_ANTHROPIC_API_KEY`. Until `results/v3_api/<model>/team_v3.json` is a measured model eval, the README says the round-table team is not scored.

Harness v3 is a change to search, python, and the critic's JSON shape. It is not the round-table team. The Qwen rescore is in `results/v3_qwen1.5b/`. This card does not restate those numbers. The README block does, and a task-success interval that includes zero is not a significant difference.

## Evaluation

The frozen test is protocol `team-eval-v1`. See [EVAL.md](EVAL.md) and `eval/PROTOCOL.md`. Each config is scored once. A paired bootstrap is reported only from measured files. A pending file stays pending.

## Intended use and limits

Use it to inspect a grounding gate, a role split, and a frozen eval. Do not use the archived task-success numbers as evidence that fine-tuning improved answers. The README states that the task-success interval includes zero. The test set is small. Each measured config was trained once. The free-T4 wall clock has not been measured.

The makerspace corpus is fictional. The checkout outage and the Northline boardroom are fictional too. The tool path does not open a socket. The code test runner also blocks network connects and is killed on timeout. The teacher script is the piece that calls a model API, and only with a train or tune prompt. Scoring Orbit Code is a separate command and needs `ORBIT_ANTHROPIC_API_KEY`.

## Ethical notes

There is no personal data in the corpus. The red team plants an unsupported count so the gate can be seen refusing it. That planted count is not a measurement and is not an answer the auditor scores.
