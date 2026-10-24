# Evaluation

This note says how a score is produced. The numbers live in `results/` and in the README block written by `python -m orbit_agents.eval.readme`. This page does not copy them.

## Protocol

Protocol id: `team-eval-v1`. The frozen file is `eval/test.jsonl`. Its digest is in `eval/manifest.json`, and the git tag of the same name points at the commit that introduced that file. `make score-dry-run` checks the file and the tag, reads no API key, and writes nothing.

`eval/dev.jsonl` is the dev split. Train and tune are slices of it, as `eval/PROTOCOL.md` describes. `python finetune/leakage.py` fails if a frozen task, a near-duplicate, or an oracle call appears in train or tune. Do not run `python -m orbit_agents.eval.build_tasks`. It rewrites the frozen split.

## What a row means

A model score has `kind: model_eval` and `status: measured`. The generator reads task success, schema-valid rate, and trap refusals from that file. A paired bootstrap appears only when `comparison.json` is measured. The v2 pair is adapter minus team base, in `results/v2_qwen1.5b/comparison.json`. That file is the ablation record. It is not retyped here.

A scripted run has `kind: system_check`. It proves the tools and the gate. It is not a language-model score. `results/harness_single.json` and `results/harness_team.json` are those checks. Files at the top of `results/` with status pending are not scores.

## Two uses of the name v3

Harness v3 is the tool and gate change: search lists tables and logs, python can read a bundled file by path, and the critic must return one JSON object. Scores for that harness would be `results/v3_qwen1.5b/`. They are not in yet, so no difference from the archived v2 numbers is claimed.

The round table is a separate team. It adds a coordinator, a researcher, a red team, and a verifier on top of the six-role crew. Its score would be `results/v3_api/<model>/team_v3.json`, written once by `make score-v3`. Until that file is measured, the README says the round-table team is not scored.

## How to score

These commands need the owner's key. They are not run in CI.

```bash
make score-dry-run MODEL=claude-haiku-5-5
export ANTHROPIC_API_KEY=your-key
make score-v1 MODEL=claude-haiku-5-5
make score-v2 MODEL=claude-haiku-5-5
make score-v3 MODEL=claude-haiku-5-5
```

`ORBIT_ANTHROPIC_API_KEY` is accepted as well. If both are unset, the make targets exit before any request and name `ORBIT_ANTHROPIC_API_KEY`. Each command refuses an output file that is already measured, then refreshes the README block.

The distill and training notebooks on a free Colab T4 are `notebooks/run_distill_and_score.ipynb` and `finetune/colab_train.ipynb`. They write under `results/v3_qwen1.5b/`. The T4 wall clock has not been measured.

## Orbit Code

Protocol id: `code-eval-v2`. The frozen files are under `data/scenarios/code_v2/`. The manifest stores the bundle hash and the split tag. Hidden tests are not in the prompt. `make score-code-dry-run` checks that hash, prints the request bound and the cost cap, and writes nothing. `code-eval-v1` under `data/scenarios/code/` is an earlier split. It was scored, and that score is superseded because the team harness stopped when the planner reply failed the schema and when a critic accept ended a red build. Those files stay in `results/code/`. They are not the current score. v2 is pending, not scored.

```bash
make score-code-dry-run MODEL=claude-haiku-5-5
export ORBIT_ANTHROPIC_API_KEY=your-key
make score-code MODEL=claude-haiku-5-5
```

`make score-code` writes `results/code_v2/<model>/single.json` and `team.json` once each. A file that is already measured is refused. The README block then reads pass@1, the hidden-test pass rate, reviewer rejections, tokens, and cost from those files. Until a file is measured, the row says pending, not scored. The superseded `code-eval-v1` record is not rewritten.

A local Qwen dry-run, which does not load weights:

```bash
python -m orbit_agents.eval.score_code --config single --provider transformers --model Qwen/Qwen2.5-1.5B-Instruct --dry-run
```

The boardroom under `data/scenarios/boardroom/` is a scripted demo on fictional files. It is not a scored split.
