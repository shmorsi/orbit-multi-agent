# Development

The test suite is the scripted harness. It does not need an API key, a GPU, or torch.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
ruff check src tests mission/build_board.py
python3 -m pytest
python3 -m orbit_agents.doctor
```

`make setup` also installs the inference pins in `requirements-infer.txt`. Mission Control needs those. The tests do not.

`make demo` serves Mission Control. `make doctor` times the scripted harness on the machine you run it on. `make doctor-live` downloads a small model; this repository does not publish that timing.

## Do not edit

These paths are the frozen protocol, the hashed corpus, or a committed measurement. Leave them alone.

- `eval/test.jsonl`, `eval/dev.jsonl`, `eval/manifest.json`, `eval/PROTOCOL.md`
- `finetune/PROTOCOL.md`, `finetune/data/`, `finetune/configs/`
- `data/manifest.json`, `data/stats.json`, `data/corpus/`
- `results/`
- `mission/traces/`, `mission/board.js`

`python -m orbit_agents.eval.build_tasks` rewrites the frozen split and the manifest. Do not run it to refresh the checkout.

`python -m orbit_agents.eval.readme` rewrites the Key results block in `README.md` from `results/`. `python -m orbit_agents.eval.readme --check` fails when that block is stale. The test suite runs the check.

## Scores that need a key or a GPU

`make score-dry-run` checks the frozen-test hash and the `team-eval-v1` tag. It does not read a key and it writes nothing.

```bash
export ORBIT_ANTHROPIC_API_KEY=your-key
make score-v1 MODEL=claude-haiku-5-5
make score-v2 MODEL=claude-haiku-5-5
```

If the variable is unset, those two targets exit before any request and name it. `score-v1` writes `results/v3_api/<model>/team_v1.json`. `score-v2` writes `team_v2.json` beside it. When both are measured, `comparison.json` is written in that directory. Each target then runs `python -m orbit_agents.eval.readme`.

The distill notebook is `notebooks/run_distill_and_score.ipynb`. Open it in Colab on a T4, set the secret `ORBIT_ANTHROPIC_API_KEY`, and run all cells. Commit only `results/v3_qwen1.5b/*.json` and `README.md`.

Training without the teacher is `finetune/colab_train.ipynb`. It scores into the same `results/v3_qwen1.5b/` directory.
