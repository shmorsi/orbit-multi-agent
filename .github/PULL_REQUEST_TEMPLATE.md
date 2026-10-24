## What changed

## How to check

- [ ] `python -m pytest`
- [ ] `python -m orbit_agents.eval.readme --check`
- [ ] `python docs/build_figures.py --check` if a diagram or a measured number changed

## Results

- [ ] This does not edit `eval/test.jsonl`, `eval/manifest.json`, or a measured file under `results/`
- [ ] Any new number on the README is generated from those files, or the slot stays pending
- [ ] This does not claim a task-success separation the paired interval does not support
