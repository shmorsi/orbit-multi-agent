PYTHON ?= python3
PROVIDER ?= anthropic
MODEL ?= claude-haiku-5-5
MAX_COST_USD ?= 5

.PHONY: setup demo doctor doctor-live traces traces-v2 test check readme score-v1 score-v2 score-dry-run finetune-data finetune-dry-run distill-dry-run

setup:
	$(PYTHON) -m pip install -e ".[dev]" -r requirements-infer.txt

demo:
	$(PYTHON) -m orbit_agents.serve

doctor:
	$(PYTHON) -m orbit_agents.doctor

doctor-live:
	$(PYTHON) -m orbit_agents.doctor --live

traces:
	$(PYTHON) -m orbit_agents.trace

traces-v2:
	$(PYTHON) -m orbit_agents.trace --team v2

# Score a team once on the frozen test with an API or local policy. Refuses an already measured config.
score-dry-run:
	$(PYTHON) -m orbit_agents.eval.score_team --team v1 --provider $(PROVIDER) --model $(MODEL) --dry-run
	$(PYTHON) -m orbit_agents.eval.score_team --team v2 --provider $(PROVIDER) --model $(MODEL) --dry-run

score-v1:
	@if [ "$(PROVIDER)" = "anthropic" ] && [ -z "$$ORBIT_ANTHROPIC_API_KEY" ]; then \
	  echo "Set ORBIT_ANTHROPIC_API_KEY to an Anthropic API key. Nothing is read from a file. make score-dry-run does not need a key." >&2; \
	  exit 1; \
	fi
	$(PYTHON) -m orbit_agents.eval.score_team --team v1 --provider $(PROVIDER) --model $(MODEL) --max-cost-usd $(MAX_COST_USD)
	$(PYTHON) -m orbit_agents.eval.readme

score-v2:
	@if [ "$(PROVIDER)" = "anthropic" ] && [ -z "$$ORBIT_ANTHROPIC_API_KEY" ]; then \
	  echo "Set ORBIT_ANTHROPIC_API_KEY to an Anthropic API key. Nothing is read from a file. make score-dry-run does not need a key." >&2; \
	  exit 1; \
	fi
	$(PYTHON) -m orbit_agents.eval.score_team --team v2 --provider $(PROVIDER) --model $(MODEL) --max-cost-usd $(MAX_COST_USD)
	$(PYTHON) -m orbit_agents.eval.readme

test:
	$(PYTHON) -m pytest

check: test

readme:
	$(PYTHON) -m orbit_agents.eval.readme

finetune-data:
	$(PYTHON) finetune/generate_data.py
	$(PYTHON) finetune/leakage.py

finetune-dry-run:
	$(PYTHON) finetune/train.py --config finetune/configs/qlora_small.json --dry-run

distill-dry-run:
	$(PYTHON) finetune/distill_teacher.py --dry-run
