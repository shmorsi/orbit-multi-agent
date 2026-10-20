PYTHON ?= python3

.PHONY: setup demo doctor doctor-live traces test check finetune-data finetune-dry-run

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

test:
	$(PYTHON) -m pytest

check: test

finetune-data:
	$(PYTHON) finetune/generate_data.py
	$(PYTHON) finetune/leakage.py

finetune-dry-run:
	$(PYTHON) finetune/train.py --config finetune/configs/qlora_small.json --dry-run
