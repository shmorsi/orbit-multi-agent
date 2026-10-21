# Fine-tune protocol

`finetune/generate_data.py` reads `eval/dev.jsonl` and builds extra dialogues by running the corpus tools. It takes no arguments and does not open `eval/test.jsonl`. Two dev tasks, `dev-005` and `dev-017`, are left out of the dialogue files because their questions are near-duplicates of frozen test questions. The manifest records `test_file_read` as false.

`finetune/leakage.py` reads the frozen test file and rejects a dialogue that contains a test id, a test question, a near-duplicate question (normalized SequenceMatcher ratio at least 0.9, or containment), a test answer string, or a test tool snippet. The shared trap sentence is the harness refusal, not a test-specific answer, so that one sentence is ignored. Training answers use "equals" so a frozen "name is value" sentence is not copied.

`finetune/train.py` refuses to open `eval/test.jsonl`. `--dry-run` prints dialogue counts and the leakage flag. It does not print a task score. The trainer's eval set is `finetune/data/tune.jsonl`. Task iteration uses `python -m orbit_agents.eval.run --split tune`. The frozen test is not an iteration split.

QLoRA imports bitsandbytes and requires CUDA. The default Colab config is `finetune/configs/qlora_medium.json`: Qwen2.5-1.5B-Instruct, 4-bit NF4, rank 16, two epochs, sequence length 768, batch 1, gradient accumulation 4. Those knobs are the free-T4 budget (16 GB, about 60 minutes for training plus the three evals). This repository has not timed that run. `finetune/configs/qlora_small.json` remains the 0.5B config from the measured v1 run. `--method lora` is the MPS and CPU alternative: the adapter is a normal LoRA, merged later in fp16 or fp32. CUDA eval loads the base in 4-bit and attaches the adapter. It does not merge into 4-bit. MPS and CPU eval do not import bitsandbytes.

The notebook trains the 1.5B config once, then scores the frozen test once: single base, team base, team plus the adapter. It does not loop over the frozen test. Colab's preinstalled torchao 0.10 breaks `PeftModel.from_pretrained`; the install cell uninstalls torchao.
