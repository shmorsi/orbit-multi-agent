# Fine-tune protocol

`finetune/generate_data.py` reads `eval/dev.jsonl` only. It refuses extra arguments. The manifest must say `built_from` is `eval/dev.jsonl only` and `test_file_read` is false.

`finetune/leakage.py` rejects a training dialogue that contains a test id, a test question, or a test tool snippet.

`finetune/train.py` refuses to open `eval/test.jsonl`. `--dry-run` prints dialogue counts and the leakage flag. It does not print a task score.

QLoRA imports bitsandbytes and requires CUDA. That path is `finetune/configs/qlora_small.json` on a free Colab T4. `--method lora` is the MPS and CPU alternative: the adapter is a normal LoRA, merged later in fp16 or fp32. The demo does not import bitsandbytes.

The notebook runs the small config once. It does not loop over the frozen test.
