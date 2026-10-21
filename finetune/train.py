"""QLoRA on CUDA, or plain LoRA on MPS/CPU. The frozen test file is never an input.

transformers 5 dropped TrainingArguments.warmup_ratio. When that parameter is
absent, the configured ratio is passed as warmup_steps. eval_strategy is used
when the installed signature has it, otherwise evaluation_strategy.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "finetune"))

from leakage import check as leakage_check  # noqa: E402
from leakage import overlap_error  # noqa: E402
from leakage import _read as read_jsonl_rows  # noqa: E402

TRAIN_PATH = ROOT / "finetune" / "data" / "train.jsonl"
TUNE_PATH = ROOT / "finetune" / "data" / "tune.jsonl"
DATA_ROOT = (ROOT / "finetune" / "data").resolve()
FORBIDDEN = (ROOT / "eval" / "test.jsonl").resolve()


def assert_training_file(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == FORBIDDEN or resolved.name == "test.jsonl":
        raise SystemExit(f"refusing to train on {resolved}")
    if resolved != DATA_ROOT and DATA_ROOT not in resolved.parents:
        raise SystemExit(f"refusing to train on {resolved}")
    if resolved.name not in {"train.jsonl", "tune.jsonl"}:
        raise SystemExit(f"refusing to train on {resolved}")
    return resolved


def data_paths(config: dict) -> tuple[Path, Path]:
    train_rel = config.get("train_file") or "finetune/data/train.jsonl"
    tune_rel = config.get("tune_file") or "finetune/data/tune.jsonl"
    return assert_training_file(ROOT / train_rel), assert_training_file(ROOT / tune_rel)


def guard_rows(train_rows: list[dict], tune_rows: list[dict], train_path: Path, tune_path: Path) -> None:
    from leakage import TEST

    error = overlap_error(train_rows, tune_rows, read_jsonl_rows(TEST))
    if error:
        raise SystemExit(error)
    if train_path.resolve() == TRAIN_PATH.resolve() and tune_path.resolve() == TUNE_PATH.resolve():
        leakage_check()


def load_rows(path: Path) -> list[dict]:
    assert_training_file(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def map_training_kwargs(raw: dict, parameters: set[str]) -> dict:
    kwargs = dict(raw)
    if "warmup_ratio" in kwargs and "warmup_ratio" not in parameters:
        if "warmup_steps" in parameters:
            kwargs["warmup_steps"] = kwargs.pop("warmup_ratio")
        else:
            kwargs.pop("warmup_ratio", None)
    if "eval_strategy" in kwargs and "eval_strategy" not in parameters:
        if "evaluation_strategy" in parameters:
            kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        else:
            kwargs.pop("eval_strategy", None)
    return {key: value for key, value in kwargs.items() if key in parameters}


def chat_text(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def seq_length_error(lengths: list[int], max_length: int) -> str | None:
    """Error when max_seq_length would cut the assistant turn off the dialog."""
    over = [length for length in lengths if length > max_length]
    if not over:
        return None
    longest = max(lengths)
    return (
        f"{len(over)} of {len(lengths)} dialogs exceed max_seq_length {max_length} "
        f"(longest {longest}). "
        "Truncation cuts the assistant turn and the assistant mask is empty. "
        f"Raise max_seq_length to at least {longest}."
    )


def guard_seq_length(tokenizer, rows: list[dict], max_length: int) -> None:
    """Refuse a max_seq_length shorter than the dialogs about to be trained.

    Distill rows carry the role system prompt plus the checker contract. With
    the Qwen2.5 tokenizer those dialogs are about 951 tokens at the median and
    about 1,400 at the max, so 768 leaves the assistant mask empty.
    """
    lengths = []
    for row in rows:
        encoded = tokenizer(chat_text(tokenizer, row["messages"]), add_special_tokens=False)
        lengths.append(len(encoded["input_ids"]))
    error = seq_length_error(lengths, max_length)
    if error:
        raise SystemExit(error)


def mask_assistant_tokens(tokenizer, messages: list[dict], max_length: int) -> dict:
    text = chat_text(tokenizer, messages)
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    if "offset_mapping" not in encoding:
        raise RuntimeError("tokenizer did not return offsets; refusing to train on user tokens")
    n_tokens = len(encoding["input_ids"])
    if n_tokens > max_length:
        raise RuntimeError(
            f"assistant mask is empty: dialog is {n_tokens} tokens and max_seq_length is {max_length}. "
            "Truncation cuts the assistant turn. "
            f"Raise max_seq_length to at least {n_tokens}."
        )
    start_marker = "<|im_start|>assistant\n"
    end_marker = "<|im_end|>"
    spans = []
    cursor = 0
    while True:
        start = text.find(start_marker, cursor)
        if start < 0:
            break
        content = start + len(start_marker)
        end = text.find(end_marker, content)
        if end < 0:
            break
        spans.append((content, end + len(end_marker)))
        cursor = end + len(end_marker)
    if not spans:
        raise RuntimeError("no assistant spans in the chat template; refusing to train")
    labels = []
    for token_id, (char_start, char_end) in zip(encoding["input_ids"], encoding["offset_mapping"], strict=True):
        inside = any(span_start <= char_start and char_end <= span_end for span_start, span_end in spans)
        labels.append(token_id if inside and char_end > char_start else -100)
    if all(label == -100 for label in labels):
        raise RuntimeError("assistant mask is empty")
    return {
        "input_ids": encoding["input_ids"],
        "labels": labels,
        "attention_mask": encoding["attention_mask"],
    }


def plot_loss(log_path: Path, figure_path: Path) -> bool:
    rows = []
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if "loss" in row:
                    rows.append(row)
    if len(rows) < 2:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    axis.plot([row["step"] for row in rows], [row["loss"] for row in rows], color="#f4f4f4", linewidth=1.6)
    axis.set_xlabel("step")
    axis.set_ylabel("training loss")
    axis.set_title("LoRA training loss")
    figure.patch.set_facecolor("#000000")
    axis.set_facecolor("#000000")
    axis.tick_params(colors="#f4f4f4")
    axis.xaxis.label.set_color("#f4f4f4")
    axis.yaxis.label.set_color("#f4f4f4")
    axis.title.set_color("#f4f4f4")
    for spine in axis.spines.values():
        spine.set_color("#ffffff")
    figure.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=140)
    plt.close(figure)
    return True


def train(config: dict, output_dir: Path) -> None:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    train_path, tune_path = data_paths(config)
    method = config["method"]
    if method == "qlora" and not torch.cuda.is_available():
        raise SystemExit(
            "QLoRA needs a CUDA GPU and is optional. Use finetune/colab_train.ipynb, "
            "or --method lora for fp16/fp32 on MPS or CPU. The demo does not need bitsandbytes."
        )
    train_rows = load_rows(train_path)
    tune_rows = load_rows(tune_path)
    guard_rows(train_rows, tune_rows, train_path, tune_path)
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_length = int(config["max_seq_length"])
    guard_seq_length(tokenizer, train_rows + tune_rows, max_length)

    if method == "qlora":
        from transformers import BitsAndBytesConfig

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            quantization_config=quantization,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        from orbit_agents.backends import inference_dtype, pick_device

        device = pick_device(torch)
        dtype = inference_dtype(device, torch)
        try:
            model = AutoModelForCausalLM.from_pretrained(config["base_model"], dtype=dtype)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(config["base_model"], torch_dtype=dtype)
        model = model.to(device)
    model = get_peft_model(
        model,
        LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=config["target_modules"],
        ),
    )
    train_dataset = [mask_assistant_tokens(tokenizer, row["messages"], max_length) for row in train_rows]
    tune_dataset = [mask_assistant_tokens(tokenizer, row["messages"], max_length) for row in tune_rows]
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"

    from transformers import TrainerCallback

    class JsonlCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs:
                return
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"step": int(state.global_step), "epoch": state.epoch, "loss": float(logs["loss"])})
                    + "\n"
                )

    raw = {
        "output_dir": str(output_dir / "trainer"),
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "num_train_epochs": config["num_train_epochs"],
        "learning_rate": config["learning_rate"],
        "warmup_ratio": config["warmup_ratio"],
        "logging_steps": config["logging_steps"],
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "bf16": method == "qlora" and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": method == "qlora" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "gradient_checkpointing": bool(config.get("gradient_checkpointing", True)),
        "report_to": [],
        "seed": config["seed"],
        "remove_unused_columns": False,
    }
    arguments = TrainingArguments(**map_training_kwargs(raw, set(inspect.signature(TrainingArguments.__init__).parameters)))
    from transformers import DataCollatorForSeq2Seq

    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=tune_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, label_pad_token_id=-100),
        callbacks=[JsonlCallback()],
    )
    trainer.train()
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    plotted = plot_loss(log_path, output_dir / "loss_curve.png")
    (output_dir / "RUN.json").write_text(
        json.dumps(
            {
                "status": "trained",
                "kind": "adapter_only",
                "base_model": config["base_model"],
                "method": method,
                "n_train_dialogs": len(train_rows),
                "n_tune_dialogs": len(tune_rows),
                "loss_curve_written": plotted,
                "task_metrics": None,
                "note": "An adapter was saved. This is not a task-success result. Score the frozen test once, separately.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a local instruct model on dev trajectories")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=["qlora", "lora"], default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "finetune" / "outputs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.method:
        config["method"] = args.method
    train_path, tune_path = data_paths(config)
    train_rows = load_rows(train_path)
    tune_rows = load_rows(tune_path)
    guard_rows(train_rows, tune_rows, train_path, tune_path)
    leakage = {"ok": True}
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "base_model": config["base_model"],
                    "method": config["method"],
                    "n_train_dialogs": len(train_rows),
                    "n_tune_dialogs": len(tune_rows),
                    "leakage_ok": leakage["ok"],
                    "test_file_read": False,
                },
                indent=2,
            )
        )
        return
    train(config, args.output)


if __name__ == "__main__":
    main()
