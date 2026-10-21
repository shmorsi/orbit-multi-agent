"""Local inference only. bitsandbytes is imported only when CUDA eval loads a 4-bit base model."""

from __future__ import annotations

import json
import os
import urllib.request


def pick_device(torch_module=None) -> str:
    """cuda, then Apple MPS, then CPU. A missing torch module means CPU."""
    if torch_module is None:
        return "cpu"
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        return "cuda"
    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def inference_dtype(device: str, torch_module):
    if device == "cpu":
        return torch_module.float32
    if device == "cuda" and torch_module.cuda.is_bf16_supported():
        return torch_module.bfloat16
    return torch_module.float16


def eval_loads_in_4bit(device: str) -> bool:
    """CUDA eval keeps the base in 4-bit and attaches the adapter. MPS and CPU stay in fp16 or fp32."""
    return device == "cuda"


def torchao_is_compatible(version: str | None) -> bool:
    """Peft rejects torchao 0.10, which Colab preinstalls. Missing torchao is fine."""
    if version is None:
        return True
    parts: list[int] = []
    for piece in version.split("."):
        digits = ""
        for char in piece:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3]) > (0, 16, 0)


def assert_torchao_allows_peft() -> None:
    try:
        import torchao
    except ImportError as exc:
        text = str(exc).lower()
        if "torchao" in text and ("incompatible" in text or "version" in text):
            raise ImportError(
                "torchao is installed at a version this stack cannot load. Run: pip uninstall -y torchao"
            ) from exc
        return
    version = getattr(torchao, "__version__", None)
    if not torchao_is_compatible(str(version) if version else None):
        raise ImportError(
            f"Found an incompatible version of torchao. Found version {version}, "
            "but only versions above 0.16.0 are supported. Run: pip uninstall -y torchao"
        )


class TransformersBackend:
    """Qwen instruct via transformers.

    On CUDA the base model is loaded in 4-bit and an adapter is attached, not merged.
    On MPS and CPU an adapter is merged in fp16 or fp32. bitsandbytes is imported only for the CUDA path.
    """

    def __init__(self, model: str, adapter: str | None = None, max_new_tokens: int = 256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = pick_device(torch)
        self.load_in_4bit = eval_loads_in_4bit(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            loaded = AutoModelForCausalLM.from_pretrained(
                model,
                quantization_config=quantization,
                device_map="auto",
            )
            if adapter:
                assert_torchao_allows_peft()
                from peft import PeftModel

                loaded = PeftModel.from_pretrained(loaded, adapter)
            self.model = loaded
        else:
            dtype = inference_dtype(self.device, torch)
            try:
                loaded = AutoModelForCausalLM.from_pretrained(model, dtype=dtype)
            except TypeError:
                loaded = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype)
            if adapter:
                assert_torchao_allows_peft()
                from peft import PeftModel

                loaded = PeftModel.from_pretrained(loaded, adapter)
                loaded = loaded.merge_and_unload()
            self.model = loaded.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def complete(self, *, role: str, messages: list[dict]) -> str:
        del role
        torch = self.torch
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


class OllamaBackend:
    def __init__(self, model: str, host: str | None = None):
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        if not (self.host.startswith("http://127.0.0.1") or self.host.startswith("http://localhost")):
            raise ValueError("Ollama must stay on localhost")

    def complete(self, *, role: str, messages: list[dict]) -> str:
        del role
        body = json.dumps({"model": self.model, "messages": messages, "stream": False}).encode("utf-8")
        request = urllib.request.Request(
            self.host + "/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload["message"]["content"]


class LlamaCppBackend:
    """OpenAI-compatible llama-server on localhost. Metal is selected by the server, not by this process."""

    def __init__(self, model: str, base_url: str | None = None):
        self.model = model
        self.base_url = (base_url or os.environ.get("LLAMA_SERVER_URL") or "http://127.0.0.1:8081/v1").rstrip("/")
        if not (self.base_url.startswith("http://127.0.0.1") or self.base_url.startswith("http://localhost")):
            raise ValueError("llama-server must stay on localhost")

    def complete(self, *, role: str, messages: list[dict]) -> str:
        del role
        body = json.dumps(
            {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": 256}
        ).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"]
