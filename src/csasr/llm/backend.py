"""LLM serving backends.

Why not vLLM: its AWQ kernels require compute capability >= 8.0 (Ampere). Kaggle
gives Turing T4s (sm75), which raise
`The quantization method awq is not supported for the current GPU`.

Why not llama.cpp on Kaggle: the high-level Python binding is single-stream.
At ~60 tok/s the ~1.4M output tokens of a full run take >6h, which does not fit
alongside TTS in one 12h session.

So the workhorse is `TransformersBackend`: bitsandbytes NF4 + *batched* generate.
Llama-3.1 is supported by transformers 4.46.1, the version parler-tts pins, so
the LLM and the TTS model can share one environment.

`LlamaCppBackend` exists for local smoke tests on small GGUF models; the 8B does
not fit the 4GB laptop card.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Sequence

from tqdm import tqdm

from .cache import ResponseCache

__all__ = [
    "Sampling",
    "LLMBackend",
    "TransformersBackend",
    "LlamaCppBackend",
    "EchoBackend",
    "build_backend",
]


@dataclass(frozen=True, slots=True)
class Sampling:
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 256
    seed: int | None = None

    def as_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
        }


class LLMBackend(abc.ABC):
    """A backend turns chat conversations into completion strings."""

    model_id: str

    @abc.abstractmethod
    def _generate(self, convs: list[list[dict]], sampling: Sampling) -> list[str]:
        """Generate one completion per conversation. Order must be preserved."""

    def chat(
        self,
        convs: Sequence[list[dict]],
        sampling: Sampling,
        *,
        cache: ResponseCache | None = None,
        sample_idx: int = 0,
        batch_size: int = 16,
        desc: str = "llm",
    ) -> list[str]:
        """Cached, batched chat. Cache hits never reach the model."""
        convs = list(convs)
        out: list[str | None] = [None] * len(convs)
        todo: list[int] = []

        if cache is not None:
            params = sampling.as_dict()
            keys = [
                ResponseCache.key(self.model_id, c, params, sample_idx) for c in convs
            ]
            for i, k in enumerate(keys):
                hit = cache.get(k)
                if hit is None:
                    todo.append(i)
                else:
                    out[i] = hit
        else:
            keys = []
            todo = list(range(len(convs)))

        if todo:
            hits = len(convs) - len(todo)
            bar = tqdm(total=len(todo), desc=f"{desc} (cached {hits})", unit="req")
            for start in range(0, len(todo), batch_size):
                idxs = todo[start : start + batch_size]
                responses = self._generate([convs[i] for i in idxs], sampling)
                for i, r in zip(idxs, responses):
                    out[i] = r
                    if cache is not None:
                        cache.put(keys[i], r)
                bar.update(len(idxs))
            bar.close()

        return [o or "" for o in out]


class TransformersBackend(LLMBackend):
    """bitsandbytes NF4 + batched generate. The Kaggle workhorse."""

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        *,
        load_in_4bit: bool = True,
        device_map: str = "auto",
        dtype: str = "float16",  # T4 is Turing: no bf16
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.model_id = model_id
        self.torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only batched generation REQUIRES left padding, otherwise the
        # pad tokens sit between the prompt and the first generated token and
        # every short sequence in the batch is silently corrupted.
        self.tokenizer.padding_side = "left"

        quant = None
        if load_in_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant,
            device_map=device_map,
            torch_dtype=getattr(torch, dtype),
        )
        self.model.eval()

    def _generate(self, convs: list[list[dict]], sampling: Sampling) -> list[str]:
        if sampling.seed is not None:
            self.torch.manual_seed(sampling.seed)

        texts = [
            self.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            for c in convs
        ]
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.model.device)

        with self.torch.inference_mode():
            out = self.model.generate(
                **enc,
                do_sample=sampling.temperature > 0,
                temperature=sampling.temperature or None,
                top_p=sampling.top_p,
                max_new_tokens=sampling.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Left padding means every sequence's completion starts at the same
        # offset: the padded prompt length.
        gen = out[:, enc["input_ids"].shape[1] :]
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)


class LlamaCppBackend(LLMBackend):
    """GGUF via llama-cpp-python. Single-stream; for local smoke tests only."""

    def __init__(
        self,
        model_id: str = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename: str = "*Q4_K_M.gguf",
        *,
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
    ):
        from llama_cpp import Llama

        self.model_id = f"{model_id}:{filename}"
        self.llm = Llama.from_pretrained(
            repo_id=model_id,
            filename=filename,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
        )

    def _generate(self, convs: list[list[dict]], sampling: Sampling) -> list[str]:
        outs = []
        for conv in convs:
            r = self.llm.create_chat_completion(
                messages=conv,
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                max_tokens=sampling.max_new_tokens,
                seed=sampling.seed,
            )
            outs.append(r["choices"][0]["message"]["content"] or "")
        return outs


class EchoBackend(LLMBackend):
    """Deterministic stub for unit tests. Returns a canned response per prompt."""

    def __init__(self, responses: dict[str, str] | None = None, default: str = "", **_ignored):
        self.model_id = "echo"
        self.responses = responses or {}
        self.default = default
        self.calls: list[list[dict]] = []

    def _generate(self, convs: list[list[dict]], sampling: Sampling) -> list[str]:
        self.calls.extend(convs)
        return [
            self.responses.get(c[-1]["content"], self.default) for c in convs
        ]


def build_backend(name: str, **kw) -> LLMBackend:
    match name:
        case "transformers":
            return TransformersBackend(**kw)
        case "llamacpp":
            return LlamaCppBackend(**kw)
        case "echo":
            return EchoBackend(**kw)
        case _:
            raise ValueError(f"unknown backend {name!r}")
