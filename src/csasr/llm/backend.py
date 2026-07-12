"""LLM serving backends.

Why not vLLM: its AWQ kernels require compute capability >= 8.0 (Ampere). Kaggle
gives Turing T4s (sm75), which raise
`The quantization method awq is not supported for the current GPU`.

Why not llama.cpp on Kaggle: the high-level Python binding is single-stream.
At ~60 tok/s the ~1.4M output tokens of a full run take >6h, which does not fit
alongside TTS in one 12h session.

So the workhorse is `TransformersBackend`: bitsandbytes NF4 + *batched* generate.

WHY THE TEXT AND AUDIO STAGES LIVE IN SEPARATE NOTEBOOKS
--------------------------------------------------------
`parler-tts` hard-pins `transformers==4.46.1`. **Gemma 4 needs `transformers>=5.5`**
(its config declares `transformers_version: 5.5.0.dev0`). Those cannot coexist in
one process, so `01a_generate_text` (transformers 5.x + Gemma 4) and
`01b_synthesize_audio` (4.46.1 + parler-tts) are separate Kaggle sessions with the
Hub as the checkpoint between them.

`LlamaCppBackend` exists for local smoke tests on small GGUF models; an 8B does
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


def _oom_message(model_id: str, original: str) -> str:
    """Diagnose 'Some modules are dispatched on the CPU or the disk'.

    bitsandbytes refuses to offload a quantized model, so accelerate raises this
    the moment the weights do not fit in FREE GPU memory. On Kaggle the cause is
    almost always a notebook kernel that already holds a model: Jupyter's `Out[]`
    history keeps references alive, so `del model` + `empty_cache()` frees
    nothing, and the subprocess inherits a half-full GPU.
    """
    lines = [
        f"{model_id} does not fit in FREE GPU memory.",
        "",
        f"  bitsandbytes error: {original[:110]}",
        "",
    ]
    try:
        import torch

        if torch.cuda.is_available():
            lines.append("  GPU memory right now:")
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                used = (total - free) / 2**30
                lines.append(
                    f"    cuda:{i}  {free / 2**30:5.1f} GiB free / {total / 2**30:.1f} GiB"
                    f"   ({used:.1f} GiB already in use)"
                )
            lines.append("")
    except Exception:  # noqa: BLE001 - diagnostics must never mask the real error
        pass

    lines += [
        "  Most likely: the NOTEBOOK KERNEL is still holding a model. `del model` does",
        "  not free VRAM in Jupyter -- the Out[] history keeps references alive. Fix:",
        "",
        "    1. Restart the kernel (Run > Restart & clear all).",
        "    2. Make sure you are on the CURRENT notebook: the smoke test must run as a",
        "       subprocess (`run(\"csasr.llm.smoke\", ...)`), NOT load the model in-kernel.",
        "       Re-download notebooks/01a_generate_text.ipynb if yours differs -- `pip",
        "       install` updates the package but NOT the notebook.",
        "",
        "  If the GPU really is empty and it still does not fit, use a smaller model:",
        "    --model google/gemma-4-E2B-it",
    ]
    return "\n".join(lines)


def _merge_system_into_user(messages: list[dict]) -> list[dict]:
    """Fold a system turn into the first user turn.

    Gemma's chat template has no `system` role (Gemma 2 raises outright). Our
    prompts are transcribed verbatim from the paper and DO have one, so fold it
    in rather than dropping it.
    """
    sys_parts = [m["content"] for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    if not sys_parts or not rest:
        return messages or rest
    merged = list(rest)
    merged[0] = {
        "role": merged[0]["role"],
        "content": "\n\n".join([*sys_parts, merged[0]["content"]]),
    }
    return merged


class TransformersBackend(LLMBackend):
    """bitsandbytes NF4 + batched generate. The Kaggle workhorse.

    Three defences, verified against the real `google/gemma-4-E4B-it` config:

    * **Multimodal checkpoints.** Gemma 4 is `Gemma4ForConditionalGeneration`
      (text + vision + audio). It happens to be registered under
      `AutoModelForCausalLM` too, so that binds -- but Gemma 3 4B+ is image-text
      and does NOT. `_load` walks the auto-classes until one binds. Every
      candidate takes text-only `input_ids` and `generate()`s identically, so
      nothing downstream cares which won.
    * **The `system` role.** Gemma 4's template DOES support it, so the paper's
      verbatim system prompt survives intact. Gemma 2's template rejects it
      outright. `_probe_system_role` detects that and folds the system turn into
      the first user turn instead of dropping it.
    * **fp16 overflow.** THIS ONE IS LIVE. Gemma 4's config declares
      `torch_dtype: bfloat16`, and the T4 is Turing -- it has NO bf16. fp16
      activations can exceed 65,504 and go non-finite. We probe the logits after
      loading and, if they are NaN/inf, transparently reload with float32
      compute. Slower, but correct beats fast-and-garbage.
    """

    def __init__(
        self,
        model_id: str = "google/gemma-4-E4B-it",
        *,
        load_in_4bit: bool = True,
        device_map: str = "auto",
        dtype: str = "float16",  # T4 is Turing: no bf16
        healthcheck: bool = True,
    ):
        import torch
        from transformers import AutoTokenizer

        self.model_id = model_id
        self.torch = torch
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only batched generation REQUIRES left padding, otherwise the
        # pad tokens sit between the prompt and the first generated token and
        # every short sequence in the batch is silently corrupted.
        self.tokenizer.padding_side = "left"

        self._supports_system = self._probe_system_role()
        if not self._supports_system:
            print(f"[llm] {model_id}: chat template has no `system` role; "
                  f"folding it into the first user turn")

        self.model = self._load(dtype)

        if healthcheck and not self._logits_finite():
            print(f"[llm] {model_id}: {dtype} logits are NOT finite (fp16 overflow -- "
                  f"Gemma is bf16-native and the T4 has no bf16). Reloading with "
                  f"float32 compute.")
            del self.model
            torch.cuda.empty_cache()
            self.model = self._load("float32")
            if not self._logits_finite():
                raise SystemExit(
                    f"{model_id} produces non-finite logits even in float32. "
                    f"This model cannot be served correctly here; pick another."
                )
            print(f"[llm] {model_id}: float32 compute OK")

    # -- loading ------------------------------------------------------------
    def _load(self, dtype: str):
        """Try the auto-classes in order of specificity.

        Text-only LLMs load as causal LMs. Gemma 3 (4B+) is image-text. Gemma 4
        is `Gemma4ForConditionalGeneration` -- text + vision + AUDIO -- and only
        `AutoModelForMultimodalLM` maps it. All of them accept text-only
        `input_ids` and `generate()` identically, so the rest of the class does
        not care which one won.
        """
        import torch
        import transformers
        from transformers import BitsAndBytesConfig

        quant = None
        if self.load_in_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
            )
        kw = dict(
            quantization_config=quant,
            device_map=self.device_map,
            torch_dtype=getattr(torch, dtype),
        )

        candidates = [
            c for c in ("AutoModelForCausalLM", "AutoModelForMultimodalLM",
                        "AutoModelForImageTextToText")
            if hasattr(transformers, c)
        ]
        errors: list[str] = []
        for name in candidates:
            try:
                model = getattr(transformers, name).from_pretrained(self.model_id, **kw)
            except (ValueError, KeyError, OSError, TypeError) as e:
                msg = str(e)
                # This one is NOT an auto-class problem -- it is "the GPU is full".
                # Every candidate will fail identically, so bail out now with a
                # diagnosis instead of three misleading tracebacks.
                if "dispatched on the CPU" in msg or "disk" in msg.lower():
                    raise SystemExit(_oom_message(self.model_id, msg)) from e
                errors.append(f"{name}: {type(e).__name__}: {msg[:90]}")
                continue
            if name != "AutoModelForCausalLM":
                print(f"[llm] {self.model_id}: loaded via {name} (text-only inputs)")
            model.eval()
            return model

        raise SystemExit(
            f"cannot load {self.model_id} with any auto-class.\n  "
            + "\n  ".join(errors)
            + f"\n\ntransformers={transformers.__version__}. Gemma 4 needs >= 5.5; "
              f"Gemma 3 needs >= 4.50."
        )

    def _probe_system_role(self) -> bool:
        try:
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                tokenize=False, add_generation_prompt=True,
            )
            return True
        except Exception:  # noqa: BLE001 - template raises whatever it likes
            return False

    def _logits_finite(self) -> bool:
        """A forward pass whose logits contain NaN/inf means the dtype is wrong."""
        enc = self.tokenizer("नमस्ते hello", return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model(**enc)
        logits = out.logits if hasattr(out, "logits") else out[0]
        return bool(self.torch.isfinite(logits).all().item())

    # -- generation ---------------------------------------------------------
    def _render(self, conv: list[dict]) -> str:
        if not self._supports_system:
            conv = _merge_system_into_user(conv)
        return self.tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True
        )

    def _generate(self, convs: list[list[dict]], sampling: Sampling) -> list[str]:
        if sampling.seed is not None:
            self.torch.manual_seed(sampling.seed)

        texts = [self._render(c) for c in convs]
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
