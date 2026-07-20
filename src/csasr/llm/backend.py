"""LLM serving backends.

TWO backends, selected by config `backend:`.

* `LlamaCppBackend` (DEFAULT) -- GGUF via llama-cpp-python, serving
  `unsloth/gemma-4-26B-A4B-it-GGUF` (25.2B / 3.8B-active MoE, ~16 GB Q4_K_M)
  split across both T4s with `tensor_split`. Far stronger than the dense E4B.
  Single-stream (llama.cpp chat completions are sequential), so the run is slow;
  disabling the model's thinking is what keeps it feasible. One process, both
  GPUs -- so no cross-GPU sharding here, unlike the transformers path.

* `TransformersBackend` -- bitsandbytes NF4 + *batched* generate. Kept as an
  alternative (config `backend: transformers`); this is what the earlier E4B runs
  used, and it batches, so it shards across GPUs cleanly.

Why not vLLM on Kaggle: its AWQ kernels require compute capability >= 8.0
(Ampere). The T4 is sm75 and raises "quantization method awq is not supported".

WHY THE TEXT AND AUDIO STAGES LIVE IN SEPARATE NOTEBOOKS
--------------------------------------------------------
`parler-tts` hard-pins `transformers==4.46.1`. The LLM stage needs neither that
pin nor even transformers (llama.cpp uses the GGUF's own chat template), so
`01a_generate_text` and `01b_synthesize_audio` (4.46.1 + parler-tts) are separate
Kaggle sessions with the Hub as the checkpoint between them.
"""

from __future__ import annotations

import abc
import re
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
    "strip_thinking",
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


# Reasoning models wrap their scratch-work in one of a few conventions. We strip
# it defensively no matter which the model uses, so it never reaches the parser
# (and never gets voiced by the TTS downstream). Order matters: block forms first,
# then a "final channel" cut, then stray control tokens.
_THINK_BLOCK = re.compile(
    r"<(think|thinking|reason(?:ing)?)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL
)
# Harmony / channel style: keep only what follows the LAST `final`/`answer` channel.
_FINAL_CHANNEL = re.compile(
    r".*<\|?channel\|?>\s*(?:final|answer)\b.*?(?:<\|?message\|?>)?", re.IGNORECASE | re.DOTALL
)
# A bare opening think token with no close: drop everything up to the first blank
# line or the next turn marker.
_OPEN_THINK = re.compile(
    r"^\s*<\|?/?think\|?>\s*", re.IGNORECASE
)
# Leftover special tokens from any template.
_SPECIALS = re.compile(
    r"<\|[^>]*\|>|</?s>|<(?:end|start)_of_turn>|<pad>", re.IGNORECASE
)


def strip_thinking(text: str) -> str:
    """Remove reasoning scratch-work from a completion, keeping the final answer.

    Handles `<think>...</think>` blocks, harmony-style `<|channel|>final` cuts, a
    dangling open think token, and stray special tokens. Idempotent and safe on
    text that contains none of these.
    """
    if not text:
        return text
    prev = None
    s = text
    while s != prev:
        prev = s
        s = _THINK_BLOCK.sub("", s)
    m = _FINAL_CHANNEL.match(s)
    if m:
        s = s[m.end():]
    s = _OPEN_THINK.sub("", s)
    s = _SPECIALS.sub("", s)
    return s.strip()


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
    """GGUF via llama-cpp-python, split across both T4s with `tensor_split`.

    This is now the DEFAULT LLM path (config `backend: llamacpp`). It serves
    `unsloth/gemma-4-26B-A4B-it-GGUF` -- a 25.2B / 3.8B-active MoE, far stronger
    than the E4B dense model, at ~16 GB in Q4_K_M. That does not fit one T4, so
    `tensor_split=[0.5, 0.5]` spreads the single model across both cards. There
    is therefore ONE process and no cross-GPU sharding: both GPUs hold one model.

    Generation is single-stream (llama.cpp chat completions are sequential), so
    the whole run is slow -- disabling thinking is what keeps it feasible.
    """

    #: exact model + file from the reference notebook
    DEFAULT_REPO = "unsloth/gemma-4-26B-A4B-it-GGUF"
    DEFAULT_FILE = "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"

    def __init__(
        self,
        model_id: str | None = None,
        filename: str | None = None,
        *,
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
        tensor_split: list[float] | None = None,
        top_k: int = 64,
        disable_thinking: bool = True,
        **_ignored,
    ):
        from llama_cpp import Llama

        repo = model_id or self.DEFAULT_REPO
        fname = filename or self.DEFAULT_FILE
        self.model_id = f"{repo}:{fname}"
        self.top_k = top_k
        self.disable_thinking = disable_thinking
        self._think_kwarg_ok: bool | None = None  # learned on first call

        import torch

        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if tensor_split is None and n_gpu > 1:
            tensor_split = [1.0 / n_gpu] * n_gpu  # even split across all GPUs

        print(f"[llm] llama.cpp loading {repo}/{fname}")
        print(f"[llm]   n_gpu_layers={n_gpu_layers}  tensor_split={tensor_split}  n_ctx={n_ctx}")
        self.llm = Llama.from_pretrained(
            repo_id=repo,
            filename=fname,
            n_gpu_layers=n_gpu_layers,
            tensor_split=tensor_split,
            main_gpu=0,
            n_ctx=n_ctx,
            verbose=False,
        )

    def _complete(self, conv: list[dict], sampling: Sampling, *, thinking_off: bool):
        kw = dict(
            messages=conv,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=self.top_k,
            max_tokens=sampling.max_new_tokens,
            seed=sampling.seed,
        )
        if thinking_off:
            # enable_thinking=False via chat_template_kwargs is an OPEN feature in
            # llama-cpp-python (abetlen/llama-cpp-python#2063), so older builds
            # raise TypeError. We try it, remember whether it took, and fall back
            # to stripping the reasoning from the output.
            kw["chat_template_kwargs"] = {"enable_thinking": False}
        return self.llm.create_chat_completion(**kw)

    def _generate(self, convs: list[list[dict]], sampling: Sampling) -> list[str]:
        outs = []
        for conv in convs:
            if self.disable_thinking and self._think_kwarg_ok is None:
                # Probe once: does this llama-cpp-python accept the kwarg?
                try:
                    r = self._complete(conv, sampling, thinking_off=True)
                    self._think_kwarg_ok = True
                    print("[llm] enable_thinking=False accepted by chat template")
                except (TypeError, ValueError) as e:
                    self._think_kwarg_ok = False
                    print(f"[llm] chat_template_kwargs unsupported ({type(e).__name__}); "
                          f"stripping reasoning from output instead")
                    r = self._complete(conv, sampling, thinking_off=False)
            else:
                r = self._complete(conv, sampling,
                                   thinking_off=self.disable_thinking and bool(self._think_kwarg_ok))
            content = r["choices"][0]["message"].get("content") or ""
            outs.append(strip_thinking(content) if self.disable_thinking else content)
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
    """Route kwargs to the selected backend, dropping ones it doesn't take.

    The CLIs pass a superset (model_id, filename, ...); `filename` is meaningful
    only to llama.cpp, so we strip it for the others rather than crash them.
    """
    match name:
        case "transformers":
            kw.pop("filename", None)
            return TransformersBackend(**kw)
        case "llamacpp":
            return LlamaCppBackend(**kw)
        case "echo":
            return EchoBackend(**kw)
        case _:
            raise ValueError(f"unknown backend {name!r}")
