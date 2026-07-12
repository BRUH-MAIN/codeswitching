"""Smoke test: load the LLM, generate bigrams, show what survives the filter.

Runs in ~1 minute and answers the only questions that matter before committing
hours of GPU time:

  * does the model load at all on this GPU (dtype, quantization, memory)?
  * are its logits finite (Gemma is bf16-native; the T4 has no bf16)?
  * does it emit usable Hindi-English switch points?
  * how many survive the deterministic script filter?

MUST BE RUN AS A SUBPROCESS, never inside the notebook kernel. Jupyter's `Out[]`
history keeps references to anything a cell produced, so `del model` does NOT
free the VRAM -- and the next subprocess then fails with
`Some modules are dispatched on the CPU or the disk`.
"""

from __future__ import annotations

import argparse
import sys

from ..manifest import read_jsonl
from .backend import Sampling, build_backend
from .filter_bigrams import script_filter
from .gen_bigrams import parse_bigrams
from .prompts import bigram_messages

__all__ = ["main"]

FALLBACK_EXAMPLES = [
    "इस spoken tutorial में आपका स्वागत है",
    "यह बुनियादी formatting के बारे में है",
    "अब हम एक नया document बनाएंगे",
    "menu bar पर click करें",
    "फिर आप file को save कर सकते हैं",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--backend", default="transformers")
    ap.add_argument("--train-manifest", default=None,
                    help="sample few-shot exemplars from here instead of the canned ones")
    ap.add_argument("--n-calls", type=int, default=2)
    ap.add_argument("--bigrams-per-call", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)

    examples = FALLBACK_EXAMPLES
    if args.train_manifest:
        import random

        texts = [r["text"] for r in read_jsonl(args.train_manifest) if r.get("text", "").strip()]
        if len(texts) >= 5:
            examples = random.Random(args.seed).sample(texts, 5)

    print(f"[smoke] model  : {args.model}")
    print(f"[smoke] prompts: {args.n_calls} x {args.bigrams_per_call} bigrams\n")

    backend = build_backend(args.backend, model_id=args.model)
    convs = [bigram_messages(examples, n=args.bigrams_per_call) for _ in range(args.n_calls)]
    responses = backend.chat(
        convs,
        Sampling(temperature=args.temperature, max_new_tokens=200, seed=args.seed),
        batch_size=args.n_calls,
        desc="smoke",
    )

    print("\n--- raw completion (first call) ---")
    print(responses[0][:500])

    phrases = [p for r in responses for p in parse_bigrams(r)]
    kept, dropped, extracted = [], [], 0
    for p in phrases:
        got = script_filter(p)
        if got is None:
            dropped.append(p)
            continue
        bigram, hi, en = got
        extracted += bigram != p
        kept.append((p, bigram, hi, en))

    print(f"\n--- {len(phrases)} phrases parsed ---")
    for p, bigram, hi, en in kept:
        note = f"   [extracted from '{p}']" if bigram != p else ""
        print(f"  KEEP  {bigram:<32} hi={hi:<14} en={en}{note}")
    for p in dropped:
        print(f"  drop  {p:<32} (no Hindi<->English switch point)")

    n = len(phrases)
    pct = len(kept) / n if n else 0.0
    print(f"\n[smoke] {len(kept)}/{n} phrases carry a switch point ({pct:.0%})")
    if extracted:
        print(f"[smoke] {extracted} were EXTRACTED from longer phrases -- the model ignored")
        print("        'a couple of words'. That is expected and handled (deviation D9).")

    if not kept:
        print("\n[smoke] FAIL: not one usable bigram. Do not start the full run.")
        print("        Check the raw completion above: garbage/NaN => the fp16 healthcheck")
        print("        did not save us; try --model google/gemma-4-E2B-it or Qwen2.5-7B.")
        return 1

    print("\n[smoke] OK - the model produces usable code-switch bigrams.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
