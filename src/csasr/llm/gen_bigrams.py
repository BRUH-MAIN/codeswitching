"""Stage 1a: few-shot generation of Hindi-English code-mixed bigrams.

Paper §3.2.1: Llama is "repeatedly prompted with batches of five Hindi-English
sentences from the training set using a few-shot approach". 44,657 raw bigrams
were produced, collapsing to 5,932 unique (13.3%).

Parsing is deliberately lenient. The raw count is a *yield* statistic we compare
against the paper (Gate 1); throwing away malformed lines here would make our
13.3% dedup rate incomparable with theirs. Validation happens in
`filter_bigrams.py`.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

from ..manifest import read_jsonl, write_jsonl
from .backend import Sampling, build_backend
from .cache import ResponseCache
from .prompts import bigram_messages

__all__ = ["parse_bigrams", "main"]

# Strip list markers the model adds despite being told not to: "1. ", "- ", "* ".
_LEAD = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")
_TRAIL_PUNCT = re.compile(r"[;,.]+\s*$")


def parse_bigrams(text: str) -> list[str]:
    """Pull candidate bigrams out of one completion. Lenient by design."""
    out: list[str] = []
    for line in text.splitlines():
        line = _LEAD.sub("", line.strip())
        line = _TRAIL_PUNCT.sub("", line).strip()
        if not line:
            continue
        # The model sometimes emits several bigrams on one line, separated by ';'
        for part in line.split(";"):
            part = " ".join(part.split())
            if part and " " in part:
                out.append(part)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("data/llm_cache/bigrams.jsonl"))
    ap.add_argument("--backend", default="transformers")
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--n-calls", type=int, default=4466, help="~44,657 / 10 per call")
    ap.add_argument("--bigrams-per-call", type=int, default=10)
    ap.add_argument("--sentences-per-prompt", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    # Kaggle gives 2x T4 but device_map="auto" fits the whole model on ONE card
    # and leaves the other idle. Run one process per GPU:
    #   CUDA_VISIBLE_DEVICES=0 ... --shard 0 --num-shards 2 &
    #   CUDA_VISIBLE_DEVICES=1 ... --shard 1 --num-shards 2 &
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args(argv)

    texts = [r["text"] for r in read_jsonl(args.train_manifest) if r.get("text", "").strip()]
    if len(texts) < args.sentences_per_prompt:
        raise SystemExit(f"only {len(texts)} training sentences available")
    print(f"[gen_bigrams] {len(texts):,} in-domain sentences available for few-shot prompting")

    # Build ALL conversations first (deterministic given --seed), then keep only
    # this shard's slice. Every shard therefore draws the same exemplars for a
    # given call_id, and call_ids stay globally unique across shards.
    rng = random.Random(args.seed)
    all_convs = [
        bigram_messages(rng.sample(texts, args.sentences_per_prompt), n=args.bigrams_per_call)
        for _ in range(args.n_calls)
    ]
    call_ids = list(range(args.n_calls))[args.shard :: args.num_shards]
    convs = [all_convs[i] for i in call_ids]
    if args.num_shards > 1:
        print(f"[gen_bigrams] shard {args.shard}/{args.num_shards}: "
              f"{len(convs):,} of {args.n_calls:,} calls")

    backend = build_backend(args.backend, model_id=args.model)
    sampling = Sampling(
        temperature=args.temperature, max_new_tokens=args.max_new_tokens, seed=args.seed
    )

    with ResponseCache(args.cache) as cache:
        responses = backend.chat(
            convs, sampling, cache=cache, batch_size=args.batch_size, desc="bigrams"
        )

    rows = []
    for call_id, resp in zip(call_ids, responses):
        for bg in parse_bigrams(resp):
            rows.append({"bigram": bg, "call_id": call_id})

    n = write_jsonl(args.out, rows)
    uniq = len({r["bigram"] for r in rows})
    print(f"[gen_bigrams] raw={n:,}  unique={uniq:,}  ({uniq/max(n,1):.1%} survive dedup)")
    print("[gen_bigrams] paper: raw=44,657  unique=5,932  (13.3%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
