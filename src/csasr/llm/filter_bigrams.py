"""Stage 1b: dedup and filter the raw bigrams.

Paper §3.2.1: "Manual evaluation revealed that some bigrams consisted entirely
of English or Hindi words, or included an English word followed by its Hindi
translation (or vice versa). To address this, we implemented a script that
filters out bad bigrams to ensure each bigram contained both Hindi and English
characters and used the Llama to verify that paired words were not direct
translations. This refinement resulted in 5,477 valid bigrams."

Two filters, in order:

1.  **Script filter** (deterministic, free). Exactly two tokens, one Devanagari
    and one Latin. Kills the "entirely English or Hindi" failures outright.

2.  **Translation check** (LLM). Rejects pairs that are direct translations, e.g.
    `दस्तावेज़ document`. We use self-consistency - 3 samples at temperature 0.7,
    majority vote - because an 8B judge is noisier than the paper's 70B. Batched
    20 pairs per call so this stage does not dominate the token budget.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from ..lid import cs_bigrams, tokenize
from ..manifest import read_jsonl, write_jsonl
from .backend import Sampling, build_backend
from .cache import ResponseCache
from .prompts import translation_check_messages

__all__ = ["script_filter", "parse_verdicts", "main"]

_VERDICT = re.compile(r"^\s*(\d+)\s*[:.)-]\s*(YES|NO)\b", re.IGNORECASE)


def script_filter(phrase: str, *, strict: bool = False) -> tuple[str, str, str] | None:
    """Extract the first Hindi<->English script-crossing pair from `phrase`.

    Returns (bigram_text, hindi_word, english_word) preserving the pair's
    original word order, or None if the phrase crosses no script boundary.

    WHY THIS EXTRACTS RATHER THAN VALIDATES  (deviation D9)
    -------------------------------------------------------
    The paper's Llama-3.3-70B obeyed "each bigram is a couple of different
    words". A far smaller model does not -- Gemma 4 reliably appends a THIRD:

        बुनियादी formatting basics   ->  (बुनियादी, formatting)   the bigram is right there
        नया document create          ->  (नया, document)
        click पर action              ->  (click, पर)              EH switch
        menu bar click               ->  None                     all English, correctly dropped

    Demanding exactly two tokens threw away 10/10 of a real Gemma-4 sample even
    though 7 of them carried a perfectly good switch point. So we pull the first
    script-crossing adjacent pair out of whatever comes back. The PROMPT stays
    verbatim from the paper; only our parser is more forgiving.

    `strict=True` restores the paper-faithful two-token-only behaviour.
    """
    if strict and len(tokenize(phrase)) != 2:
        return None
    for bg in cs_bigrams(phrase):
        if bg.kind == "HE":                       # hi then en
            return f"{bg.first} {bg.second}", bg.first, bg.second
        return f"{bg.first} {bg.second}", bg.second, bg.first   # EH: en then hi
    return None


def parse_verdicts(text: str, n: int) -> dict[int, bool]:
    """Parse '<idx>: YES|NO' lines into {0-based index: is_translation}."""
    out: dict[int, bool] = {}
    for line in text.splitlines():
        m = _VERDICT.match(line)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if 0 <= idx < n:
            out[idx] = m.group(2).upper() == "YES"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True, help="bigrams_raw.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="bigrams_valid.jsonl")
    ap.add_argument("--cache", type=Path, default=Path("data/llm_cache/transcheck.jsonl"))
    ap.add_argument("--backend", default="llamacpp")
    ap.add_argument("--model", default="unsloth/gemma-4-26B-A4B-it-GGUF")
    ap.add_argument("--gguf-file", default="gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
                    help="llama.cpp only: GGUF filename in the --model repo")
    ap.add_argument("--items-per-call", type=int, default=20)
    ap.add_argument("--n-samples", type=int, default=3, help="self-consistency votes")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--skip-translation-check", action="store_true")
    ap.add_argument("--strict-bigrams", action="store_true",
                    help="paper-faithful: require exactly two tokens, do not extract "
                         "the switch pair from a longer phrase")
    # Dedup + script filter are deterministic and cheap, so every shard computes
    # them identically; only the LLM translation check is split.
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args(argv)

    raw = [r["bigram"] for r in read_jsonl(args.raw)]
    unique = sorted(set(raw))
    print(f"[filter] raw={len(raw):,}  unique={len(unique):,}  ({len(unique)/max(len(raw),1):.1%})")

    # Extract the switch-point pair from each phrase. Gemma emits trigrams
    # ("बुनियादी formatting basics"), so the bigram we keep is the EXTRACTED pair,
    # not the raw line. Dedup again afterwards: several phrases can yield the
    # same pair.
    seen: set[str] = set()
    pairs: list[tuple[str, str, str]] = []  # (bigram, hi, en)
    for phrase in unique:
        got = script_filter(phrase, strict=args.strict_bigrams)
        if not got:
            continue
        bigram, hi, en = got
        if bigram in seen:
            continue
        seen.add(bigram)
        pairs.append((bigram, hi, en))

    n_extracted = sum(1 for b, _, _ in pairs if b not in unique)
    print(f"[filter] script filter: {len(pairs):,} distinct switch-point pairs "
          f"from {len(unique):,} unique phrases "
          f"({len(pairs)/max(len(unique),1):.1%})")
    if n_extracted:
        print(f"[filter]   of which {n_extracted:,} were EXTRACTED from longer phrases "
              f"(the model ignored 'a couple of words'; see script_filter docstring)")

    # Split only the LLM translation check. `pairs` is identical in every shard
    # (dedup + script filter are deterministic), so slicing it partitions cleanly.
    if args.num_shards > 1:
        pairs = pairs[args.shard :: args.num_shards]
        print(f"[filter] shard {args.shard}/{args.num_shards}: {len(pairs):,} pairs to check")

    is_translation: dict[str, bool] = {bg: False for bg, _, _ in pairs}

    if not args.skip_translation_check and pairs:
        chunks = [pairs[i : i + args.items_per_call] for i in range(0, len(pairs), args.items_per_call)]
        convs = [translation_check_messages([(hi, en) for _, hi, en in ch]) for ch in chunks]
        backend = build_backend(args.backend, model_id=args.model, filename=args.gguf_file)
        sampling = Sampling(temperature=args.temperature, max_new_tokens=8 * args.items_per_call, seed=args.seed)

        votes: dict[str, Counter] = {bg: Counter() for bg, _, _ in pairs}
        with ResponseCache(args.cache) as cache:
            for s in range(args.n_samples):
                responses = backend.chat(
                    convs, sampling, cache=cache, sample_idx=s,
                    batch_size=args.batch_size, desc=f"transcheck {s+1}/{args.n_samples}",
                )
                for ch, resp in zip(chunks, responses):
                    verdicts = parse_verdicts(resp, len(ch))
                    for i, (bg, _, _) in enumerate(ch):
                        if i in verdicts:
                            votes[bg][verdicts[i]] += 1

        for bg, c in votes.items():
            # Unparseable / no votes => keep the bigram (do not silently drop data).
            is_translation[bg] = c[True] > c[False]

    valid = [
        {"bigram": bg, "hi_word": hi, "en_word": en,
         "checks": {"script": True, "not_translation": not is_translation[bg]}}
        for bg, hi, en in pairs
        if not is_translation[bg]
    ]

    n = write_jsonl(args.out, valid)
    print(f"[filter] valid={n:,}  ({n/max(len(unique),1):.1%} of unique)")
    print("[filter] paper: unique=5,932 -> valid=5,477 (92.3%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
