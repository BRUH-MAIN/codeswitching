"""Stage 2a: expand each valid bigram into four code-mixed sentences.

Paper §3.2.2: four sentences per bigram, two with English as the matrix language
and two with Hindi. 5,477 x 4 = 21,908 in theory, but "manual review identified
instances where the LLM deviated from the requested number of sentences",
yielding ~16,000 unique sentences.

We validate each sentence: it must contain the bigram (as adjacent tokens) and
carry both scripts. That reproduces the paper's shortfall honestly rather than
padding the count with malformed output.

Matrix language is assigned by majority script, not by trusting the model's
ordering of its own output.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

from ..lid import Lang, all_bigrams, count_words
from ..manifest import read_jsonl, write_jsonl
from ..normalize import normalize
from .backend import Sampling, build_backend
from .cache import ResponseCache
from .prompts import sentence_messages

__all__ = ["parse_sentences", "validate", "matrix_lang", "main"]

_LEAD = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")


def parse_sentences(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = _LEAD.sub("", line.strip())
        if len(line.split()) >= 3:
            out.append(" ".join(line.split()))
    return out


def matrix_lang(sentence: str) -> str:
    c = count_words(sentence)
    return "hi" if c[Lang.HI] >= c[Lang.EN] else "en"


def validate(
    sentence: str, hi_word: str, en_word: str, *, require_order: bool = False
) -> bool:
    """Sentence must contain the bigram adjacently and carry both scripts.

    By default either order is accepted. `hi en` is a Hindi->English switch point
    and `en hi` is an English->Hindi one; the paper wants coverage of both (the
    test set has 4,189 HE and 5,176 EH bigrams), so a reversed placement is still
    a useful training example. Set `require_order=True` to demand the exact
    ordered collocation the bigram was generated as.
    """
    norm = normalize(sentence, "punct")
    c = count_words(norm)
    if c[Lang.HI] == 0 or c[Lang.EN] == 0:
        return False
    bigrams = {(a.lower(), b.lower()) for a, b in all_bigrams(norm)}
    hi, en = hi_word.lower(), en_word.lower()
    if require_order:
        return (hi, en) in bigrams
    return (hi, en) in bigrams or (en, hi) in bigrams


def _sent_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bigrams", type=Path, required=True, help="bigrams_valid.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="sentences.jsonl")
    ap.add_argument("--cache", type=Path, default=Path("data/llm_cache/sentences.jsonl"))
    ap.add_argument("--backend", default="transformers")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-chars", type=int, default=400, help="Whisper label cap is 448 tokens")
    args = ap.parse_args(argv)

    bigrams = list(read_jsonl(args.bigrams))
    print(f"[gen_sentences] {len(bigrams):,} valid bigrams")

    convs = [sentence_messages(b["bigram"]) for b in bigrams]
    backend = build_backend(args.backend, model_id=args.model)
    sampling = Sampling(
        temperature=args.temperature, max_new_tokens=args.max_new_tokens, seed=args.seed
    )

    with ResponseCache(args.cache) as cache:
        responses = backend.chat(
            convs, sampling, cache=cache, batch_size=args.batch_size, desc="sentences"
        )

    seen: set[str] = set()
    rows = []
    n_parsed = n_valid = 0
    for b, resp in zip(bigrams, responses):
        for s in parse_sentences(resp):
            n_parsed += 1
            if len(s) > args.max_chars:
                continue
            if not validate(s, b["hi_word"], b["en_word"]):
                continue
            n_valid += 1
            sid = _sent_id(s)
            if sid in seen:
                continue
            seen.add(sid)
            rows.append({
                "sent_id": sid,
                "text": s,
                "bigram": b["bigram"],
                "matrix_lang": matrix_lang(s),
            })

    n = write_jsonl(args.out, rows)
    theoretical = len(bigrams) * 4
    n_hi = sum(r["matrix_lang"] == "hi" for r in rows)
    print(f"[gen_sentences] parsed={n_parsed:,}  valid={n_valid:,}  unique={n:,}")
    print(f"[gen_sentences] theoretical max {theoretical:,} (4 x {len(bigrams):,})")
    print(f"[gen_sentences] matrix language: hi={n_hi:,}  en={n - n_hi:,}")
    print("[gen_sentences] paper: ~16,000 unique from a theoretical 21,908")
    return 0


if __name__ == "__main__":
    sys.exit(main())
