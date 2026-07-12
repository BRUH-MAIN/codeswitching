"""Repair a sentences manifest in place -- no LLM re-run.

Gemma 4 prefixes each sentence with its matrix language:

    English: Many software programs have different aliases निर्धारित for commands.
    Hindi: कई सॉफ्टवेयर प्रोग्रामों में कमांड के लिए अलग-अलग aliases निर्धारित होते हैं।

An older `parse_sentences` stripped only bullets and numbering, so those labels
landed in `sentences.jsonl`. Left alone they would be **spoken by Parler-TTS**
("English colon, many software programs...") and then **learned by Whisper**,
quietly poisoning the whole synthetic corpus.

The sentences are already generated and pushed, so this re-cleans the text,
re-validates it against its bigram, recomputes the matrix language (the leading
"English:" biased it), de-duplicates, and writes a fixed manifest. Seconds, not
hours.

    python -m csasr.llm.fix_sentences --in sentences.jsonl --out sentences.jsonl
    python -m csasr.llm.fix_sentences --hf RohanRamesh/hi-en-synth-cs --out sentences.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ..manifest import read_jsonl, write_jsonl
from .gen_sentences import clean_sentence, matrix_lang, validate

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--in", dest="inp", type=Path, help="existing sentences.jsonl")
    src.add_argument("--hf", help="pull the `sentences` config from this dataset repo")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bigrams", type=Path, default=None,
                    help="bigrams_valid.jsonl, to re-validate against hi/en words")
    ap.add_argument("--max-chars", type=int, default=400)
    args = ap.parse_args(argv)

    if args.hf:
        from datasets import load_dataset

        ds = load_dataset(args.hf, "sentences", split="train",
                          token=os.environ.get("HF_TOKEN"))
        rows = [dict(r) for r in ds]
    else:
        rows = list(read_jsonl(args.inp))
    print(f"[fix] {len(rows):,} sentences in")

    words: dict[str, tuple[str, str]] = {}
    if args.bigrams:
        for b in read_jsonl(args.bigrams):
            words[b["bigram"]] = (b["hi_word"], b["en_word"])

    seen: set[str] = set()
    out, changed, dropped, relabelled = [], 0, 0, 0
    for r in rows:
        old = r["text"]
        new = clean_sentence(old)
        if not new or len(new) > args.max_chars or len(new.split()) < 3:
            dropped += 1
            continue

        # The bigram must survive the cleaning.
        hi_en = words.get(r.get("bigram", ""))
        if hi_en and not validate(new, *hi_en):
            dropped += 1
            continue

        if new in seen:
            dropped += 1
            continue
        seen.add(new)

        changed += new != old
        ml = matrix_lang(new)          # "English:" biased the old label
        relabelled += ml != r.get("matrix_lang")
        out.append({**r, "text": new, "matrix_lang": ml})

    n = write_jsonl(args.out, out)
    print(f"[fix] cleaned {changed:,} sentences (stripped 'English:'/'Hindi:' labels)")
    print(f"[fix] matrix language corrected on {relabelled:,}")
    print(f"[fix] dropped {dropped:,} (empty, duplicate, or bigram destroyed)")
    print(f"[fix] {n:,} sentences out -> {args.out}")

    if changed:
        print("\n  before/after sample:")
        for r_old, r_new in zip(rows, out):
            if r_old["text"] != r_new["text"]:
                print(f"    -  {r_old['text'][:78]}")
                print(f"    +  {r_new['text'][:78]}")
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
