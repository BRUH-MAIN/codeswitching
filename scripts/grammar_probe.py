"""Is the synthetic code-switching structurally the same *kind* as the real corpus?

Evidence for deviation D14. Real Hindi-English tutorial speech is *insertional*:
a Hindi matrix frame with short English technical terms dropped in. If the
synthetic text is instead *alternational* -- whole English clauses alternating
with Hindi -- that is exactly what reads as bad cross-language grammar, and it
trains Whisper's decoder (which is an autoregressive LM) on the wrong register.

Measured, not asserted: matrix-language split, contiguous same-language run
lengths, and whether the synthetic switch points occur in real speech at all.

    python scripts/grammar_probe.py \
        --real manifests/mucs_train.jsonl \
        --synth temp/sentences_15k.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csasr.lid import Lang, count_words, cs_bigrams, tokenize, word_lang
from csasr.manifest import read_jsonl
from csasr.normalize import normalize


def runs(text: str) -> list[tuple[Lang, int]]:
    """Contiguous same-language runs: [(lang, n_words), ...].

    Run length is the whole point: a 1-word English run is an insertion, a
    5-word run is an alternation. Words that are neither HI nor EN (digits,
    stray punctuation) are skipped rather than breaking a run, so `account
    5 बनाना` still reads as one English run, not two.
    """
    out: list[tuple[Lang, int]] = []
    cur: Lang | None = None
    n = 0
    for w in tokenize(text):
        lang = word_lang(w)
        if lang not in (Lang.HI, Lang.EN):
            continue
        if lang == cur:
            n += 1
        else:
            if cur is not None:
                out.append((cur, n))
            cur, n = lang, 1
    if cur is not None:
        out.append((cur, n))
    return out


def matrix_lang(text: str) -> str:
    """Which language supplies the sentence frame -- by simple word majority."""
    c = count_words(text)
    return "hi" if c[Lang.HI] >= c[Lang.EN] else "en"


def profile(name: str, corpus: list[str]) -> None:
    en_runs: list[int] = []
    hi_runs: list[int] = []
    mats: Counter[str] = Counter()

    for raw in corpus:
        text = normalize(raw, "scoring")
        mats[matrix_lang(text)] += 1
        for lang, n in runs(text):
            (en_runs if lang is Lang.EN else hi_runs).append(n)

    total = sum(mats.values())
    print(f"\n=== {name}  ({total:,} sentences) ===")
    print(f"  matrix language     : {mats['hi'] / total:5.1%} Hindi   "
          f"{mats['en'] / total:5.1%} English")
    for label, rs in (("English", en_runs), ("Hindi  ", hi_runs)):
        if not rs:
            continue
        one = 100 * sum(1 for x in rs if x == 1) / len(rs)
        four = 100 * sum(1 for x in rs if x >= 4) / len(rs)
        print(f"  {label} runs        : mean {sum(rs) / len(rs):4.2f} words   "
              f"1-word {one:4.1f}%   >=4-word {four:4.1f}%")


def switch_overlap(real: list[str], synth: list[str]) -> None:
    """Do our switch points occur in real speech?

    Low overlap is *expected* and arguably intended -- generating unseen switch
    points is the method's whole purpose. Reported for completeness, not alarm.
    """
    def bigrams(corpus: list[str]) -> Counter[tuple[str, str]]:
        c: Counter[tuple[str, str]] = Counter()
        for t in corpus:
            for b in cs_bigrams(normalize(t, "scoring")):
                c[(b.first, b.second)] += 1
        return c

    real_bg, syn_bg = bigrams(real), bigrams(synth)
    shared = set(syn_bg) & set(real_bg)
    tok = sum(syn_bg[b] for b in shared) / max(sum(syn_bg.values()), 1)

    print("\n=== switch bigrams: do ours occur in real speech? ===")
    print(f"  synthetic switch-bigram types      : {len(syn_bg):,}")
    print(f"  also seen in the real corpus       : {len(shared):,} "
          f"({len(shared) / max(len(syn_bg), 1):.1%} of types)")
    print(f"  token-weighted                     : {tok:.1%}")
    print(f"  -> {1 - tok:.0%} of our switch points are pairs a real speaker "
          f"never produced (expected: novel coverage is the method's purpose)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", default="manifests/mucs_train.jsonl")
    ap.add_argument("--synth", default="manifests/sentences.jsonl")
    args = ap.parse_args()

    real = [r["text"] for r in read_jsonl(args.real)]
    synth = [r["text"] for r in read_jsonl(args.synth)]

    profile("REAL MUCS (spoken tutorials)", real)
    profile("SYNTHETIC (LLM-generated)", synth)
    switch_overlap(real, synth)

    print("\n=== what the paper's own prompt ASKS for ===")
    from csasr.llm.prompts import SENTENCE_USER_TEMPLATE
    print("  " + SENTENCE_USER_TEMPLATE.format(bigram="<bigram>").strip())
    print("\n  -> a 50/50 matrix split, against a test set that is 88/12 Hindi.")
    print("     The mismatch is inherited from the paper, not introduced here.")


if __name__ == "__main__":
    main()
