"""Mixed-language Error Rate (MER).

The paper cites Zhang et al. (2022), where MER is defined for Mandarin-English:
character-level errors on the Mandarin side, word-level on the English side.
Mandarin needs character units because it is not whitespace-segmented.

Hindi *is* whitespace-segmented, so for Hindi-English that definition collapses
to plain word-level WER over the mixed transcript. That is `mode="word"`, our
default. `mode="hybrid"` implements the literal SEAME reading (characters for
Devanagari words, words for Latin) in case the authors applied it verbatim.

Gate 3 settles which is right empirically: whichever mode puts zero-shot
whisper-large-v2 near the paper's reported 52.0 on the test set.

MER can exceed 100%: it is (S + D + I) / N, and a model that mis-detects the
language and transliterates will insert freely. Observed with whisper-tiny
zero-shot on this test set (MER 125%, CBA 0.0) - that is a real result, not a
bug.
"""

from __future__ import annotations

from typing import Literal, Sequence

import jiwer

from ..lid import Lang, tokenize, word_lang
from ..normalize import normalize

__all__ = ["mer", "to_units"]

Mode = Literal["word", "hybrid"]


def to_units(text: str) -> list[str]:
    """Split into scoring units: Devanagari words -> characters, others -> words."""
    units: list[str] = []
    for tok in tokenize(text):
        if word_lang(tok) is Lang.HI:
            units.extend(tok)
        else:
            units.append(tok)
    return units


def mer(
    refs: Sequence[str],
    hyps: Sequence[str],
    *,
    mode: Mode = "word",
    preset: str = "scoring",
) -> float:
    """Corpus MER as a percentage (total errors / total reference units)."""
    if len(refs) != len(hyps):
        raise ValueError(f"refs/hyps length mismatch: {len(refs)} vs {len(hyps)}")

    r = [normalize(x, preset) for x in refs]
    h = [normalize(x, preset) for x in hyps]

    if mode == "hybrid":
        # jiwer tokenizes on whitespace, so re-join units with spaces. Devanagari
        # characters become individual "words" and are scored at char level.
        r = [" ".join(to_units(x)) for x in r]
        h = [" ".join(to_units(x)) for x in h]

    # jiwer skips empty references; keep the pair only if the reference has content.
    pairs = [(a, b) for a, b in zip(r, h) if a.strip()]
    if not pairs:
        return 0.0
    rr, hh = zip(*pairs)
    return 100.0 * jiwer.wer(list(rr), list(hh))
