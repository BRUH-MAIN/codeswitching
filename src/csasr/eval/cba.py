"""Code-Switch Bigram Accuracy (CBA).

From the paper (§5): CBA "measures correctly recognized bigrams at switch points
relative to their total count in the test set."

Two details the paper pins down implicitly:

1.  **Multiset, not set.** Table 1 reports 4,189 HE bigrams in the test set but
    only 2,347 *unique* ones. The denominator counts tokens, so a bigram
    occurring three times in the reference contributes three to the denominator
    and can contribute up to three to the numerator.

2.  **The hypothesis side is unrestricted.** We ask whether the reference's
    switch-point bigram appears *anywhere* as an adjacent pair in the
    hypothesis, not whether the hypothesis independently classifies it as a
    switch point. Otherwise a model that transliterates Hindi into Latin would
    be scored as if it had no switch points at all rather than as wrong.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..lid import all_bigrams, cs_bigrams
from ..normalize import normalize

__all__ = ["CBAResult", "cba", "reference_cs_counts", "corpus_cs_bigram_stats"]


@dataclass(frozen=True, slots=True)
class CBAResult:
    he_matched: int
    he_total: int
    eh_matched: int
    eh_total: int

    @property
    def he(self) -> float:
        return 100.0 * self.he_matched / self.he_total if self.he_total else 0.0

    @property
    def eh(self) -> float:
        return 100.0 * self.eh_matched / self.eh_total if self.eh_total else 0.0

    @property
    def total(self) -> float:
        m, t = self.he_matched + self.eh_matched, self.he_total + self.eh_total
        return 100.0 * m / t if t else 0.0


def reference_cs_counts(text: str) -> tuple[Counter, Counter]:
    """(HE, EH) multisets of switch-point bigrams in one reference utterance."""
    he: Counter = Counter()
    eh: Counter = Counter()
    for bg in cs_bigrams(text):
        (he if bg.kind == "HE" else eh)[(bg.first, bg.second)] += 1
    return he, eh


def cba(
    refs: Sequence[str],
    hyps: Sequence[str],
    *,
    preset: str = "scoring",
) -> CBAResult:
    """Corpus-level CBA over aligned reference/hypothesis lists."""
    if len(refs) != len(hyps):
        raise ValueError(f"refs/hyps length mismatch: {len(refs)} vs {len(hyps)}")

    def matched(ref_counter: Counter, hyp_counter: Counter) -> int:
        return sum(min(c, hyp_counter[bg]) for bg, c in ref_counter.items())

    he_m = he_t = eh_m = eh_t = 0
    for ref, hyp in zip(refs, hyps):
        ref_he, ref_eh = reference_cs_counts(normalize(ref, preset))
        hyp_bigrams = Counter(all_bigrams(normalize(hyp, preset)))

        he_m += matched(ref_he, hyp_bigrams)
        he_t += sum(ref_he.values())
        eh_m += matched(ref_eh, hyp_bigrams)
        eh_t += sum(ref_eh.values())

    return CBAResult(he_m, he_t, eh_m, eh_t)


def corpus_cs_bigram_stats(texts: Iterable[str], *, preset: str = "raw") -> dict:
    """Token and type counts of switch-point bigrams. Used by Gate 0."""
    he: Counter = Counter()
    eh: Counter = Counter()
    for t in texts:
        a, b = reference_cs_counts(normalize(t, preset))
        he.update(a)
        eh.update(b)
    return {
        "he_tokens": sum(he.values()),
        "eh_tokens": sum(eh.values()),
        "he_types": len(he),
        "eh_types": len(eh),
    }
