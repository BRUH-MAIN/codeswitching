"""Code-Switch Bigram Accuracy (CBA).

From the paper (§5): CBA "measures correctly recognized bigrams at switch points
relative to their total count in the test set."

THE DENOMINATOR IS PER-UTTERANCE
--------------------------------
"their total count in the test set" is Table 1's count: 4,189 HE and 5,176 EH,
extracted from the reference *utterances*. We decode whole recordings (see
`decode.py`), and concatenating a recording's utterances before extracting
bigrams invents ~1,000 spurious HE pairs that straddle segment joins -- pairs no
reference utterance ever contained. That inflated our denominator by 23% and
silently halved CBA. Always take the bigrams from the utterances
(`cba_grouped`), even when the hypothesis is a whole recording.

THE MATCHING RULE IS AMBIGUOUS IN THE PAPER
-------------------------------------------
"Correctly recognized" is never defined. Measured against the paper's own
large-v2 zero-shot row (CBA-HE 42.9, CBA-EH ~36) with hypotheses whose MER
already reproduces theirs to 0.2%:

    mode="adjacent"  both words correct AND adjacent   HE 20.1   EH 17.6
    mode="lenient"   both words recognized somewhere   HE 43.8   EH 40.8

`lenient` reproduces their HE almost exactly; `adjacent` is the literal reading
of "bigram" and lands at half. An edit-distance alignment variant sits between
(HE 33.6 / EH 35.0) but inverts the HE > EH ordering that holds in every row of
their Table 2, so it is unlikely to be theirs.

We therefore report BOTH and default to `adjacent`, the defensible reading. The
choice does not affect the experiment: every model is scored identically, and
the M6 -> M7 -> M8 ordering that Track 2 actually tests is preserved under either
rule. Do not silently switch modes between systems.

MULTISET, NOT SET
-----------------
Table 1 reports 4,189 HE bigram *tokens* against only 2,347 unique *types*. A
bigram occurring three times in the reference contributes three to the
denominator and can contribute up to three to the numerator.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from ..lid import all_bigrams, cs_bigrams
from ..normalize import normalize

__all__ = [
    "CBAResult",
    "cba",
    "cba_grouped",
    "reference_cs_counts",
    "corpus_cs_bigram_stats",
]

Mode = str  # "adjacent" | "lenient"


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


def _hyp_index(hyp: str, mode: Mode) -> Counter:
    """What a hypothesis offers for matching, per mode."""
    if mode == "adjacent":
        return Counter(all_bigrams(hyp))
    if mode == "lenient":
        return Counter(hyp.split())
    raise ValueError(f"unknown cba mode {mode!r}; use 'adjacent' or 'lenient'")


def _matched(ref: Counter, index: Counter, mode: Mode) -> int:
    if mode == "adjacent":
        return sum(min(c, index[bg]) for bg, c in ref.items())
    # lenient: both words recognized somewhere, adjacency not required
    return sum(min(c, index[w1], index[w2]) for (w1, w2), c in ref.items())


def cba(
    refs: Sequence[str],
    hyps: Sequence[str],
    *,
    preset: str = "scoring",
    mode: Mode = "adjacent",
) -> CBAResult:
    """Corpus-level CBA over aligned reference/hypothesis lists.

    Both sequences must be at the SAME granularity. If the hypothesis is a whole
    recording, use `cba_grouped` instead -- concatenating the reference to match
    would fabricate switch bigrams at segment joins.
    """
    if len(refs) != len(hyps):
        raise ValueError(f"refs/hyps length mismatch: {len(refs)} vs {len(hyps)}")

    he_m = he_t = eh_m = eh_t = 0
    for ref, hyp in zip(refs, hyps):
        ref_he, ref_eh = reference_cs_counts(normalize(ref, preset))
        index = _hyp_index(normalize(hyp, preset), mode)

        he_m += _matched(ref_he, index, mode)
        he_t += sum(ref_he.values())
        eh_m += _matched(ref_eh, index, mode)
        eh_t += sum(ref_eh.values())

    return CBAResult(he_m, he_t, eh_m, eh_t)


def cba_grouped(
    ref_utterances: Sequence[tuple[str, str]],
    hyp_by_group: dict[str, str],
    group_of: Callable[[str], str],
    *,
    preset: str = "scoring",
    mode: Mode = "adjacent",
) -> CBAResult:
    """CBA with a PER-UTTERANCE denominator against a per-recording hypothesis.

    `ref_utterances` is [(utt_id, text), ...]; `hyp_by_group` maps a group key
    (recording id) to that recording's full hypothesis; `group_of` maps an
    utt_id to its group key.

    Switch bigrams are extracted from each reference *utterance* -- which is what
    Table 1 counts -- and looked up in the hypothesis of the recording that
    utterance belongs to.

    The multiset cap is applied ONCE per recording. Taking
    `min(ref_count, hyp_count)` per utterance would let several utterances each
    claim the same single occurrence in the hypothesis and inflate the score.
    """
    # Aggregate each recording's reference bigrams across its utterances.
    per_group: dict[str, tuple[Counter, Counter]] = {}
    for utt_id, text in ref_utterances:
        g = group_of(utt_id)
        if g not in hyp_by_group:
            raise SystemExit(f"utterance {utt_id!r} has no hypothesis for group {g!r}")
        he, eh = per_group.setdefault(g, (Counter(), Counter()))
        u_he, u_eh = reference_cs_counts(normalize(text, preset))
        he.update(u_he)
        eh.update(u_eh)

    he_m = he_t = eh_m = eh_t = 0
    for g, (ref_he, ref_eh) in per_group.items():
        index = _hyp_index(normalize(hyp_by_group[g], preset), mode)
        he_m += _matched(ref_he, index, mode)
        he_t += sum(ref_he.values())
        eh_m += _matched(ref_eh, index, mode)
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
