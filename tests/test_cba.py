"""CBA must use multiset semantics.

The paper counts 4,189 HE bigram *tokens* against 2,347 unique *types* in the
test set, so a bigram repeated in the reference must be able to contribute more
than once to both numerator and denominator. A set-based implementation would
silently under-count the denominator and inflate CBA.
"""

import pytest

from csasr.eval.cba import cba, cba_grouped, corpus_cs_bigram_stats, reference_cs_counts
from csasr.eval.mer import mer


class TestReferenceCounts:
    def test_repeated_bigram_counted_twice(self):
        he, eh = reference_cs_counts("इस document और इस document")
        assert he[("इस", "document")] == 2
        assert sum(he.values()) == 2
        # "document और" is EN->HI
        assert eh[("document", "और")] == 1


class TestCBA:
    def test_perfect_hypothesis_scores_100(self):
        refs = ["इस document में formatting ठीक"]
        r = cba(refs, refs)
        assert r.he == 100.0
        assert r.eh == 100.0

    def test_missed_switch_point_scores_zero(self):
        refs = ["इस document में"]
        hyps = ["इस दस्तावेज़ में"]  # Hindi translation, switch point destroyed
        r = cba(refs, hyps)
        assert r.he_total == 1 and r.he_matched == 0
        assert r.eh_total == 1 and r.eh_matched == 0

    def test_multiset_partial_credit(self):
        # Reference contains the HE bigram twice; hypothesis contains it once.
        # Multiset semantics => 1/2 = 50%. A set-based impl would say 100%.
        refs = ["इस document और इस document"]
        hyps = ["इस document और इस दस्तावेज़"]
        r = cba(refs, hyps)
        assert r.he_total == 2
        assert r.he_matched == 1
        assert r.he == 50.0

    def test_hypothesis_bigram_not_required_to_be_a_switch_point(self):
        # The pair must merely appear adjacently in the hypothesis.
        refs = ["इस document"]
        hyps = ["कुछ इस document कुछ"]
        assert cba(refs, hyps).he == 100.0

    def test_normalization_applied_to_both_sides(self):
        refs = ["इस document।"]
        hyps = ["इस Document"]
        assert cba(refs, hyps).he == 100.0


class TestCBAGrouped:
    """Recording-level hypotheses with a per-utterance denominator."""

    REF = [
        ("spk_R_0000", "इस document में"),   # HE: (इस, document); EH: (document, में)
        ("spk_R_0001", "यह file है"),        # HE: (यह, file);     EH: (file, है)
    ]
    GROUP = lambda self, u: u.split("_")[-2]

    def test_denominator_comes_from_utterances_not_concatenation(self):
        # Concatenating gives "... में यह ..." (HI->HI, no switch), but if the
        # join happened to straddle scripts it would invent a bigram. The
        # denominator must equal the sum over utterances: 2 HE, 2 EH.
        r = cba_grouped(self.REF, {"R": "इस document में यह file है"}, self.GROUP)
        assert (r.he_total, r.eh_total) == (2, 2)
        assert r.he == 100.0 and r.eh == 100.0

    def test_concatenation_would_have_invented_a_bigram(self):
        # utt0 ends in English, utt1 starts in Hindi -> concatenating creates a
        # spurious EH pair ("document", "यह") that no utterance contains.
        refs = [("s_R_0000", "इस document"), ("s_R_0001", "यह file")]
        r = cba_grouped(refs, {"R": "इस document यह file"}, self.GROUP)
        assert r.he_total == 2          # (इस,document) and (यह,file)
        assert r.eh_total == 0          # NOT 1 -- the join pair is not counted

    def test_multiset_cap_is_per_recording_not_per_utterance(self):
        # The same HE bigram in two utterances, but only ONE occurrence in the
        # hypothesis. Both utterances must not each claim it: 1/2, not 2/2.
        refs = [("s_R_0000", "इस document"), ("s_R_0001", "इस document")]
        r = cba_grouped(refs, {"R": "इस document और कुछ"}, self.GROUP)
        assert r.he_total == 2
        assert r.he_matched == 1
        assert r.he == 50.0

    def test_lenient_mode_ignores_adjacency(self):
        refs = [("s_R_0000", "इस document")]
        hyp = {"R": "document कुछ और इस"}      # both words present, not adjacent
        assert cba_grouped(refs, hyp, self.GROUP, mode="adjacent").he == 0.0
        assert cba_grouped(refs, hyp, self.GROUP, mode="lenient").he == 100.0

    def test_lenient_is_still_capped_by_multiset(self):
        refs = [("s_R_0000", "इस document"), ("s_R_0001", "इस document")]
        hyp = {"R": "इस x document"}           # one 'इस', one 'document'
        r = cba_grouped(refs, hyp, self.GROUP, mode="lenient")
        assert r.he_total == 2 and r.he_matched == 1

    def test_missing_hypothesis_fails_loudly(self):
        with pytest.raises(SystemExit, match="no hypothesis"):
            cba_grouped(self.REF, {}, self.GROUP)

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match="unknown cba mode"):
            cba_grouped(self.REF, {"R": "x"}, self.GROUP, mode="fuzzy")


class TestCorpusStats:
    def test_token_and_type_counts_differ_when_repeated(self):
        stats = corpus_cs_bigram_stats(
            ["इस document", "इस document", "यह file"], preset="raw"
        )
        assert stats["he_tokens"] == 3
        assert stats["he_types"] == 2


class TestMER:
    def test_identical_is_zero(self):
        assert mer(["इस document में"], ["इस document में"]) == 0.0

    def test_one_substitution_of_three_words(self):
        assert mer(["इस document में"], ["इस file में"]) == pytest.approx(100 / 3)

    def test_hybrid_mode_scores_devanagari_at_char_level(self):
        # "है" vs "हो": word mode = 1 error / 1 word = 100%.
        # hybrid = 1 char sub out of 2 chars = 50%.
        assert mer(["है"], ["हो"], mode="word") == 100.0
        assert mer(["है"], ["हो"], mode="hybrid") == 50.0
