"""CBA must use multiset semantics.

The paper counts 4,189 HE bigram *tokens* against 2,347 unique *types* in the
test set, so a bigram repeated in the reference must be able to contribute more
than once to both numerator and denominator. A set-based implementation would
silently under-count the denominator and inflate CBA.
"""

import pytest

from csasr.eval.cba import cba, corpus_cs_bigram_stats, reference_cs_counts
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
