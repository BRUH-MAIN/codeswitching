import pytest

from csasr.lid import (
    Lang,
    all_bigrams,
    count_words,
    cs_bigrams,
    split_script_boundaries,
    tokenize,
    word_lang,
)
from csasr.normalize import normalize


class TestSplitScriptBoundaries:
    """MUCS contains dropped-space typos that hide real switch points."""

    def test_splits_hindi_glued_to_english(self):
        assert split_script_boundaries("दायाँclick") == ["दायाँ", "click"]
        assert split_script_boundaries("करेंspoken") == ["करें", "spoken"]
        assert split_script_boundaries("detailsको") == ["details", "को"]

    def test_digits_attach_and_do_not_split(self):
        assert split_script_boundaries("mp3") == ["mp3"]
        assert split_script_boundaries("334") == ["334"]

    def test_pure_tokens_are_untouched(self):
        assert split_script_boundaries("document") == ["document"]
        assert split_script_boundaries("दस्तावेज़") == ["दस्तावेज़"]

    def test_tokenize_recovers_the_hidden_switch_point(self):
        assert tokenize("दायाँclick करें") == ["दायाँ", "click", "करें"]
        kinds = [b.kind for b in cs_bigrams("दायाँclick करें")]
        assert kinds == ["HE", "EH"]

    def test_split_mixed_can_be_disabled(self):
        assert tokenize("दायाँclick", split_mixed=False) == ["दायाँclick"]

    def test_no_mixed_tokens_survive_tokenization(self):
        assert all(word_lang(t) is not Lang.MIXED for t in tokenize("दायाँclick detailsको"))


class TestWordLang:
    @pytest.mark.parametrize(
        "word",
        ["नमस्ते", "दस्तावेज़", "बुनियादी", "क्लिक", "है"],
    )
    def test_devanagari_is_hindi(self, word):
        assert word_lang(word) is Lang.HI

    @pytest.mark.parametrize("word", ["document", "formatting", "spoken", "Whisper"])
    def test_latin_is_english(self, word):
        assert word_lang(word) is Lang.EN

    @pytest.mark.parametrize("word", ["123", "42", "...", "।", "—"])
    def test_digits_and_punct_are_other(self, word):
        assert word_lang(word) is Lang.OTHER

    def test_both_scripts_in_one_token_is_mixed(self):
        assert word_lang("फाइलformat") is Lang.MIXED

    def test_bare_danda_is_not_a_hindi_word(self):
        # U+0964 sits inside the Devanagari block but is category Po. A block
        # membership test alone would call it Hindi and inflate Words_H.
        assert word_lang("।") is Lang.OTHER
        assert word_lang("॥") is Lang.OTHER

    def test_devanagari_digits_are_other(self):
        # U+0966-U+096F are Nd, inside the Devanagari block. Consistent with
        # Latin digits being OTHER.
        assert word_lang("५") is Lang.OTHER
        assert word_lang("१२३") is Lang.OTHER

    def test_word_with_trailing_danda_is_still_hindi(self):
        assert word_lang("है।") is Lang.HI

    def test_combining_marks_do_not_split_a_word(self):
        # दस्तावेज़ is 9 codepoints, 5 of which are letters and 4 combining marks.
        # It must remain a single Hindi token.
        assert word_lang("दस्तावेज़") is Lang.HI
        assert len(normalize("दस्तावेज़", "scoring").split()) == 1


class TestNormalize:
    def test_scoring_preserves_devanagari_marks(self):
        assert normalize("दस्तावेज़", "scoring") == "दस्तावेज़"

    def test_scoring_strips_danda(self):
        assert normalize("यह ठीक है।", "scoring") == "यह ठीक है"

    def test_scoring_lowercases_latin(self):
        assert normalize("Click The Button", "scoring") == "click the button"

    def test_whisper_basic_is_destructive_on_devanagari(self):
        # Documents exactly why we do not use it. One word becomes four.
        assert len(normalize("दस्तावेज़", "whisper_basic").split()) == 4


class TestCSBigrams:
    def test_he_and_eh_directions(self):
        # "इस document में formatting ठीक"
        #    HI   ->  EN   (HE)
        #        EN -> HI  (EH)
        #           HI -> EN (HE)
        #                  EN -> HI (EH)
        text = "इस document में formatting ठीक"
        got = [(b.first, b.second, b.kind) for b in cs_bigrams(text)]
        assert got == [
            ("इस", "document", "HE"),
            ("document", "में", "EH"),
            ("में", "formatting", "HE"),
            ("formatting", "ठीक", "EH"),
        ]

    def test_monolingual_has_no_switch_points(self):
        assert list(cs_bigrams("यह एक वाक्य है")) == []
        assert list(cs_bigrams("this is a sentence")) == []

    def test_digits_do_not_open_a_switch_point(self):
        # हर्ट्ज़ -> 5 is HI->OTHER, 5 -> hertz is OTHER->EN. Neither counts.
        assert list(cs_bigrams("हर्ट्ज़ 5 hertz")) == []

    def test_all_bigrams_includes_non_switch_pairs(self):
        assert all_bigrams("a b c") == [("a", "b"), ("b", "c")]


class TestCountWords:
    def test_counts_by_language(self):
        c = count_words("इस document में")
        assert c[Lang.HI] == 2
        assert c[Lang.EN] == 1
        assert c[Lang.OTHER] == 0
