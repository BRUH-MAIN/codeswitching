import pytest

from csasr.llm.backend import EchoBackend, Sampling
from csasr.llm.cache import ResponseCache
from csasr.llm.filter_bigrams import parse_verdicts, script_filter
from csasr.llm.gen_bigrams import parse_bigrams
from csasr.llm.gen_sentences import matrix_lang, parse_sentences, validate
from csasr.llm.prompts import bigram_messages, sentence_messages, translation_check_messages


class TestParseBigrams:
    def test_strips_numbering_and_bullets(self):
        out = parse_bigrams("1. प्रतीत document\n- बुनियादी formatting\n* इस spoken")
        assert out == ["प्रतीत document", "बुनियादी formatting", "इस spoken"]

    def test_splits_semicolon_separated_line(self):
        assert parse_bigrams("प्रतीत document; बुनियादी formatting") == [
            "प्रतीत document",
            "बुनियादी formatting",
        ]

    def test_drops_single_token_lines(self):
        assert parse_bigrams("document\nप्रतीत document") == ["प्रतीत document"]

    def test_lenient_keeps_malformed_pairs_for_the_yield_statistic(self):
        # Two English words is invalid, but raw yield must still count it so our
        # dedup rate is comparable with the paper's 13.3%.
        assert "open document" in parse_bigrams("open document")


class TestScriptFilter:
    def test_accepts_hindi_then_english(self):
        assert script_filter("प्रतीत document") == ("प्रतीत document", "प्रतीत", "document")

    def test_english_then_hindi_keeps_word_order_but_labels_correctly(self):
        # bigram text preserves the EH order; hi/en are labelled by script.
        assert script_filter("document प्रतीत") == ("document प्रतीत", "प्रतीत", "document")

    def test_rejects_monolingual(self):
        assert script_filter("open document") is None
        assert script_filter("यह दस्तावेज़") is None

    def test_rejects_single_token(self):
        assert script_filter("प्रतीत") is None

    def test_extracts_the_switch_pair_from_a_longer_phrase(self):
        # Gemma 4 emits trigrams; the bigram is inside. Deviation D9.
        assert script_filter("बुनियादी formatting basics")[0] == "बुनियादी formatting"
        assert script_filter("नया document create")[0] == "नया document"
        assert script_filter("click पर action")[0] == "click पर"

    def test_extraction_drops_a_monolingual_phrase(self):
        assert script_filter("menu bar click") is None
        assert script_filter("file save document") is None

    def test_strict_mode_is_paper_faithful(self):
        # The paper's 70B obeyed "a couple of words"; strict reproduces that.
        assert script_filter("बुनियादी formatting basics", strict=True) is None
        assert script_filter("बुनियादी formatting", strict=True) is not None

    def test_real_gemma4_sample_yields_8_of_10(self):
        """Verbatim output from the Kaggle smoke test. Strict kept 0/10."""
        sample = [
            "बुनियादी formatting basics", "नया document create", "menu bar click",
            "file save document", "प्रस्तुति presentation skill", "यह tutorial guide",
            "आपका स्वागत welcome", "अब हम new", "click पर action", "करना do task",
        ]
        assert sum(script_filter(p) is not None for p in sample) == 8
        assert sum(script_filter(p, strict=True) is not None for p in sample) == 0


class TestParseVerdicts:
    def test_parses_yes_no(self):
        out = parse_verdicts("1: YES\n2: NO\n3: yes", n=3)
        assert out == {0: True, 1: False, 2: True}

    def test_ignores_out_of_range_and_garbage(self):
        assert parse_verdicts("9: YES\nblah\n1: NO", n=2) == {0: False}


class TestSentences:
    def test_parse_strips_markers(self):
        out = parse_sentences("1. इस document को open करो\n2. Please इस file को देखें")
        assert out == ["इस document को open करो", "Please इस file को देखें"]

    def test_validate_requires_adjacent_bigram(self):
        assert validate("मैंने इस document को खोला", "इस", "document")
        # bigram present but not adjacent
        assert not validate("इस बड़े document को खोला", "इस", "document")

    def test_validate_requires_both_scripts(self):
        assert not validate("this document is open", "इस", "document")

    def test_validate_accepts_either_order_by_default(self):
        # "document इस" is an EN->HI switch point: still a useful CS example.
        assert validate("The document इस file में है", "इस", "document")

    def test_require_order_rejects_the_reversed_placement(self):
        assert not validate(
            "The document इस file में है", "इस", "document", require_order=True
        )
        assert validate("मैंने इस document को खोला", "इस", "document", require_order=True)

    def test_matrix_lang_by_majority(self):
        assert matrix_lang("इस document को खोलो और देखो") == "hi"
        assert matrix_lang("Please open इस file now") == "en"


class TestPrompts:
    def test_bigram_prompt_is_verbatim_and_seeds_examples(self):
        msgs = bigram_messages(["इस spoken tutorial में", "यह basic formatting है"], n=10)
        assert msgs[0]["role"] == "system"
        assert "mix-bigrams generator" in msgs[0]["content"]
        user = msgs[1]["content"]
        assert "Generate 10 language mix bigrams in Hindi-English" in user
        assert "Never put just a translation" in user
        assert "प्रस्तुति document" in user  # examples recovered from the corpus
        assert "इस spoken tutorial में" in user

    def test_sentence_prompt_asks_for_two_and_two(self):
        user = sentence_messages("प्रतीत document")[1]["content"]
        assert "Generate four sentences" in user
        assert "2 sentences with English as the main language" in user
        assert user.endswith("प्रतीत document")

    def test_translation_check_numbers_items(self):
        user = translation_check_messages([("दस्तावेज़", "document"), ("इस", "spoken")])[1]["content"]
        assert "1. दस्तावेज़ / document" in user
        assert "2. इस / spoken" in user


class TestCacheAndBackend:
    def test_cache_round_trips_and_persists(self, tmp_path):
        p = tmp_path / "c.jsonl"
        with ResponseCache(p) as c:
            k = c.key("m", [{"role": "user", "content": "hi"}], {"t": 1}, 0)
            assert c.get(k) is None
            c.put(k, "yo")
            assert c.get(k) == "yo"
        with ResponseCache(p) as c2:
            assert c2.get(k) == "yo"
            assert len(c2) == 1

    def test_sample_index_changes_the_key(self):
        a = ResponseCache.key("m", [{"role": "u", "content": "x"}], {}, 0)
        b = ResponseCache.key("m", [{"role": "u", "content": "x"}], {}, 1)
        assert a != b

    def test_chat_preserves_order_and_uses_cache(self, tmp_path):
        be = EchoBackend(responses={"a": "A", "b": "B"})
        convs = [[{"role": "user", "content": x}] for x in ("a", "b", "a")]
        with ResponseCache(tmp_path / "c.jsonl") as cache:
            out = be.chat(convs, Sampling(), cache=cache, batch_size=2)
            assert out == ["A", "B", "A"]
            n_first = len(be.calls)
            # Second run must be a pure cache hit: the backend is never called.
            out2 = be.chat(convs, Sampling(), cache=cache, batch_size=2)
            assert out2 == ["A", "B", "A"]
            assert len(be.calls) == n_first

    def test_chat_dedups_identical_prompts_via_cache_within_a_run(self, tmp_path):
        be = EchoBackend(responses={"a": "A"})
        convs = [[{"role": "user", "content": "a"}]] * 3
        with ResponseCache(tmp_path / "c.jsonl") as cache:
            assert be.chat(convs, Sampling(), cache=cache, batch_size=1) == ["A", "A", "A"]
