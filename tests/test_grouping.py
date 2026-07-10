import pytest

from csasr.eval.grouping import (
    concat_refs,
    group_by_recording,
    recording_id,
    segment_index,
)

ROWS = [
    {"utt_id": "103085_AAA_0002", "text": "third"},
    {"utt_id": "103085_AAA_0000", "text": "first"},
    {"utt_id": "999_BBB_0001", "text": "beta two"},
    {"utt_id": "103085_AAA_0001", "text": "second"},
    {"utt_id": "999_BBB_0000", "text": "beta one"},
]


class TestIds:
    def test_recording_id_is_the_middle_field(self):
        assert recording_id("103085_w5Jyq3XMbb3WwiKQ_0000") == "w5Jyq3XMbb3WwiKQ"

    def test_segment_index_is_the_trailing_number(self):
        assert segment_index("103085_AAA_0042") == 42

    def test_speaker_prefix_is_ignored_for_grouping(self):
        # Two speakers must not split one recording.
        assert recording_id("1_XYZ_0000") == recording_id("2_XYZ_0001") == "XYZ"


class TestMalformed:
    def test_too_few_fields_raises(self):
        with pytest.raises(ValueError, match="unexpected utt_id shape"):
            recording_id("noseparator")

    def test_non_numeric_index_raises(self):
        with pytest.raises(ValueError, match="numeric index"):
            segment_index("a_b_xyz")


class TestGrouping:
    def test_groups_and_sorts_chronologically(self):
        g = group_by_recording(ROWS)
        assert set(g) == {"AAA", "BBB"}
        assert [r["text"] for r in g["AAA"]] == ["first", "second", "third"]
        assert [r["text"] for r in g["BBB"]] == ["beta one", "beta two"]

    def test_input_order_does_not_matter(self):
        assert group_by_recording(ROWS) == group_by_recording(list(reversed(ROWS)))

    def test_concat_refs_joins_in_order(self):
        refs = concat_refs(ROWS)
        assert refs["AAA"] == "first second third"
        assert refs["BBB"] == "beta one beta two"

    def test_concat_skips_empty_segments(self):
        rows = [
            {"utt_id": "s_R_0000", "text": "a"},
            {"utt_id": "s_R_0001", "text": "   "},
            {"utt_id": "s_R_0002", "text": "b"},
        ]
        assert concat_refs(rows)["R"] == "a b"

    def test_concat_preserves_code_switch_bigrams_across_a_join(self):
        # A switch that straddles a segment boundary must survive concatenation.
        from csasr.lid import cs_bigrams

        rows = [
            {"utt_id": "s_R_0000", "text": "यह इस"},
            {"utt_id": "s_R_0001", "text": "document है"},
        ]
        kinds = [b.kind for b in cs_bigrams(concat_refs(rows)["R"])]
        assert kinds == ["HE", "EH"]
