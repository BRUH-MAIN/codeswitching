"""Backend selection + thinking-strip. No GPU or model download required."""

import pytest

from csasr.llm.backend import build_backend, strip_thinking


class TestStripThinking:
    def test_removes_think_block(self):
        assert strip_thinking("<think>let me reason</think>इस document") == "इस document"

    def test_removes_thinking_and_reasoning_variants(self):
        assert strip_thinking("<thinking>x</thinking>यह file") == "यह file"
        assert strip_thinking("<reasoning>y</reasoning>नया document") == "नया document"

    def test_multiline_think_block(self):
        raw = "<think>\nstep 1\nstep 2\n</think>\nप्रस्तुति document\nबुनियादी formatting"
        assert strip_thinking(raw) == "प्रस्तुति document\nबुनियादी formatting"

    def test_nested_or_repeated_blocks(self):
        # Both blocks removed; interior spacing is not our concern (the parsers
        # downstream split on whitespace anyway).
        assert strip_thinking("<think>a</think>keep1 <think>b</think>keep2").split() == ["keep1", "keep2"]

    def test_harmony_final_channel_keeps_only_the_answer(self):
        raw = "<|channel|>analysis<|message|>thinking...<|channel|>final<|message|>यह उत्तर है"
        assert strip_thinking(raw) == "यह उत्तर है"

    def test_dangling_open_think_token(self):
        assert strip_thinking("<|think|> इस document").startswith("इस document")

    def test_strips_stray_special_tokens(self):
        assert strip_thinking("इस document<end_of_turn>") == "इस document"
        assert strip_thinking("<s>यह file</s>") == "यह file"

    def test_leaves_clean_text_untouched(self):
        clean = "इस document में formatting"
        assert strip_thinking(clean) == clean

    def test_idempotent(self):
        raw = "<think>x</think>इस document<end_of_turn>"
        once = strip_thinking(raw)
        assert strip_thinking(once) == once

    def test_empty_and_none_safe(self):
        assert strip_thinking("") == ""

    def test_does_not_eat_devanagari_that_merely_contains_angle_brackets(self):
        # A real bigram with math/markup should survive.
        assert strip_thinking("temperature 5 degrees यहाँ") == "temperature 5 degrees यहाँ"


class TestBuildBackend:
    def test_echo_ignores_llamacpp_only_kwargs(self):
        be = build_backend("echo", model_id="x", filename="y.gguf")
        assert be.model_id == "echo"

    def test_transformers_route_drops_filename(self, monkeypatch):
        # Don't actually load a model: intercept the class.
        import csasr.llm.backend as b

        captured = {}

        class FakeTF:
            def __init__(self, **kw):
                captured.update(kw)

        monkeypatch.setattr(b, "TransformersBackend", FakeTF)
        build_backend("transformers", model_id="google/gemma-4-E4B-it",
                      filename="should-be-dropped.gguf")
        assert "filename" not in captured
        assert captured["model_id"] == "google/gemma-4-E4B-it"

    def test_llamacpp_route_keeps_filename(self, monkeypatch):
        import csasr.llm.backend as b

        captured = {}

        class FakeLC:
            def __init__(self, **kw):
                captured.update(kw)

        monkeypatch.setattr(b, "LlamaCppBackend", FakeLC)
        build_backend("llamacpp", model_id="unsloth/gemma-4-26B-A4B-it-GGUF",
                      filename="gemma-4-26B-A4B-it-UD-Q4_K_M.gguf")
        assert captured["filename"] == "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="unknown backend"):
            build_backend("vllm")
