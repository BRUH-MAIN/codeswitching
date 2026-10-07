"""The ablation path: fine-tuning methods, post-hoc selection, the LLM server
backend, the matched prompt, and CTC decoding.

These are the pieces whose failure would not crash -- it would produce a
plausible, wrong number in a results table (a merge that changes the model, a
window stitch that drops audio, a selection that reuses a stale decode).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from csasr.eval.decode_ctc import windows
from csasr.eval.select import latin_pct, pick_best, write_row
from csasr.llm.backend import OpenAIBackend, Sampling, build_backend
from csasr.llm.prompts import SENTENCE_USER_TEMPLATE, sentence_messages
from csasr.manifest import write_jsonl
from csasr.train.train_ctc import DELIM, PAD, UNK, build_vocab, ctc_text
from csasr.train.train_whisper import _build_parser, _plan_hardware

from test_train_hardware import LARGE_V2, FakeCuda, FakeTorch


# ---- fine-tuning method -----------------------------------------------------
def _args(**kw):
    base = dict(precision="auto", tf32="auto", gradient_checkpointing="auto", ft_method="full")
    return SimpleNamespace(**{**base, **kw})


def test_parser_defaults_keep_the_paper_recipe():
    a = _build_parser().parse_args(["--out", "x"])
    assert a.ft_method == "full" and a.generate_eval == "on" and a.save_steps is None
    assert a.lora_targets == ["q_proj", "v_proj"]


def test_peft_never_auto_enables_gradient_checkpointing():
    # Full FT of large-v2 on a 40 GB card needs it; the same model under LoRA
    # carries no grads or Adam state for the base, so it must stay off.
    torch = FakeTorch(FakeCuda("A100", 8, 0, 40))
    assert _plan_hardware(_args(), torch, LARGE_V2)["gradient_checkpointing"] is True
    for ft in ("lora", "qlora"):
        hw = _plan_hardware(_args(ft_method=ft), torch, LARGE_V2)
        assert hw["gradient_checkpointing"] is False
    on = _plan_hardware(_args(ft_method="lora", gradient_checkpointing="on"), torch, LARGE_V2)
    assert on["gradient_checkpointing"] is True


def test_merged_adapter_computes_the_same_function(tmp_path):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    from csasr.train.export import adapter_base, is_adapter_dir, merge_adapter

    cfg = WhisperConfig(
        vocab_size=64, d_model=16, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32,
        decoder_ffn_dim=32, num_mel_bins=8, max_source_positions=20, max_target_positions=16,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1,
        suppress_tokens=None, begin_suppress_tokens=None,
    )
    torch.manual_seed(0)
    base_dir = tmp_path / "base"
    WhisperForConditionalGeneration(cfg).save_pretrained(base_dir)

    model = WhisperForConditionalGeneration.from_pretrained(base_dir)
    pm = peft.get_peft_model(model, peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
    for n, p in pm.named_parameters():
        if "lora_B" in n:  # B initializes to zero, which would make the test vacuous
            torch.nn.init.normal_(p, std=0.5)
    pm.save_pretrained(tmp_path / "adapter")
    assert is_adapter_dir(tmp_path / "adapter") and not is_adapter_dir(base_dir)
    assert adapter_base(tmp_path / "adapter") == str(base_dir)

    out = merge_adapter(tmp_path / "adapter", tmp_path / "merged", save_processor=False)
    merged = WhisperForConditionalGeneration.from_pretrained(out)
    feats = torch.randn(2, 8, 40)
    dec = torch.tensor([[1, 5, 7], [1, 9, 3]])
    with torch.inference_mode():
        want = pm.eval()(input_features=feats, decoder_input_ids=dec).logits
        got = merged.eval()(input_features=feats, decoder_input_ids=dec).logits
        plain = WhisperForConditionalGeneration.from_pretrained(base_dir).eval()(
            input_features=feats, decoder_input_ids=dec).logits
    assert torch.allclose(want, got, atol=1e-4)
    assert not torch.allclose(plain, got, atol=1e-3), "the adapter changed nothing"


# ---- post-hoc selection -------------------------------------------------------
def test_pick_best_prefers_lower_mer_then_earlier_step():
    curve = [{"step": 200, "mer": 50.0}, {"step": 400, "mer": 48.0}, {"step": 600, "mer": 48.0}]
    assert pick_best(curve)["step"] == 400


def test_latin_pct_counts_script_not_meaning(tmp_path):
    hyp = tmp_path / "h.jsonl"
    write_jsonl(hyp, [{"utt_id": "r", "hyp": "यह menu bar है ऑफिस"}])
    assert latin_pct(hyp) == 40.0  # 2 Latin of 5; ऑफिस is English written in Devanagari


def test_write_row_overwrites_its_own_run_only(tmp_path):
    hyp = tmp_path / "h.jsonl"
    write_jsonl(hyp, [{"utt_id": "r", "hyp": "x"}])
    write_row(tmp_path / "res", {"run": "a", "mer": 50.0}, hyp)
    write_row(tmp_path / "res", {"run": "b", "mer": 40.0}, hyp)
    write_row(tmp_path / "res", {"run": "a", "mer": 45.0}, hyp)
    rows = {p.stem: json.loads(p.read_text()) for p in (tmp_path / "res" / "rows").glob("*.json")}
    assert rows == {"a": {"run": "a", "mer": 45.0}, "b": {"run": "b", "mer": 40.0}}
    assert (tmp_path / "res" / "hyps" / "a.jsonl").exists()


# ---- CTC ----------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 400, 30 * 16000, 30 * 16000 + 1, 47 * 16000, 600 * 16000 + 123])
def test_windows_tile_the_recording_exactly(n):
    chunk, stride = 30 * 16000, 5 * 16000
    kept = []
    for start, end, dl, dr in windows(n, chunk, stride):
        assert 0 <= start < end <= n and end - start <= chunk
        kept.append((start + dl, end - dr))
    assert kept[0][0] == 0 and kept[-1][1] == n
    for (_, a_end), (b_start, _) in zip(kept, kept[1:]):
        assert a_end == b_start, "kept regions must neither overlap nor leave a gap"


def test_ctc_vocab_covers_both_scripts_and_reserves_blank():
    v = build_vocab(["यह Menu bar है।", "File को save करें"])
    assert v[PAD] == 0 and UNK in v and DELIM in v
    assert "m" in v and "M" not in v, "targets are lowercased, like the scorer"
    assert "।" not in v and " " not in v
    assert "ह" in v and "ै" in v, "combining marks are their own symbols"
    assert ctc_text("File, को save करें।") == "file को save करें"


# ---- LLM server backend ---------------------------------------------------------
class _Stub(BaseHTTPRequestHandler):
    seen: list[dict] = []
    reject_template_kwargs = False

    def do_POST(self):  # noqa: N802 - http.server's name
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append(body)
        if type(self).reject_template_kwargs and "chat_template_kwargs" in body:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "unknown field chat_template_kwargs"}')
            return
        text = "<think>scratch</think>" + body["messages"][-1]["content"].upper()
        payload = json.dumps({"choices": [{"message": {"content": text}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


@pytest.fixture
def stub_server():
    _Stub.seen, _Stub.reject_template_kwargs = [], False
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_openai_backend_preserves_order_and_strips_reasoning(stub_server):
    be = build_backend("openai", model_id="m", filename="ignored.gguf",
                       base_url=stub_server, concurrency=4)
    convs = [[{"role": "user", "content": f"q{i}"}] for i in range(12)]
    out = be.chat(convs, Sampling(seed=10), sample_idx=2, batch_size=12)
    assert out == [f"Q{i}" for i in range(12)]
    assert all(b["seed"] == 12 for b in _Stub.seen), "sample index must vary the seed"
    assert all(b["chat_template_kwargs"] == {"enable_thinking": False} for b in _Stub.seen)


def test_openai_backend_survives_an_endpoint_without_template_kwargs(stub_server):
    _Stub.reject_template_kwargs = True
    be = OpenAIBackend("m", base_url=stub_server, concurrency=1)
    assert be.chat([[{"role": "user", "content": "hi"}]], Sampling()) == ["HI"]
    assert be._think_kwarg_ok is False


# ---- sentence sanity filter -------------------------------------------------------
@pytest.mark.parametrize("line", [
    # real Sarvam-30B output that passed bigram validation
    'I need to generate four sentences using the bigram "में default" in a natural way.',
    'Wait, let me reconsider. The bigram is "अपने typing" which is a combination',
    "English main language: The software can automatically capture करने data.",
    "जब **लोलक oscillates** करता है, तो मुझे बहुत अच्छा लगता है।",
    "I don't understand the implementation का[implementation ka] of this new policy.",
    "यहाँ color use karke drawing ko aur accha bana sakte hain.",
    "Action नहीं ḥe करने के लिए हमें और मेहनत करनी होगी।",
    "क्या आप मुझे किसी अच्छे चिकित्सक (dentist) का नाम बता सकते हैं?",
])
def test_commentary_markup_and_romanized_hindi_are_not_speech(line):
    from csasr.llm.gen_sentences import looks_spoken

    assert not looks_spoken(line)


@pytest.mark.parametrize("line", [
    "यह menu bar है",
    "अब हम एक नया document बनाएंगे।",
    "Please save the file और फिर close करें.",
    "We'll open the editor, फिर code लिखेंगे!",
    "slide में अब सफेद background है?",
    "The theme of this tutorial is spoken clearly.",   # "the", "is": not romanized Hindi
])
def test_ordinary_code_mixed_sentences_are_speech(line):
    from csasr.llm.gen_sentences import looks_spoken

    assert looks_spoken(line)


# ---- prompts --------------------------------------------------------------------
def test_paper_prompt_is_untouched_and_matched_is_hindi_matrix_only():
    paper = sentence_messages("नया document")
    assert paper[1]["content"] == SENTENCE_USER_TEMPLATE.format(bigram="नया document")
    assert "2 sentences with English as the main language" in paper[1]["content"]

    ex = ["यह menu bar है", "file को save करें।"]
    matched = sentence_messages("नया document", variant="matched", example_sentences=ex)
    text = matched[1]["content"]
    assert "English as the main language" not in text
    assert "यह menu bar है." in text and "करें।." not in text
    assert text.endswith("नया document")
    assert matched[0] == paper[0], "the system prompt stays the paper's"
    with pytest.raises(ValueError):
        sentence_messages("x y", variant="matched")
