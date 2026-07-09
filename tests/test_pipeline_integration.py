"""End-to-end wiring test: gen_bigrams -> filter_bigrams -> gen_sentences.

Uses a scripted stub backend, so this runs on CPU in milliseconds and still
exercises the real CLIs, the real cache, and the real parsers. It catches
argparse/plumbing regressions that unit tests miss.
"""

from __future__ import annotations

import json
import re

import pytest

from csasr.llm import backend as backend_mod
from csasr.llm.backend import LLMBackend
from csasr.manifest import read_jsonl, write_jsonl


class ScriptedBackend(LLMBackend):
    """Answers by inspecting the user prompt, so one stub serves all three stages."""

    model_id = "scripted"

    def _generate(self, convs, sampling):
        out = []
        for conv in convs:
            user = conv[-1]["content"]
            if "language mix bigrams" in user:
                out.append(
                    "1. प्रस्तुति document\n"
                    "2. बुनियादी formatting\n"
                    "3. इस spoken\n"
                    "4. open document\n"          # monolingual -> killed by script filter
                    "5. दस्तावेज़ document\n"      # direct translation -> killed by LLM check
                )
            elif "direct translations" in user:
                # Echo back the item numbers the prompt actually used. Numbering
                # here is 1-based and must not include the instruction line.
                lines = []
                for line in user.splitlines():
                    m = re.match(r"^\s*(\d+)\.\s*(\S+)\s*/\s*(\S+)\s*$", line)
                    if not m:
                        continue
                    num, hi, en = m.groups()
                    is_trans = hi == "दस्तावेज़" and en == "document"
                    lines.append(f"{num}: {'YES' if is_trans else 'NO'}")
                out.append("\n".join(lines))
            elif "Generate four sentences" in user:
                bigram = user.rsplit(":", 1)[1].strip()
                hi, en = bigram.split()
                out.append(
                    f"1. Please open the {en} {hi} तरह से\n"
                    f"2. मैंने {hi} {en} को खोला\n"
                    f"3. यह {hi} {en} बहुत अच्छा है\n"
                    f"4. this line has no bigram at all\n"   # dropped by validate()
                )
            else:
                out.append("")
        return out


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setattr(backend_mod, "build_backend", lambda name, **kw: ScriptedBackend())
    for mod in ("gen_bigrams", "filter_bigrams", "gen_sentences"):
        m = __import__(f"csasr.llm.{mod}", fromlist=["build_backend"])
        monkeypatch.setattr(m, "build_backend", lambda name, **kw: ScriptedBackend())


def test_full_text_pipeline(tmp_path, stub):
    from csasr.llm import filter_bigrams, gen_bigrams, gen_sentences

    train = tmp_path / "train.jsonl"
    write_jsonl(train, [{"utt_id": f"u{i}", "text": "इस spoken tutorial में document है"} for i in range(10)])

    raw = tmp_path / "raw.jsonl"
    assert gen_bigrams.main([
        "--train-manifest", str(train), "--out", str(raw),
        "--cache", str(tmp_path / "c1.jsonl"), "--n-calls", "3", "--batch-size", "2",
    ]) == 0
    raw_rows = list(read_jsonl(raw))
    assert len(raw_rows) == 15                       # 3 calls x 5 parsed lines
    assert len({r["bigram"] for r in raw_rows}) == 5  # dedup

    valid = tmp_path / "valid.jsonl"
    assert filter_bigrams.main([
        "--raw", str(raw), "--out", str(valid),
        "--cache", str(tmp_path / "c2.jsonl"), "--items-per-call", "10", "--n-samples", "1",
    ]) == 0
    kept = {r["bigram"] for r in read_jsonl(valid)}
    assert kept == {"प्रस्तुति document", "बुनियादी formatting", "इस spoken"}
    assert "open document" not in kept       # script filter: monolingual
    assert "दस्तावेज़ document" not in kept   # translation check

    sents = tmp_path / "sents.jsonl"
    assert gen_sentences.main([
        "--bigrams", str(valid), "--out", str(sents),
        "--cache", str(tmp_path / "c3.jsonl"), "--batch-size", "2",
    ]) == 0
    rows = list(read_jsonl(sents))
    assert len(rows) == 9                     # 3 bigrams x 3 valid sentences
    assert all("sent_id" in r and "matrix_lang" in r for r in rows)
    assert len({r["sent_id"] for r in rows}) == 9
    assert {r["matrix_lang"] for r in rows} <= {"hi", "en"}


def test_pipeline_is_resumable(tmp_path, stub):
    """A second run must hit the cache and produce byte-identical output."""
    from csasr.llm import gen_bigrams

    train = tmp_path / "train.jsonl"
    # gen_bigrams samples 5 distinct sentences per prompt, so it needs >= 5.
    write_jsonl(train, [{"utt_id": f"u{i}", "text": f"इस spoken tutorial {i} में"} for i in range(6)])
    cache = tmp_path / "c.jsonl"

    out1, out2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    gen_bigrams.main(["--train-manifest", str(train), "--out", str(out1), "--cache", str(cache), "--n-calls", "2"])
    n_cached = sum(1 for _ in read_jsonl(cache))
    gen_bigrams.main(["--train-manifest", str(train), "--out", str(out2), "--cache", str(cache), "--n-calls", "2"])

    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")
    assert sum(1 for _ in read_jsonl(cache)) == n_cached  # no new LLM calls


def test_subset_and_manifest_round_trip(tmp_path):
    """Train_T1 ids must all resolve inside Train_T2."""
    from csasr.tts.make_subset import main as subset_main

    t2 = tmp_path / "t2.jsonl"
    write_jsonl(t2, [
        {"utt_id": f"s{i}", "text": "x", "dur": 5.0, "bigram": f"bg{i % 7}"}
        for i in range(200)
    ])
    ids_path = tmp_path / "t1_ids.json"
    assert subset_main(["--t2", str(t2), "--out", str(ids_path), "--hours", "0.1"]) == 0

    ids = json.loads(ids_path.read_text())
    all_ids = {r["utt_id"] for r in read_jsonl(t2)}
    assert set(ids) < all_ids
    assert len(ids) == len(set(ids))
