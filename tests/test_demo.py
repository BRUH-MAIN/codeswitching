"""The demo module: model registry, excerpt picking, script statistics, report."""

from __future__ import annotations

import numpy as np
import pytest

from csasr.demo import registry
from csasr.demo.engines import Result, pick_device
from csasr.demo.render import analyse, ansi, build_html, html_text
from csasr.demo.samples import Clip, pick_excerpts


# ---- registry ---------------------------------------------------------------------
def test_keys_cover_the_ablation_grid():
    keys = registry.all_models()
    for k in ("zeroshot-small", "zeroshot-mms", "medium-qlora", "large-sarvam30",
              "large-matched", "mms-lora"):
        assert k in keys
    assert "mms-qlora" not in keys            # no 4-bit CTC path was run


def test_fine_tuned_models_point_at_the_selected_checkpoint(tmp_path, monkeypatch):
    run = tmp_path / "g26__whisper-medium__qlora"
    run.mkdir()
    (run / "hyp_test.step").write_text("50")
    monkeypatch.setattr(registry, "EXP", tmp_path)
    monkeypatch.setattr(registry, "ROWS", tmp_path / "none.jsonl")
    m = registry.all_models()["medium-qlora"]
    assert m.step == 50 and m.path.endswith("checkpoint-50") and not m.available
    assert not m.zero_shot and m.kind == "whisper"
    assert registry.all_models()["mms-full"].kind == "ctc"
    assert registry.all_models()["zeroshot-mms"].target_lang == "hin"


def test_resolve_expands_presets_and_rejects_unknown():
    got = registry.resolve(["quick", "zeroshot-medium"], require_available=False)
    assert [m.key for m in got] == ["zeroshot-medium", "medium-qlora"]   # de-duplicated, ordered
    with pytest.raises(SystemExit, match="unknown model"):
        registry.resolve(["medium-qlorra"], require_available=False)


# ---- excerpts ---------------------------------------------------------------------
def _fake(n_reco=4, per=10, dur=6.0):
    ids, durs = [], []
    for r in range(n_reco):
        for i in range(per):
            ids.append(f"100{r}_reco{r}_{i:04d}")
            durs.append(dur)
    return ids, durs


def test_excerpts_are_consecutive_long_enough_and_from_distinct_recordings():
    ids, durs = _fake()
    ex = pick_excerpts(ids, durs, n=3, seconds=25, seed=1)
    assert len(ex) == 3
    recos = set()
    for run in ex:
        assert sum(durs[i] for i in run) >= 25
        idx = [int(ids[i].split("_")[-1]) for i in run]
        assert idx == list(range(idx[0], idx[0] + len(idx)))
        recos.add(ids[run[0]].split("_")[1])
    assert len(recos) == 3


def test_excerpts_are_reproducible_and_seed_dependent():
    ids, durs = _fake()
    a = pick_excerpts(ids, durs, 3, 25, seed=7)
    assert a == pick_excerpts(ids, durs, 3, 25, seed=7)
    assert a != pick_excerpts(ids, durs, 3, 25, seed=8)


def test_excerpts_can_be_pinned_to_recordings_and_reject_typos():
    ids, durs = _fake()
    ex = pick_excerpts(ids, durs, 2, 12, recordings=["reco2"])
    assert all(ids[i].split("_")[1] == "reco2" for run in ex for i in run)
    with pytest.raises(SystemExit):
        pick_excerpts(ids, durs, 1, 12, recordings=["nope"])


def test_short_recording_still_yields_an_excerpt():
    ids, durs = _fake(n_reco=1, per=2)
    (run,) = pick_excerpts(ids, durs, 1, 60)
    assert run == [0, 1]


# ---- script statistics and rendering ------------------------------------------------
def test_analyse_counts_english_share_and_switches():
    s = analyse("इस tutorial में हम सीखेंगे")
    assert s.words == 5 and s.switches == 2 and s.english_pct == pytest.approx(20.0)
    z = analyse("यह पूरा वाक्य हिंदी है")
    assert z.english_pct == 0 and z.switches == 0
    assert analyse("").words == 0


def test_digits_and_punctuation_do_not_create_switches():
    assert analyse("version 3.3 में").switches == 1


def test_markup_marks_english_without_losing_text():
    assert "tutorial" in ansi("इस tutorial में", True) and "\033[" in ansi("इस tutorial में", True)
    assert ansi("इस tutorial में", False) == "इस tutorial में"
    assert '<span class="en">tutorial</span>' in html_text("इस tutorial में")
    assert "&lt;b&gt;" in html_text("<b> इस")          # escaped


# ---- report ---------------------------------------------------------------------------
def _report(score):
    m = registry.resolve(["medium-qlora"], require_available=False)[0]
    clip = Clip("c1", np.zeros(16_000, dtype=np.float32) if score else None,
                "इस tutorial में हम सीखेंगे", "MUCS test, recording x")
    res = {m.key: {"c1": Result("इस tutorial में हम सीखेंगे", 1.5)}}
    return build_html("T", "sub", [clip], [m], res, score=score)


def test_report_with_audio_is_self_contained_and_scored():
    page = _report(True)
    assert "<audio" in page and "data:audio/flac;base64," in page
    assert "MER, these excerpts" in page and 'class="en"' in page and "http" not in page.replace(
        "http-equiv", "")


def test_report_without_scoring_has_no_excerpt_mer_column():
    page = _report(False)
    assert "MER, these excerpts" not in page and "<audio" not in page
    assert "Reference" in page


# ---- device policy --------------------------------------------------------------------
def test_auto_never_picks_a_gpu_outside_a_slurm_job(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert pick_device("auto") == "cpu"
    assert pick_device("cpu") == "cpu"


def test_report_parses_as_html():
    from html.parser import HTMLParser

    seen = []

    class P(HTMLParser):
        def handle_starttag(self, tag, attrs):
            seen.append(tag)

    P().feed(_report(True))
    assert {"table", "audio", "span", "style"} <= set(seen)
