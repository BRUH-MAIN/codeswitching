"""Intrinsic comparison of the synthetic TEXT corpora, before any audio exists.

The LLM and prompt axes of the ablation change only the text. This table says
how each corpus differs from the target domain on the properties that the
fine-tuned models were found to inherit -- above all the share of Latin-script
words in Hindi-matrix sentences, which a Whisper trained on the corpus copies
into its output almost exactly.

    python scripts/text_stats.py --data data/ablation \
        --mucs manifests/mucs_train.jsonl --test-refs results/refs_test.jsonl

Reads `<data>/<corpus>/sentences.jsonl` for every corpus directory present; the
existing Gemma-4-26B-A4B corpus is taken from the Hub (`--g26-hf`) or from a
local sentences file (`--g26`). CPU only, seconds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from csasr.lid import Lang, count_words, cs_bigrams, tag_words
from csasr.manifest import read_jsonl
from csasr.normalize import normalize

LABELS = {
    "g26": "Gemma-4-26B-A4B", "g31": "Gemma-4-31B", "s30": "Sarvam-30B",
    "s105": "Sarvam-105B",
}


def label(corpus: str) -> str:
    if corpus.endswith("m") and corpus[:-1] in LABELS:
        return LABELS[corpus[:-1]] + " (matched prompt)"
    return LABELS.get(corpus, corpus)


def _latin(texts) -> float:
    hi = en = 0
    for t in texts:
        c = count_words(t)
        hi += c[Lang.HI]
        en += c[Lang.EN]
    return 100.0 * en / max(hi + en, 1)


def _is_hi_matrix(text: str) -> bool:
    c = count_words(text)
    return c[Lang.HI] >= c[Lang.EN]


def _en_runs(text: str) -> list[int]:
    """Lengths of maximal runs of consecutive English words."""
    runs, cur = [], 0
    for _, lang in tag_words(text):
        if lang is Lang.EN:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def _switch_types(texts) -> set[tuple[str, str]]:
    return {(b.first.lower(), b.second.lower()) for t in texts for b in cs_bigrams(t)}


def profile(texts: list[str], test_types: Counter, mucs_en: set[str]) -> dict:
    norm = [normalize(t, "scoring") for t in texts]
    hi_m = [t for t in norm if _is_hi_matrix(t)]
    en_m = [t for t in norm if not _is_hi_matrix(t)]
    runs = [r for t in norm for r in _en_runs(t)]
    types = _switch_types(norm)
    en_tok = Counter(w for t in norm for w, lang in tag_words(t) if lang is Lang.EN)
    covered = sum(n for bg, n in test_types.items() if bg in types)
    words = sum(len(t.split()) for t in norm)
    return {
        "sentences": len(texts),
        "est_hours": round(words / 2.4 * 0.954 / 3600, 1),
        "latin_pct": round(_latin(norm), 1),
        "hi_matrix_pct": round(100.0 * len(hi_m) / max(len(norm), 1), 1),
        "latin_pct_hi_matrix": round(_latin(hi_m), 1),
        "latin_pct_en_matrix": round(_latin(en_m), 1) if en_m else None,
        "switches_per_sentence": round(
            sum(len(list(cs_bigrams(t))) for t in norm) / max(len(norm), 1), 2),
        "mean_en_run": round(sum(runs) / max(len(runs), 1), 2),
        "en_runs_ge4_pct": round(100.0 * sum(r >= 4 for r in runs) / max(len(runs), 1), 1),
        "en_tokens_in_mucs_pct": round(
            100.0 * sum(n for w, n in en_tok.items() if w in mucs_en)
            / max(sum(en_tok.values()), 1), 1),
        # What CBA is measured on: how many of the test set's switch-bigram
        # TOKENS have their (word, word) type somewhere in the training text.
        "test_switch_coverage_pct": round(
            100.0 * covered / max(sum(test_types.values()), 1), 1),
        "switch_types": len(types),
    }


def yields(d: Path) -> dict:
    out = {}
    if (d / "bigrams_raw.jsonl").exists():
        raw = [r["bigram"] for r in read_jsonl(d / "bigrams_raw.jsonl")]
        out["raw_bigrams"] = len(raw)
        out["unique_bigrams"] = len(set(raw))
        out["dedup_survival_pct"] = round(100.0 * len(set(raw)) / max(len(raw), 1), 1)
    if (d / "bigrams_valid.jsonl").exists():
        out["valid_bigrams"] = sum(1 for _ in read_jsonl(d / "bigrams_valid.jsonl"))
    if (d / "n_calls.txt").exists():
        out["bigram_calls"] = int((d / "n_calls.txt").read_text().strip())
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/ablation"))
    ap.add_argument("--mucs", type=Path, default=Path("manifests/mucs_train.jsonl"))
    ap.add_argument("--test-refs", type=Path, default=Path("results/refs_test.jsonl"))
    ap.add_argument("--g26", type=Path, default=None, help="local sentences.jsonl for g26")
    ap.add_argument("--g26-hf", default="RohanRamesh/hi-en-synth-cs")
    ap.add_argument("--out", type=Path, default=Path("results/ablation/text_stats.json"))
    args = ap.parse_args(argv)

    test = [normalize(r["text"], "scoring") for r in read_jsonl(args.test_refs)]
    test_types = Counter((b.first.lower(), b.second.lower()) for t in test for b in cs_bigrams(t))
    mucs = [r["text"] for r in read_jsonl(args.mucs)] if args.mucs.exists() else []
    mucs_en = {w for t in mucs for w, lang in tag_words(normalize(t, "scoring"))
               if lang is Lang.EN}

    corpora: dict[str, tuple[list[str], dict]] = {}
    if args.g26 and args.g26.exists():
        corpora["g26"] = ([r["text"] for r in read_jsonl(args.g26)], {})
    else:
        try:
            from datasets import load_dataset

            ds = load_dataset(args.g26_hf, "synth_t2", split="train",
                              token=os.environ.get("HF_TOKEN")).select_columns(["text"])
            corpora["g26"] = (ds["text"], {})
        except Exception as e:  # noqa: BLE001 - the table is still useful without it
            print(f"[text_stats] g26 unavailable ({type(e).__name__}: {str(e)[:80]})")
    if args.data.exists():
        for d in sorted(p for p in args.data.iterdir() if (p / "sentences.jsonl").exists()):
            corpora[d.name] = ([r["text"] for r in read_jsonl(d / "sentences.jsonl")], yields(d))

    rows = []
    if mucs:
        rows.append({"corpus": "mucs_train", "label": "MUCS train (real)",
                     **profile(mucs, test_types, mucs_en)})
    rows.append({"corpus": "mucs_test", "label": "MUCS test (target)",
                 **profile(test, test_types, mucs_en)})
    for name, (texts, y) in corpora.items():
        rows.append({"corpus": name, "label": label(name),
                     **profile(texts, test_types, mucs_en), **y})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    cols = [("label", "corpus", 30), ("sentences", "sents", 7), ("est_hours", "~h", 6),
            ("latin_pct", "%Lat", 6), ("hi_matrix_pct", "%hi-mx", 7),
            ("latin_pct_hi_matrix", "%Lat|hi", 8), ("switches_per_sentence", "sw/sent", 8),
            ("mean_en_run", "en-run", 7), ("en_runs_ge4_pct", "%run>=4", 8),
            ("test_switch_coverage_pct", "test-cov", 9), ("valid_bigrams", "bigrams", 8),
            ("bigram_calls", "calls", 6)]
    print("  ".join(f"{h:<{w}}" if k == "label" else f"{h:>{w}}" for k, h, w in cols))
    for r in rows:
        print("  ".join(
            f"{str(r.get(k, '') if r.get(k) is not None else ''):<{w}}" if k == "label"
            else f"{str(r.get(k, '') if r.get(k) is not None else ''):>{w}}"
            for k, _, w in cols))
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
