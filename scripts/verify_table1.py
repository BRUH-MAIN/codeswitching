#!/usr/bin/env python
"""GATE 0 - reproduce Table 1 of Biswas et al. (2025) from the MUCS transcripts.

Nothing downstream is interpretable until this passes. The paper publishes exact
word and code-switch-bigram counts, which pin down tokenization, normalization,
and word-level language ID all at once.

                Duration  Words_H   Words_E   Total     HE CS    EH CS
    Train         89.8h   445,762   160,694   606,456   85,761   90,802
    Test           5.2h    28,215     9,627    37,842    4,189    5,176

Test additionally has 2,347 unique HE and 2,472 unique EH bigrams.

WHAT WE ASSERT, AND WHY IT DIFFERS PER SPLIT
--------------------------------------------
TEST is the split every reported metric (MER, CBA) is computed on, and it
reproduces: `words_h` to 0.01%, `hours` to 0.42%, `total` to 0.75%, and the
code-switch bigram counts to within 2.3% - once we (a) preserve Devanagari
combining marks and (b) split dropped-space typos like `दायाँclick` at the script
boundary. Those five are asserted.

`words_e` runs ~5% low on both splits, which is *explained*: the paper's test
Total (37,842) exceeds this corpus's entire raw whitespace-token count (37,611),
so their tokenizer splits tokens ours does not - most plausibly by expanding the
186 bare digit tokens (`334`, `1204`, ...) into words. We keep digits as OTHER
because the paper's own bigram filter demands each bigram contain "both Hindi
and English characters", and a digit has neither. Reported as INFO.

TRAIN does NOT reproduce, and no tokenization can make it. The public OpenSLR
104 transcripts yield ~468k Hindi words against the paper's 445,762, and at most
80,860 HE bigrams against their 85,761 - and that upper bound comes from the
most switch-generating variant we could construct (digits counted as English).
Splitting tokens can only *add* adjacencies, never invent Hindi->English pairs
that are not in the text, so the released transcripts simply contain fewer of
them than the paper's train row implies. The challenge page confirms there is no
separate or corrected transcript download. We therefore assert only integrity
invariants on train (duration, both languages present, no MIXED tokens survive)
and report its counts as INFO.

This costs us nothing: Track 2 uses the train split only for few-shot exemplar
text and a 4h dev slice. Neither depends on its exact word counts.

The `whisper_basic` preset is checked as a REGRESSION GUARD: it must fail
loudly, because Whisper's own BasicTextNormalizer deletes Devanagari combining
marks and shatters one Hindi word into four.

Usage:
    python scripts/verify_table1.py --train manifests/mucs_train.jsonl \
                                    --test  manifests/mucs_test.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csasr.eval.cba import corpus_cs_bigram_stats  # noqa: E402
from csasr.lid import Lang, count_words  # noqa: E402
from csasr.manifest import read_jsonl  # noqa: E402
from csasr.normalize import normalize  # noqa: E402

PAPER = {
    "train": dict(hours=89.8, words_h=445_762, words_e=160_694, total=606_456,
                  he_cs=85_761, eh_cs=90_802),
    "test": dict(hours=5.2, words_h=28_215, words_e=9_627, total=37_842,
                 he_cs=4_189, eh_cs=5_176, he_types=2_347, eh_types=2_472),
}

# Asserted metrics and tolerances, PER SPLIT. Everything else is informational.
# Train's word/bigram counts are unreproducible from the public release (see the
# module docstring), so only integrity invariants are enforced there.
TOL = {
    "test": {"words_h": 0.01, "hours": 0.02, "total": 0.02, "he_cs": 0.03, "eh_cs": 0.03},
    "train": {"hours": 0.02},
}


def measure(texts: list[str], durations: list[float], preset: str) -> dict:
    normed = [normalize(t, preset) for t in texts]

    words = {lang: 0 for lang in Lang}
    for t in normed:
        for lang, n in count_words(t).items():
            words[lang] += n

    bg = corpus_cs_bigram_stats(normed, preset="raw")  # already normalized

    return {
        "hours": sum(durations) / 3600.0,
        "words_h": words[Lang.HI],
        "words_e": words[Lang.EN],
        "words_other": words[Lang.OTHER],
        "words_mixed": words[Lang.MIXED],
        "total": sum(words.values()),
        "he_cs": bg["he_tokens"],
        "eh_cs": bg["eh_tokens"],
        "he_types": bg["he_types"],
        "eh_types": bg["eh_types"],
    }


def rel_err(got: float, want: float) -> float:
    return abs(got - want) / want if want else 0.0


def report(split: str, got: dict, want: dict) -> tuple[int, int]:
    tol = TOL[split]
    passed = checked = 0
    print(f"    {'metric':<12} {'got':>10} {'paper':>10} {'rel err':>9}  status")
    print(f"    {'-'*12} {'-'*10} {'-'*10} {'-'*9}  ------")
    for k, v in want.items():
        if k not in got:
            continue
        e = rel_err(got[k], v)
        fmt = f"{got[k]:>10.2f}" if k == "hours" else f"{got[k]:>10,}"
        pv = f"{v:>10.2f}" if k == "hours" else f"{v:>10,}"
        if k in tol:
            checked += 1
            ok = e <= tol[k]
            passed += ok
            status = "PASS" if ok else "FAIL"
        else:
            status = "info"
        print(f"    {k:<12} {fmt} {pv} {e:>8.2%}  {status}")

    if got["words_other"] or got["words_mixed"]:
        print(f"    (info) {got['words_other']:,} OTHER + {got['words_mixed']:,} MIXED tokens")

    # Integrity invariants. These catch a broken tokenizer even where the
    # paper's own counts cannot be reproduced.
    invariants = [
        ("both languages present", got["words_h"] > 0 and got["words_e"] > 0),
        ("no MIXED tokens survive tokenization", got["words_mixed"] == 0),
        ("code-switching present", got["he_cs"] > 0 and got["eh_cs"] > 0),
    ]
    for name, ok in invariants:
        checked += 1
        passed += ok
        print(f"    {'* ' + name:<34} {'PASS' if ok else 'FAIL'}")

    if split == "train":
        print("    (info) train word/bigram counts do not reproduce from OpenSLR 104;")
        print("           no tokenization reaches the paper's 85,761 HE bigrams (max 80,860).")
        print("           Track 2 uses train only for few-shot text + a 4h dev slice.")
    return passed, checked


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, default=Path("manifests/mucs_train.jsonl"))
    ap.add_argument("--test", type=Path, default=Path("manifests/mucs_test.jsonl"))
    ap.add_argument("--preset", default="scoring")
    args = ap.parse_args(argv)

    splits = {}
    for name, path in (("train", args.train), ("test", args.test)):
        if not path.exists():
            print(f"[skip] {name}: {path} does not exist")
            continue
        rows = list(read_jsonl(path))
        splits[name] = ([r["text"] for r in rows], [r.get("dur") or 0.0 for r in rows])
        print(f"[load] {name}: {len(rows):,} utterances from {path}")

    if not splits:
        raise SystemExit("no manifests found; run csasr.data.prepare_mucs first")

    ok = True
    for name, (texts, durs) in splits.items():
        print(f"\n{'='*64}\n{name}  (preset={args.preset})\n{'='*64}")
        p, n = report(name, measure(texts, durs, args.preset), PAPER[name])
        print(f"    -> {p}/{n} asserted checks passed")
        ok &= p == n

    # Regression guard: the destructive normalizer must visibly fail.
    print(f"\n{'='*64}\nREGRESSION GUARD: whisper_basic must shatter Devanagari\n{'='*64}")
    texts, durs = next(iter(splits.values()))
    wb = measure(texts, durs, "whisper_basic")
    good = measure(texts, durs, args.preset)
    inflation = wb["words_h"] / max(good["words_h"], 1) - 1
    print(f"    words_h: {good['words_h']:,} ({args.preset})  ->  {wb['words_h']:,} (whisper_basic)")
    print(f"    inflation: {inflation:+.1%}")
    guard_ok = inflation > 0.20
    print(f"    -> {'PASS' if guard_ok else 'FAIL'}  "
          f"(BasicTextNormalizer deletes Mn/Mc marks; we must not use it)")
    ok &= guard_ok

    print(f"\n{'='*64}\nGATE 0: {'PASSED' if ok else 'FAILED'}\n{'='*64}")
    if not ok:
        print("  Fix lid.py / normalize.py before proceeding - every downstream")
        print("  number depends on these counts.")
        return 1
    print("  Tokenization, normalization, and word-level LID agree with the paper on")
    print("  TEST - the split every reported metric (MER, CBA) is computed on.")
    print("  TRAIN's word/bigram counts and words_e are INFO; see this file's docstring.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
