#!/usr/bin/env python
"""GATE 1.5 - audit the synthetic corpus BEFORE spending 4-6h of TTS on it.

Gate 1 counts sentences. It does not check whether they are any GOOD. This does:
label leakage, script mix, code-switch density, domain overlap with the real
corpus, and the TTS duration you can actually expect.

    python scripts/audit_sentences.py --sentences temp/sentences.jsonl \
                                      --mucs manifests/mucs_train.jsonl
    python scripts/audit_sentences.py --hf RohanRamesh/hi-en-synth-cs \
                                      --mucs manifests/mucs_train.jsonl
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csasr.lid import Lang, count_words, cs_bigrams, word_lang  # noqa: E402
from csasr.manifest import read_jsonl  # noqa: E402
from csasr.normalize import normalize  # noqa: E402

_LABEL = re.compile(r"^\s*\**\s*(english|hindi|hinglish)\s*\**\s*[:\-]", re.IGNORECASE)

# Parler-TTS speaks Indic at roughly this rate. A rough estimate is enough to
# tell 8h from 22h, which is the only decision it feeds.
WORDS_PER_SEC = 2.4


def _mix(texts: list[str]) -> tuple[float, float]:
    h = e = o = 0
    for t in texts:
        c = count_words(normalize(t, "scoring"))
        h += c[Lang.HI]
        e += c[Lang.EN]
        o += c[Lang.OTHER]
    tot = h + e + o or 1
    return h / tot, e / tot


def _en_vocab(texts: list[str]) -> Counter:
    v: Counter = Counter()
    for t in texts:
        for w in normalize(t, "scoring").split():
            if word_lang(w) is Lang.EN:
                v[w] += 1
    return v


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sentences", type=Path)
    src.add_argument("--hf", help="dataset repo; reads its `sentences` config")
    ap.add_argument("--mucs", type=Path, default=Path("manifests/mucs_train.jsonl"))
    ap.add_argument("--sample", type=int, default=12)
    args = ap.parse_args(argv)

    if args.hf:
        from datasets import load_dataset

        ds = load_dataset(args.hf, "sentences", split="train",
                          token=os.environ.get("HF_TOKEN"))
        sents = [r["text"] for r in ds]
    else:
        sents = [r["text"] for r in read_jsonl(args.sentences)]

    mucs = [r["text"] for r in read_jsonl(args.mucs)] if args.mucs.exists() else []
    print(f"{len(sents):,} synthetic sentences   vs   {len(mucs):,} real MUCS sentences\n")

    fails = 0

    # 1. Label leak -- these would be SPOKEN by the TTS and LEARNED by Whisper.
    leak = [s for s in sents if _LABEL.match(s)]
    ok = not leak
    fails += not ok
    print(f"1. matrix-language label leak   : {len(leak):,}   {'OK' if ok else 'BLOCKER'}")
    if leak:
        print(f"     e.g. {leak[0][:70]}")

    # 2. Monolingual -- no switch point can exist, so the row is dead weight.
    mono = [s for s in sents
            if (lambda c: c[Lang.HI] == 0 or c[Lang.EN] == 0)(
                count_words(normalize(s, "scoring")))]
    ok = not mono
    fails += not ok
    print(f"2. monolingual rows             : {len(mono):,}   {'OK' if ok else 'BLOCKER'}")

    if not mucs:
        print("\n(no MUCS manifest: skipping the distribution checks)")
        return 1 if fails else 0

    # 3. Script mix vs the real domain.
    sh, se = _mix(sents)
    mh, me = _mix(mucs)
    print(f"\n3. script mix   synthetic       : {sh:5.1%} Devanagari  {se:5.1%} Latin")
    print(f"                real MUCS       : {mh:5.1%} Devanagari  {me:5.1%} Latin")
    print(f"   -> synthetic is {'MORE' if se > me else 'less'} English-heavy "
          f"({se - me:+.1%} Latin)")

    # 4. Code-switch density.
    def dens(texts):
        return sum(len(list(cs_bigrams(normalize(t, "scoring")))) for t in texts) / len(texts)
    print(f"\n4. switch points / sentence     : {dens(sents):.2f} synthetic "
          f"vs {dens(mucs):.2f} real")

    # 5. Domain overlap. TOKEN coverage is the honest number: it says how much of
    #    the synthetic English actually comes from the target domain's vocabulary.
    sv, mv = _en_vocab(sents), _en_vocab(mucs)
    shared = set(sv) & set(mv)
    tok = sum(sv[w] for w in shared) / max(sum(sv.values()), 1)
    print(f"\n5. English vocab overlap with MUCS")
    print(f"     types : {len(shared):,} / {len(sv):,} = {len(shared)/max(len(sv),1):.1%}")
    print(f"     tokens: {tok:.1%} of synthetic English words appear in the real corpus")
    oov = [(w, c) for w, c in sv.most_common(400) if w not in mv][:10]
    if oov:
        print(f"     most common OUT-of-domain: "
              + ", ".join(f"{w}({c})" for w, c in oov))

    # 6. How many hours of audio is this actually going to be?
    wc = [len(s.split()) for s in sents]
    est_h = sum(w / WORDS_PER_SEC for w in wc) / 3600
    print(f"\n6. length: median {sorted(wc)[len(wc)//2]} words, mean {sum(wc)/len(wc):.1f}")
    print(f"   estimated TTS duration ~{est_h:.1f} h   (paper's Train_T2 = 22 h, "
          f"Train_T1 = 8 h)")
    if est_h < 8:
        print("   BLOCKER: not even enough for Train_T1 (8 h).")
        fails += 1
    elif est_h < 20:
        print("   NOTE: short of the paper's 22 h. Train_T1 (8 h) is safe; Train_T2")
        print("         will be smaller than theirs -- document it as a deviation.")

    print(f"\n--- random sample of {args.sample} ---")
    random.seed(7)
    for s in random.sample(sents, min(args.sample, len(sents))):
        print("   ", s[:96])

    print(f"\n{'AUDIT PASSED' if not fails else f'AUDIT FAILED ({fails} blocker(s))'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
