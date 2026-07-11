"""Score a hypotheses file against a reference manifest.

Runs on CPU. Reads the JSONL that `decode.py` writes on Kaggle and emits MER +
CBA. Both metrics normalize with the `scoring` preset, which preserves
Devanagari combining marks (see `normalize.py`).

`--group recording` collapses a per-utterance reference manifest into one
reference per recording, matching `decode.py --mode recording`. MER is a
corpus-level ratio (total errors / total reference words), so concatenating
before scoring is equivalent; CBA differs only for bigrams that straddle a
segment join, which concatenation correctly *includes*.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..manifest import read_jsonl
from .cba import cba, cba_grouped
from .grouping import concat_refs, recording_id
from .mer import mer

__all__ = ["score", "main"]


def score(
    refs_path: Path,
    hyps_path: Path,
    *,
    mer_mode: str = "word",
    preset: str = "scoring",
    group: str = "none",
    cba_mode: str = "adjacent",
) -> dict:
    ref_rows = list(read_jsonl(refs_path))
    hyps = {r["utt_id"]: r.get("hyp", r.get("text", "")) for r in read_jsonl(hyps_path)}

    if group == "recording":
        # MER: concatenate each recording's utterances (MER is a corpus-level
        # ratio, so this is equivalent).
        refs = concat_refs(ref_rows)
    else:
        refs = {r["utt_id"]: r["text"] for r in ref_rows}

    missing = set(refs) - set(hyps)
    extra = set(hyps) - set(refs)
    if missing and len(missing) == len(refs):
        raise SystemExit(
            "no reference id matches any hypothesis id. Recording-level "
            "hypotheses need `--group recording` and a PER-UTTERANCE refs "
            "manifest (decode.py --refs-out writes one).\n"
            f"  refs e.g. {sorted(refs)[:2]}\n  hyps e.g. {sorted(hyps)[:2]}"
        )
    if missing:
        raise SystemExit(
            f"{len(missing)} references have no hypothesis "
            f"(e.g. {sorted(missing)[:3]}). Decoding is incomplete."
        )
    if extra:
        print(f"[score] warning: ignoring {len(extra)} hypotheses with no reference")

    ids = sorted(refs)
    m = mer([refs[i] for i in ids], [hyps[i] for i in ids], mode=mer_mode, preset=preset)

    # CBA: the denominator must come from the reference UTTERANCES (Table 1's
    # 4,189 HE / 5,176 EH). Concatenating first would fabricate ~1,000 extra HE
    # bigrams at segment joins and halve the score.
    if group == "recording":
        c = cba_grouped(
            [(r["utt_id"], r["text"]) for r in ref_rows],
            hyps, recording_id, preset=preset, mode=cba_mode,
        )
    else:
        c = cba([refs[i] for i in ids], [hyps[i] for i in ids],
                preset=preset, mode=cba_mode)

    return {
        "n_items": len(ids),
        "grouping_applied": group,
        "mer": round(m, 2),
        "mer_mode": mer_mode,
        "cba_mode": cba_mode,
        "cba_he": round(c.he, 2),
        "cba_eh": round(c.eh, 2),
        "cba_total": round(c.total, 2),
        "he_matched": c.he_matched,
        "he_total": c.he_total,
        "eh_matched": c.eh_matched,
        "eh_total": c.eh_total,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refs", type=Path, required=True, help="reference manifest JSONL")
    ap.add_argument("--hyps", type=Path, required=True, help="hypotheses JSONL")
    ap.add_argument("--mer-mode", default="word", choices=["word", "hybrid"])
    ap.add_argument("--preset", default="scoring")
    ap.add_argument("--group", default="none", choices=["none", "recording"],
                    help="hypotheses are per recording; refs manifest is per utterance")
    ap.add_argument("--cba-mode", default="adjacent", choices=["adjacent", "lenient"],
                    help="'adjacent' = both words correct and adjacent (literal reading); "
                         "'lenient' = both words recognized somewhere (reproduces the "
                         "paper's numbers). See cba.py.")
    ap.add_argument("--name", default=None, help="label for this system")
    ap.add_argument("--out", type=Path, default=None, help="append JSON result here")
    args = ap.parse_args(argv)

    res = score(args.refs, args.hyps, mer_mode=args.mer_mode, preset=args.preset,
                group=args.group, cba_mode=args.cba_mode)
    if args.name:
        res = {"system": args.name, **res}

    print(json.dumps(res, indent=2, ensure_ascii=False))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
