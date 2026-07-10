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
from .cba import cba
from .grouping import concat_refs
from .mer import mer

__all__ = ["score", "main"]


def score(
    refs_path: Path,
    hyps_path: Path,
    *,
    mer_mode: str = "word",
    preset: str = "scoring",
    group: str = "none",
) -> dict:
    ref_rows = list(read_jsonl(refs_path))
    if group == "recording":
        refs = concat_refs(ref_rows)
    else:
        refs = {r["utt_id"]: r["text"] for r in ref_rows}
    hyps = {r["utt_id"]: r.get("hyp", r.get("text", "")) for r in read_jsonl(hyps_path)}

    missing = set(refs) - set(hyps)
    extra = set(hyps) - set(refs)
    if missing and len(missing) == len(refs):
        raise SystemExit(
            "no reference id matches any hypothesis id. Did you decode at "
            "recording level but score per-utterance? Pass --group recording, "
            "or use the refs file decode.py wrote next to the hypotheses.\n"
            f"  refs e.g. {sorted(refs)[:2]}\n  hyps e.g. {sorted(hyps)[:2]}"
        )
    if missing:
        raise SystemExit(
            f"{len(missing)} reference utterances have no hypothesis "
            f"(e.g. {sorted(missing)[:3]}). Decoding is incomplete."
        )
    if extra:
        print(f"[score] warning: ignoring {len(extra)} hypotheses with no reference")

    ids = sorted(refs)
    r = [refs[i] for i in ids]
    h = [hyps[i] for i in ids]

    m = mer(r, h, mode=mer_mode, preset=preset)
    c = cba(r, h, preset=preset)

    return {
        "n_items": len(ids),
        # What *this* call did to the refs. It does not claim the refs file's
        # own granularity: decode.py --mode recording already writes grouped refs.
        "grouping_applied": group,
        "mer": round(m, 2),
        "mer_mode": mer_mode,
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
                    help="collapse a per-utterance refs manifest to one ref per recording")
    ap.add_argument("--name", default=None, help="label for this system")
    ap.add_argument("--out", type=Path, default=None, help="append JSON result here")
    args = ap.parse_args(argv)

    res = score(args.refs, args.hyps, mer_mode=args.mer_mode, preset=args.preset, group=args.group)
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
