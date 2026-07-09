"""Build Train_T1 (~8h) as a by-reference subset of Train_T2 (~22h).

Paper: "creating two synthetic training sets: Train_T1 (8 hours) and Train_T2
(22 hours), where Train_T1 is a subset of Train_T2."

We emit a JSON list of sent_ids, applied with `.filter()` at load time. Copying
8 hours of audio we already have would cost ~0.9 GB for nothing.

Selection is a seeded shuffle stratified over bigrams, so the 8h subset covers
as many distinct code-switch points as possible rather than over-sampling a few.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from ..manifest import read_jsonl

__all__ = ["select_subset", "main"]


def select_subset(rows: list[dict], target_hours: float, seed: int = 1234) -> list[str]:
    """Round-robin over bigram groups until the duration target is met."""
    rng = random.Random(seed)

    by_bigram: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_bigram[r.get("bigram") or ""].append(r)

    groups = list(by_bigram.values())
    for g in groups:
        rng.shuffle(g)
    rng.shuffle(groups)

    target = target_hours * 3600.0
    picked: list[str] = []
    acc = 0.0
    depth = 0
    while acc < target:
        progressed = False
        for g in groups:
            if depth >= len(g):
                continue
            progressed = True
            r = g[depth]
            picked.append(r["utt_id"])
            acc += r.get("dur") or 0.0
            if acc >= target:
                break
        if not progressed:
            break  # exhausted the corpus before hitting the target
        depth += 1

    return picked


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t2", type=Path, required=True, help="train_t2.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="t1_ids.json")
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)

    rows = list(read_jsonl(args.t2))
    total_h = sum(r.get("dur") or 0.0 for r in rows) / 3600
    ids = select_subset(rows, args.hours, args.seed)

    by_id = {r["utt_id"]: r for r in rows}
    picked_h = sum(by_id[i].get("dur") or 0.0 for i in ids) / 3600

    # GATE 2: T1 must be a strict subset of T2.
    assert set(ids) <= set(by_id), "Train_T1 contains ids absent from Train_T2"
    assert len(ids) == len(set(ids)), "Train_T1 contains duplicate ids"
    assert len(ids) < len(rows), "Train_T1 is not a strict subset of Train_T2"

    n_bigrams_t2 = len({r.get("bigram") for r in rows})
    n_bigrams_t1 = len({by_id[i].get("bigram") for i in ids})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(sorted(ids), indent=0), encoding="utf-8")

    print(f"[subset] Train_T2: {len(rows):,} clips, {total_h:.2f} h, {n_bigrams_t2:,} bigrams")
    print(f"[subset] Train_T1: {len(ids):,} clips, {picked_h:.2f} h, {n_bigrams_t1:,} bigrams "
          f"({n_bigrams_t1/max(n_bigrams_t2,1):.0%} bigram coverage)")
    print(f"[subset] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
