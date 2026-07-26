"""GATE 2 --- verify Train_T1 is a usable subset of what was actually PUBLISHED.

`make_subset` asserts the subset property against Kaggle's local manifest, and
`push_to_hub --verify` checks synth_t2's row count / duration / sample rate. But
nothing checks the *join between them*: t1_ids.json is uploaded by a separate
raw `upload_file` call, and if its ids do not match the `utt_id` column in the
uploaded parquet, M6's `.filter()` silently selects zero rows and M6 trains on
nothing. A silent empty-subset is far worse than a crash, so assert it.

    # against the Hub (downloads the parquet; slow, ~2.5 GB for synth_t2)
    python scripts/verify_gate2.py --repo RohanRamesh/hi-en-synth-cs

    # against a local manifest (free -- use this on Kaggle right after the push)
    python scripts/verify_gate2.py --t2 manifests/train_t2.jsonl --t1 t1_ids.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _from_hub(repo: str, n_shards: int, config: str) -> tuple[list[str], list[float], list[str]]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem, hf_hub_download

    fs = HfFileSystem()
    ids: list[str] = []
    durs: list[float] = []
    for i in range(n_shards):
        p = f"datasets/{repo}/{config}/train-{i:05d}-of-{n_shards:05d}.parquet"
        with fs.open(p, "rb") as f:
            t = pq.read_table(f, columns=["utt_id", "dur"])
        ids += t.column("utt_id").to_pylist()
        durs += t.column("dur").to_pylist()
        print(f"  shard {i}: {t.num_rows:,} rows")

    t1_path = hf_hub_download(repo, "t1_ids.json", repo_type="dataset")
    return ids, durs, json.loads(Path(t1_path).read_text(encoding="utf-8"))


def _from_local(t2: Path, t1: Path) -> tuple[list[str], list[float], list[str]]:
    from csasr.manifest import read_jsonl

    rows = list(read_jsonl(t2))
    return (
        [r["utt_id"] for r in rows],
        [r.get("dur") or 0.0 for r in rows],
        json.loads(t1.read_text(encoding="utf-8")),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", help="check the Hub, e.g. RohanRamesh/hi-en-synth-cs")
    ap.add_argument("--config", default="synth_t2")
    ap.add_argument("--num-shards", type=int, default=6)
    ap.add_argument("--t2", type=Path, help="local train_t2.jsonl (instead of --repo)")
    ap.add_argument("--t1", type=Path, help="local t1_ids.json")
    ap.add_argument("--t1-hours", type=float, default=8.0)
    ap.add_argument("--tol", type=float, default=0.05, help="hours tolerance")
    args = ap.parse_args(argv)

    if args.repo:
        ids, durs, t1 = _from_hub(args.repo, args.num_shards, args.config)
        where = f"hub:{args.repo}:{args.config}"
    elif args.t2 and args.t1:
        ids, durs, t1 = _from_local(args.t2, args.t1)
        where = str(args.t2)
    else:
        ap.error("pass --repo, or both --t2 and --t1")

    t2_set, t1_set = set(ids), set(t1)
    by_id = dict(zip(ids, durs))
    t1_h = sum(by_id.get(i) or 0.0 for i in t1_set) / 3600
    missing = t1_set - t2_set

    print(f"\nTrain_T2 ({where}): {len(ids):,} rows, {len(t2_set):,} unique, "
          f"{sum(d or 0 for d in durs) / 3600:.2f} h")
    print(f"t1_ids.json        : {len(t1):,} ids, {len(t1_set):,} unique")
    print(f"T1 ids missing from T2 : {len(missing):,}"
          + (f"  e.g. {list(missing)[:5]}" if missing else ""))
    print(f"Train_T1 hours         : {t1_h:.2f} h")

    fails = []
    if missing:
        fails.append(f"{len(missing):,} T1 ids are absent from T2 -- M6 would train on a "
                     "partial or empty subset")
    if len(t1) != len(t1_set):
        fails.append("t1_ids.json contains duplicates")
    if len(ids) != len(t2_set):
        fails.append(f"T2 has {len(ids) - len(t2_set):,} duplicate utt_ids")
    if not t1_set < t2_set:
        fails.append("T1 is not a strict subset of T2")
    if abs(t1_h - args.t1_hours) > args.tol:
        fails.append(f"T1 is {t1_h:.2f} h, expected {args.t1_hours} +/- {args.tol}")

    if fails:
        print("\nGATE 2: FAIL")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\nGATE 2: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
