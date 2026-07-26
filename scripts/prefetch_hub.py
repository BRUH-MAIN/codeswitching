"""Pre-download everything training needs, for compute nodes without internet.

Many HPC clusters give login nodes internet and compute nodes none. Run this on
a login node, then launch training with HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
and the same HF_HOME -- every load_dataset / from_pretrained call resolves from
cache instead of reaching out and failing an hour into your allocation.

    HF_HOME=/scratch/$USER/hf python scripts/prefetch_hub.py \
        --model openai/whisper-large-v2

Prints the resolved cache size at the end so you can check it against your
scratch quota before committing a job.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SYNTH = "RohanRamesh/hi-en-synth-cs"
REAL = "RohanRamesh/mucs-he-cs"


def _dir_size(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="openai/whisper-large-v2")
    ap.add_argument("--skip-cv", action="store_true",
                    help="skip Common Voice (M8 only); M6/M7 do not need it")
    args = ap.parse_args(argv)

    # Never read a token from argv: it lands in tracebacks and in `ps`.
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[prefetch] WARNING: HF_TOKEN unset; private configs will fail")

    home = os.environ.get("HF_HOME")
    print(f"[prefetch] HF_HOME = {home or '(default ~/.cache/huggingface)'}")
    if not home:
        print("[prefetch] set HF_HOME to scratch -- the model alone is ~6 GB and the "
              "featurized Arrow cache later reaches 35 GB")

    from datasets import load_dataset
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    print(f"[prefetch] model {args.model}")
    WhisperProcessor.from_pretrained(args.model)
    WhisperForConditionalGeneration.from_pretrained(args.model)

    wanted = [(SYNTH, "synth_t2"), (REAL, "dev"), (REAL, "test")]
    if not args.skip_cv:
        wanted += [(REAL, "cv_hi"), (REAL, "cv_en")]

    failed = []
    for repo, cfg in wanted:
        try:
            ds = load_dataset(repo, cfg, split="train", token=token)
            print(f"[prefetch] {repo}:{cfg:9s} {len(ds):>7,} rows")
        except Exception as e:  # a missing cv_en must not block M6/M7
            print(f"[prefetch] {repo}:{cfg:9s} FAILED: {type(e).__name__}: {e}")
            failed.append(f"{repo}:{cfg}")

    from huggingface_hub import hf_hub_download

    try:
        p = hf_hub_download(SYNTH, "t1_ids.json", repo_type="dataset", token=token)
        print(f"[prefetch] t1_ids.json -> {p}")
    except Exception as e:
        print(f"[prefetch] t1_ids.json FAILED: {e}")
        failed.append("t1_ids.json")

    if home and Path(home).exists():
        print(f"[prefetch] cache is now {_dir_size(Path(home)):.1f} GB")

    if failed:
        print(f"\n[prefetch] {len(failed)} item(s) unavailable: {', '.join(failed)}")
        print("  cv_en missing -> M6 and M7 are unaffected; only M8 needs it.")
        return 1
    print("\n[prefetch] all cached. Launch with HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
