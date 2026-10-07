"""Post-hoc checkpoint selection with the test decoder, then one test decode.

WHY NOT SELECT DURING TRAINING
------------------------------
The in-training dev metric decodes isolated ~6 s utterances greedily. On this
task that is a different problem from the one the test set poses (see
`decode.py`), and it is unstable: across consecutive evals of one run dev MER
read 52.7, 59.1, 75.2, 51.3 -- so "best checkpoint" was close to a coin toss,
and each of those evals cost ~18 minutes.

Here every saved checkpoint is decoded on dev at RECORDING level by the same
engine and settings the test set gets, the lowest-MER checkpoint wins, and only
that one touches the test set. Dev has 2,362 utterances from 509 recordings, so
a pass takes minutes. The same code produces the zero-shot rows (`--zero-shot`),
which double as step 0 of every dev curve.

Works for Whisper (full checkpoints or LoRA/QLoRA adapters, via `ct2.py`) and
for CTC models (`--engine ctc`, via `decode_ctc.py`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from ..lid import Lang, count_words
from ..manifest import read_jsonl
from .score import score

__all__ = ["latin_pct", "score_all", "pick_best", "write_row", "main"]

_STEP = re.compile(r"checkpoint-(\d+)$")


def latin_pct(hyps_path: Path) -> float:
    """Share of script-identifiable words written in Latin. The reference is
    ~24%; a model that renders English terms in Devanagari scores far below."""
    hi = en = 0
    for r in read_jsonl(hyps_path):
        c = count_words(r.get("hyp", ""))
        hi += c[Lang.HI]
        en += c[Lang.EN]
    return round(100.0 * en / max(hi + en, 1), 2)


def score_all(refs: Path, hyps: Path) -> dict:
    """Hybrid MER plus CBA under both matching rules, flattened into one dict."""
    kw = dict(mer_mode="hybrid", group="recording")
    len_ = score(refs, hyps, cba_mode="lenient", **kw)
    adj = score(refs, hyps, cba_mode="adjacent", **kw)
    return {
        "mer": len_["mer"],
        "cba_he": len_["cba_he"], "cba_eh": len_["cba_eh"], "cba_total": len_["cba_total"],
        "cba_he_adj": adj["cba_he"], "cba_eh_adj": adj["cba_eh"],
        "latin_pct": latin_pct(hyps),
    }


def pick_best(curve: list[dict]) -> dict:
    """Lowest dev MER; ties go to the EARLIER step (less drift from pretrained)."""
    return min(curve, key=lambda r: (r["mer"], r["step"]))


def write_row(results_dir: Path, row: dict, hyp_test: Path) -> Path:
    """One file per run, keyed by run name.

    Several ablation jobs finish concurrently on different GPUs; a shared JSONL
    would need read-modify-write locking over NFS. One file per run cannot
    collide, and re-running a run overwrites its own row instead of leaving two
    for the report to double-count. The test hypotheses ride along so the
    report can bootstrap without the multi-GB run directories.
    """
    rows, hyps = results_dir / "rows", results_dir / "hyps"
    rows.mkdir(parents=True, exist_ok=True)
    hyps.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(hyp_test, hyps / f"{row['run']}.jsonl")
    out = rows / f"{row['run']}.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, out)
    return out


def _decode(args, model: str, config: str, out: Path, refs: Path, ct2_cache: Path) -> None:
    """One decode in a subprocess, so each model's GPU memory is fully released."""
    if out.exists() and out.stat().st_size > 0:
        print(f"[select] reusing {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial.jsonl")
    common = ["--model", model, "--test-hf", args.real_hf, "--test-config", config,
              "--out", str(tmp), "--refs-out", str(refs), "--batch-size", str(args.batch_size)]
    if args.limit:
        common += ["--limit", str(args.limit)]
    if args.engine == "ctc":
        cmd = [sys.executable, "-m", "csasr.eval.decode_ctc", *common]
        if args.target_lang:
            cmd += ["--target-lang", args.target_lang]
    else:
        cmd = [sys.executable, "-m", "csasr.eval.decode", *common,
               "--language", "none", "--lang-detect-segments", "8",
               "--ct2-cache", str(ct2_cache)]
    subprocess.run(cmd, check=True)
    os.replace(tmp, out)  # only a COMPLETE decode is ever reused


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="training output dir (holds checkpoint-*), or a scratch dir "
                         "for --zero-shot")
    ap.add_argument("--name", required=True, help="unique run name: the results key")
    ap.add_argument("--zero-shot", default=None, metavar="MODEL",
                    help="score this pretrained model instead of checkpoints")
    ap.add_argument("--zero-shot-dir", type=Path, default=None,
                    help="a finished --zero-shot run dir for the same base model; "
                         "adds it to the dev curve as step 0 (never selected)")
    ap.add_argument("--engine", choices=("whisper", "ctc"), default="whisper")
    ap.add_argument("--target-lang", default=None, help="ctc zero-shot: MMS adapter, e.g. hin")
    ap.add_argument("--real-hf", default="RohanRamesh/mucs-he-cs")
    ap.add_argument("--results-dir", type=Path, default=Path("results/ablation"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--meta", nargs="*", default=[], metavar="KEY=VALUE",
                    help="extra columns for the results row (corpus=..., asr=..., ft=...)")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke tests: decode only the first N utterances of each split")
    ap.add_argument("--prune", action="store_true",
                    help="delete the weights of every checkpoint that was not selected")
    args = ap.parse_args(argv)

    run = args.run_dir
    sel = run / "select"
    ct2 = run / "ct2"
    # Per run, not shared: concurrent jobs would otherwise rewrite one file
    # under each other's readers.
    refs_dev = sel / "refs_dev.jsonl"
    refs_test = sel / "refs_test.jsonl"

    if args.zero_shot:
        candidates = [(0, args.zero_shot)]
    else:
        candidates = sorted(
            (int(m.group(1)), str(d)) for d in run.iterdir()
            if d.is_dir() and (m := _STEP.search(d.name))
        )
        if not candidates:
            raise SystemExit(f"{run}: no checkpoint-* directories to select from")

    curve = []
    for step, model in candidates:
        hyp = sel / f"dev_step{step}.jsonl"
        _decode(args, model, "dev", hyp, refs_dev, ct2)
        row = {"step": step, **score_all(refs_dev, hyp)}
        curve.append(row)
        print(f"[select] dev step {step:>5}  MER {row['mer']:6.2f}  CBA-HE {row['cba_he']:5.1f}  "
              f"CBA-EH {row['cba_eh']:5.1f}  Latin {row['latin_pct']:4.1f}%")

    best = pick_best(curve)
    best_model = dict(candidates)[best["step"]]
    print(f"[select] selected step {best['step']} (dev MER {best['mer']:.2f})")

    hyp_test = run / "hyp_test.jsonl"
    marker = run / "hyp_test.step"
    # A test decode left over from a different selection must not be reused.
    if hyp_test.exists() and (not marker.exists()
                              or marker.read_text().strip() != str(best["step"])):
        hyp_test.unlink()
    _decode(args, best_model, "test", hyp_test, refs_test, ct2)
    marker.write_text(str(best["step"]))
    test = score_all(refs_test, hyp_test)

    if args.zero_shot_dir and (args.zero_shot_dir / "select" / "dev_curve.json").exists():
        zs = json.loads((args.zero_shot_dir / "select" / "dev_curve.json").read_text())
        curve = zs["curve"][:1] + curve
    (sel / "dev_curve.json").write_text(
        json.dumps({"run": args.name, "selected_step": best["step"], "curve": curve}, indent=2),
        encoding="utf-8")

    meta = dict(kv.split("=", 1) for kv in args.meta)
    run_meta = {}
    if (run / "run_meta.json").exists():
        run_meta = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
    row = {
        "run": args.name, **meta,
        "selected_step": best["step"], "dev_mer": best["mer"],
        **test,
        **{k: run_meta[k] for k in ("params_total", "params_trainable", "trainable_pct",
                                    "peak_vram_gb", "train_runtime_s", "lr", "train_utts",
                                    "resumed") if k in run_meta},
    }
    out = write_row(args.results_dir, row, hyp_test)
    if not (args.results_dir / "refs_test.jsonl").exists():
        shutil.copyfile(refs_test, args.results_dir / "refs_test.jsonl")
    print(f"[select] TEST {args.name}: MER {test['mer']:.2f}  CBA-HE {test['cba_he']:.1f}  "
          f"CBA-EH {test['cba_eh']:.1f}  Latin {test['latin_pct']:.1f}%  -> {out}")

    # CTranslate2 copies are derived; the selected one is not needed again either
    # because its hypotheses are on disk.
    shutil.rmtree(ct2, ignore_errors=True)
    if args.prune and not args.zero_shot:
        for step, model in candidates:
            if step == best["step"]:
                continue
            for f in Path(model).glob("*.safetensors"):
                f.unlink()
        print(f"[select] pruned weights of {len(candidates) - 1} unselected checkpoint(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
