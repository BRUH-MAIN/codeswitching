"""Demo of the trained Hindi-English code-switching ASR models.

    python -m csasr.demo models                       what can be demoed, with test scores
    python -m csasr.demo showcase                     instant: stored test transcripts, no GPU
    python -m csasr.demo samples  --models story      live: transcribe MUCS test excerpts
    python -m csasr.demo transcribe talk.wav --models medium-qlora zeroshot-medium

Models are named by short keys (`medium-qlora`, `large-sarvam30`, `zeroshot-large`)
or presets (`quick`, `methods`, `asr`, `story`, `all`, `best`); see `models`.
Each fine-tuned model loads the checkpoint the ablation selected on dev.

`samples` and `transcribe` run the models for real. Large-v2 wants a GPU: submit
`sbatch scripts/demo.sh` for that. Everything runs on CPU too (whisper-medium takes
about a minute per 25 s excerpt); the login node's GPUs are never used outside a
SLURM allocation. Weights come from the local cache; set HF_HUB_OFFLINE=0 to
download anything missing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .registry import PRESETS, REPO

DEFAULT_OUT = REPO / "results" / "demo"


def _setup_env(device: str | None) -> None:
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    hf = REPO / "csasr-work" / "hf"
    if hf.is_dir():
        os.environ.setdefault("HF_HOME", str(hf))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""   # never touch another user's GPU


def _fmt(v, nd=1):
    return "–" if v is None else f"{v:.{nd}f}"


# ---------------------------------------------------------------------------
def cmd_models(args) -> int:
    from .registry import all_models

    models = all_models()
    print(f"{'key':<18} {'model':<72} {'step':>4} {'test MER':>9} {'CBA-HE':>7}  on disk")
    for m in models.values():
        print(f"{m.key:<18} {m.label:<72} {_fmt(m.step, 0):>4} {_fmt(m.mer):>9} "
              f"{_fmt(m.cba_he):>7}  {'yes' if m.available else 'NO'}")
    print("\npresets: " + "; ".join(f"{k} = {', '.join(v)}" for k, v in PRESETS.items())
          + "; all (every model on disk); best (lowest test MER)")
    print("MER: lower is better. CBA-HE: Hindi→English switch accuracy, higher is better.")
    return 0


# ---------------------------------------------------------------------------
def _transcribe_all(models, clips, device, threads, beams):
    """Load each model once, transcribe every clip. One failing model does not stop the rest."""
    from .engines import load_engine

    results: dict[str, dict] = {}
    for m in models:
        print(f"[demo] {m.key}: loading ({m.path if not m.zero_shot else 'public checkpoint'})"
              f" on {device} -- first load of a fine-tuned model converts it, minutes",
              file=sys.stderr, flush=True)
        try:
            t0 = time.monotonic()
            eng = load_engine(m, device, threads=threads, beams=beams)
            print(f"[demo] {m.key}: loaded in {time.monotonic() - t0:.0f}s",
                  file=sys.stderr, flush=True)
        except Exception as e:  # noqa: BLE001 - report and carry on with the other models
            print(f"[demo] {m.key}: FAILED to load: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        results[m.key] = {}
        try:
            for i, c in enumerate(clips, 1):
                r = eng.transcribe(c.audio)
                results[m.key][c.clip_id] = r
                print(f"[demo] {m.key}: clip {i}/{len(clips)} in {r.seconds:.1f}s",
                      file=sys.stderr, flush=True)
        finally:
            eng.close()
    return results


def _show(clips, models, results, color):
    from .render import analyse, ansi, score_mer

    for n, c in enumerate(clips, 1):
        dur = f" ({c.seconds:.0f} s)" if c.audio is not None else ""
        print(f"\n=== Clip {n}: {c.source}{dur}")
        if c.ref:
            s = analyse(c.ref)
            print(f"  [Reference]\n    {ansi(c.ref, color)}\n    "
                  f"{s.english_pct:.0f}% English words, {s.switches} switches")
        for m in models:
            r = results.get(m.key, {}).get(c.clip_id)
            if r is None:
                continue
            s = analyse(r.text)
            bits = [f"{s.english_pct:.0f}% English words", f"{s.switches} switches"]
            if c.ref and c.audio is not None:
                bits.insert(0, f"MER {score_mer([c.ref], [r.text]):.1f}")
            print(f"  [{m.label}]\n    {ansi(r.text, color) or '(no speech detected)'}\n    "
                  + ", ".join(bits))


def _finish(args, title, subtitle, clips, models, results, notes, *, score=True) -> int:
    from .render import build_html, use_color

    if not any(results.values()):
        print("[demo] no model produced output", file=sys.stderr)
        return 1
    done = [m for m in models if m.key in results]
    _show(clips, done, results, use_color())
    out = Path(args.html) if args.html else DEFAULT_OUT / f"{args.command}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(title, subtitle, clips, done, results, notes=notes, score=score),
                   encoding="utf-8")
    print(f"\n[demo] report with audio -> {out}" if any(c.audio is not None for c in clips)
          else f"\n[demo] report -> {out}")
    return 0


def _live(args, clips, models, title, subtitle, notes) -> int:
    from .engines import pick_device

    device = pick_device(args.device)
    _setup_env(device)
    results = _transcribe_all(models, clips, device, args.threads, args.beams)
    return _finish(args, title, subtitle, clips, models, results, notes)


# ---------------------------------------------------------------------------
def cmd_samples(args) -> int:
    from .engines import pick_device
    from .registry import resolve

    _setup_env(pick_device(args.device))     # hide GPUs before torch/datasets load
    models = resolve(args.models)
    from .samples import load_test_excerpts

    clips = load_test_excerpts(args.n, args.seconds, args.seed, args.recordings,
                               args.min_english)
    return _live(
        args, clips, models, "Code-switching ASR: live transcription of MUCS test excerpts",
        f"{len(clips)} excerpts of about {args.seconds:.0f} s from real spoken-tutorial "
        f"recordings the models never saw (seed {args.seed}).",
        "Models are decoded as in the evaluation: language auto-detected, beam "
        f"{args.beams}. MER on a few excerpts is noisy; the 'whole set' column is the "
        "30-recording test score.")


def cmd_transcribe(args) -> int:
    from .engines import load_audio, pick_device
    from .registry import resolve
    from .samples import Clip

    _setup_env(pick_device(args.device))
    models = resolve(args.models)
    if args.ref and len(args.audio) != 1:
        raise SystemExit("--ref takes one reference text, so give exactly one audio file")
    clips = []
    for p in args.audio:
        if not Path(p).is_file():
            raise SystemExit(f"no such audio file: {p}")
        ref = Path(args.ref).read_text(encoding="utf-8").strip() if args.ref else None
        clips.append(Clip(clip_id=Path(p).name, audio=load_audio(p), ref=ref, source=str(p)))
    return _live(args, clips, models, "Code-switching ASR: your audio",
                 f"{len(clips)} file(s) transcribed by {len(models)} model(s).", "")


def cmd_showcase(args) -> int:
    """Stored test transcripts: instant, no GPU, no audio. Opening words of each recording."""
    import random

    from ..eval.grouping import recording_id, segment_index
    from .engines import Result
    from .registry import resolve
    from .samples import Clip

    res_dir = REPO / "results" / "ablation"
    refs_f = res_dir / "refs_test.jsonl"
    if not refs_f.exists():
        raise SystemExit(f"{refs_f} not found; the showcase needs the stored ablation results")
    models = []
    for m in resolve(args.models, require_available=False):
        if (res_dir / "hyps" / f"{m.run}.jsonl").exists():
            models.append(m)
        else:
            print(f"[demo] no stored transcripts for {m.key}; skipped", file=sys.stderr)
    if not models:
        raise SystemExit("none of the requested models has stored transcripts")

    rows = [json.loads(line) for line in refs_f.read_text(encoding="utf-8").splitlines() if line]
    rows.sort(key=lambda r: (recording_id(r["utt_id"]), segment_index(r["utt_id"])))
    refs: dict[str, str] = {}
    for r in rows:
        refs[recording_id(r["utt_id"])] = (refs.get(recording_id(r["utt_id"]), "") + " "
                                           + r["text"].strip()).strip()
    hyps = {m.key: {h["utt_id"]: h["hyp"] for h in
                    map(json.loads, (res_dir / "hyps" / f"{m.run}.jsonl")
                        .read_text(encoding="utf-8").splitlines()) if h}
            for m in models}

    names = sorted(refs)
    picked = random.Random(args.seed).sample(names, min(args.n, len(names)))

    def head(text):
        return " ".join(text.split()[: args.words])

    clips = [Clip(clip_id=r, audio=None, ref=head(refs[r]),
                  source=f"MUCS test, recording {r}, first {args.words} words (text only)")
             for r in picked]
    results = {m.key: {r: Result(head(hyps[m.key][r]), 0.0) for r in picked if r in hyps[m.key]}
               for m in models}
    args.command = "showcase"
    return _finish(
        args, "Code-switching ASR: stored test transcripts",
        f"The opening of {len(clips)} test recordings, as transcribed during the ablation.",
        clips, models, results,
        "Transcripts are from the stored whole-recording decodes; reference and model "
        "text are cut at the same word count, so the endings can drift apart and no "
        "per-excerpt MER is computed. Use the whole-set test MER column for accuracy.",
        score=False)


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m csasr.demo", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p, default_models):
        p.add_argument("--models", nargs="+", default=default_models, metavar="MODEL",
                       help=f"model keys or presets (default: {' '.join(default_models)})")
        p.add_argument("--html", default=None, help=f"report path (default {DEFAULT_OUT}/<command>.html)")

    def live(p):
        p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                       help="auto = CUDA only inside a SLURM job, else CPU")
        p.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4),
                       help="CPU threads")
        p.add_argument("--beams", type=int, default=5, help="Whisper beam size (evaluation: 5)")

    sub.add_parser("models", help="list the demo models and their test scores").set_defaults(
        func=cmd_models)

    p = sub.add_parser("showcase", help="stored test transcripts, instant, no GPU")
    common(p, ["story"])
    p.add_argument("--n", type=int, default=4, help="recordings to show")
    p.add_argument("--words", type=int, default=60, help="words of each recording to show")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_showcase)

    p = sub.add_parser("samples", help="live transcription of MUCS test excerpts")
    common(p, ["quick"])
    live(p)
    p.add_argument("--n", type=int, default=3, help="excerpts")
    p.add_argument("--seconds", type=float, default=25.0, help="length of each excerpt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--recordings", nargs="+", default=None, help="restrict to these recording ids")
    p.add_argument("--min-english", type=float, default=0.0, metavar="PCT",
                   help="only excerpts whose reference has at least PCT%% English words "
                        "(shows code-switching; default 0 = a plain random draw)")
    p.set_defaults(func=cmd_samples)

    p = sub.add_parser("transcribe", help="transcribe your own audio files")
    common(p, ["quick"])
    live(p)
    p.add_argument("audio", nargs="+", help="audio file(s): wav, flac, mp3, ...")
    p.add_argument("--ref", default=None, help="reference text file, to score MER (one audio file)")
    p.set_defaults(func=cmd_transcribe)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
