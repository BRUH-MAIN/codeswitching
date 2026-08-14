"""Stage 3: fine-tune Whisper on synthetic code-mixed speech (M6 / M7 / M8).

Keeps the paper's optimizer recipe: full fine-tune, AdamW, peak lr 2e-5,
effective batch 64, mixed precision, max 5k steps. whisper-small (244M) fits a
16 GB T4 with ~4 GB of optimizer state, so no LoRA or quantization is needed --
which also means `predict_with_generate` works and we can select checkpoints on
a real MER rather than on validation loss.

T4 is Turing: no bf16, no FlashAttention-2. fp16 autocast + GradScaler (which is
what `fp16=True` turns on inside Trainer).

Language prompt is `<|hi|>` for all of M6/M7/M8, per Table 2. Dev decoding uses
`language=None` so checkpoint selection tracks the same auto-detect condition the
test set is scored under.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# MUST precede any transformers/tokenizers import. `datasets.map(num_proc=N)`
# forks, and the parent has already used the Rust fast tokenizer (via
# set_prefix_tokens) and touched the CUDA driver. A forked child that inherits
# the tokenizers thread pool deadlocks: the map bar sits at 0/N forever with
# zero CPU, which reads like "slow" but never finishes. Setting this at import
# time is the only placement that is guaranteed early enough.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__all__ = ["main"]


def _report_to(choice: str) -> list[str]:
    """`TensorBoardCallback` raises at construction if tensorboard is absent, so
    never hardcode it. Kaggle has it; a bare venv does not."""
    if choice != "auto":
        return [] if choice in ("none", "") else [choice]
    import importlib.util

    return ["tensorboard"] if importlib.util.find_spec("tensorboard") else []


def _prepare(batch, feature_extractor, tokenizer):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--out", type=Path, required=True)

    # data: HF hub (normal) or local manifests (smoke tests)
    ap.add_argument("--train-hf", default=None, help="e.g. RohanRamesh/hi-en-synth-cs")
    ap.add_argument("--train-config", default="synth_t2")
    ap.add_argument("--train-manifest", type=Path, default=None)
    ap.add_argument("--subset-ids", type=Path, default=None, help="t1_ids.json -> M6")
    ap.add_argument("--extra-hf", nargs="*", default=[], help="repo:config for CV mono -> M8")
    ap.add_argument("--extra-manifest", type=Path, nargs="*", default=[])
    ap.add_argument("--dev-hf", default=None)
    ap.add_argument("--dev-config", default="dev")
    ap.add_argument("--dev-manifest", type=Path, default=None)
    # Never pass a token in argv: it lands in every traceback and in `ps` output.
    ap.add_argument("--hf-token", default=None, help="prefer the HF_TOKEN env var")

    # paper's recipe
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=4)  # 16 x 4 = 64
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--eval-steps", type=int, default=250)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--language", default="hi", help="training prompt language token")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dataset-fraction", type=float, default=1.0)
    # Default 1, deliberately. Featurization is numpy log-mel + FLAC decode, so
    # 2 workers saves only a few minutes across the whole corpus -- not worth
    # reintroducing the fork deadlock that TOKENIZERS_PARALLELISM above guards
    # against. Raise it only if you have confirmed the map actually progresses.
    ap.add_argument("--num-proc", type=int, default=1, help="datasets.map workers")
    ap.add_argument("--dataloader-workers", type=int, default=2)
    ap.add_argument("--report-to", default="auto", help="'auto' uses tensorboard if installed")

    # Resumability: the Trainer's full checkpoint (model + optimizer + scheduler
    # + RNG + trainer_state.json) is pushed to this Hub repo every --eval-steps
    # and cleared once the run finishes normally. A session killed mid-run
    # leaves its last checkpoint in place, so re-running the same command
    # resumes instead of starting over. Omit to disable (local-only, as before).
    ap.add_argument("--hub-checkpoint-repo", default=None,
                    help="e.g. RohanRamesh/csasr-train-checkpoints")
    ap.add_argument("--run-name", default=None,
                    help="subfolder key within --hub-checkpoint-repo; defaults to --out's basename")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore any existing Hub checkpoint and start this run from scratch")
    args = ap.parse_args(argv)

    import torch
    from transformers import (
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    from ..data.loaders import apply_subset, concat, load_hub_dataset, load_manifest_dataset
    from ..eval.cba import cba
    from ..eval.mer import mer
    from .collator import WhisperCollator
    from .hub_checkpoint import build_hub_checkpoint_callback, clear_checkpoint, download_checkpoint

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    run_name = args.run_name or args.out.name
    hub_resume_dir = str(args.out) + "-hub-resume"

    resume_checkpoint = None
    if args.hub_checkpoint_repo and not args.no_resume:
        resume_checkpoint = download_checkpoint(
            args.hub_checkpoint_repo, run_name, token=hf_token,
            local_dir=hub_resume_dir, kind="last",
        )
        if resume_checkpoint:
            print(f"[train] resuming {run_name!r} from Hub checkpoint: {resume_checkpoint}")
        else:
            print(f"[train] no existing Hub checkpoint for {run_name!r}; starting fresh")

    if not torch.cuda.is_available():
        print("[train] WARNING: no CUDA device; this will be unusably slow")

    processor = WhisperProcessor.from_pretrained(args.model, language=args.language, task="transcribe")
    tokenizer = processor.tokenizer
    tokenizer.set_prefix_tokens(language=args.language, task="transcribe")

    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"

    # ---- data ----------------------------------------------------------
    def _load(hf, cfg, manifest, split="train"):
        if hf:
            return load_hub_dataset(hf, cfg, split=split, token=hf_token)
        if manifest:
            return load_manifest_dataset(manifest)
        return None

    train = _load(args.train_hf, args.train_config, args.train_manifest)
    if train is None:
        raise SystemExit("need --train-hf or --train-manifest")
    if args.subset_ids:
        train = apply_subset(train, args.subset_ids)

    extras = []
    for spec in args.extra_hf:
        repo, _, cfg = spec.partition(":")
        extras.append(load_hub_dataset(repo, cfg or None, token=hf_token))
    extras += [load_manifest_dataset(p) for p in args.extra_manifest]
    if extras:
        train = concat([train, *extras])

    dev = _load(args.dev_hf, args.dev_config, args.dev_manifest, split="train")
    if dev is None:
        raise SystemExit("need --dev-hf or --dev-manifest")

    if args.dataset_fraction < 1.0:
        n = max(1, int(len(train) * args.dataset_fraction))
        train = train.shuffle(seed=args.seed).select(range(n))
        dev = dev.select(range(min(len(dev), max(8, n // 10))))

    print(f"[train] train={len(train):,} utts   dev={len(dev):,} utts")

    fn_kwargs = {"feature_extractor": processor.feature_extractor, "tokenizer": tokenizer}
    train = train.map(_prepare, fn_kwargs=fn_kwargs, remove_columns=train.column_names,
                      num_proc=args.num_proc, desc="featurize train")
    dev = dev.map(_prepare, fn_kwargs=fn_kwargs, remove_columns=dev.column_names,
                  num_proc=args.num_proc, desc="featurize dev")

    collator = WhisperCollator(processor, model.config.decoder_start_token_id)

    # ---- metric --------------------------------------------------------
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        hyps = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        refs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        c = cba(refs, hyps)
        return {"mer": mer(refs, hyps), "cba_he": c.he, "cba_eh": c.eh}

    targs = Seq2SeqTrainingArguments(
        output_dir=str(args.out),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),  # Turing: fp16 + GradScaler, never bf16
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=2,
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="mer",
        greater_is_better=False,
        report_to=_report_to(args.report_to),
        seed=args.seed,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=args.dataloader_workers,
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=args.patience)]
    if args.hub_checkpoint_repo:
        # NOTE: EarlyStoppingCallback's own non-improvement counter is plain
        # instance state, not part of `trainer_state.json` -- resuming restores
        # `best_metric` correctly (so a resumed run can't regress its notion of
        # "best"), but the count of consecutive bad evals restarts at 0. Net
        # effect is a small conservative bias (occasionally a few evals later
        # to stop than an uninterrupted run), never early stopping too soon.
        callbacks.append(
            build_hub_checkpoint_callback(args.hub_checkpoint_repo, run_name, token=hf_token)
        )

    trainer = Seq2SeqTrainer(
        args=targs,
        model=model,
        train_dataset=train,
        eval_dataset=dev,
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=callbacks,
    )

    result = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
    )

    # `load_best_model_at_end` already tried to reload `state.best_model_checkpoint`
    # -- a path on THIS machine's disk. That's a no-op in the common case (best ==
    # last, or nothing crashed), but if a previous session died right after saving
    # a checkpoint that was later beaten, then got resumed here, the true best
    # checkpoint's weights may only exist on the Hub, not on this fresh disk.
    # Recover them explicitly rather than silently shipping a worse final model.
    best_path = trainer.state.best_model_checkpoint
    if args.hub_checkpoint_repo and best_path and not Path(best_path).exists():
        print(f"[train] best checkpoint {best_path} is not on local disk "
              f"(training was resumed on a different machine); pulling it from the Hub")
        best_dir = download_checkpoint(
            args.hub_checkpoint_repo, run_name, token=hf_token,
            local_dir=hub_resume_dir, kind="best",
        )
        if best_dir:
            trainer.model = WhisperForConditionalGeneration.from_pretrained(str(best_dir))
            print(f"[train] loaded best checkpoint from the Hub -> {best_dir}")
        else:
            print("[train] WARNING: could not recover the best checkpoint from the Hub; "
                  "saving the model as it currently stands (the last resumed step) instead")

    trainer.save_model(str(args.out))
    processor.save_pretrained(str(args.out))

    # Only clear the Hub's resumable state once the final model is safely on
    # local disk. Doing this any earlier (e.g. from an on_train_end callback,
    # which fires before this point) would delete the one thing a crash during
    # save_model() could still be recovered from.
    if args.hub_checkpoint_repo:
        clear_checkpoint(args.hub_checkpoint_repo, run_name, token=hf_token)

    print(f"[train] finished at step {result.global_step:,} "
          f"(max_steps={args.max_steps}; early stopping may have fired)")
    print(f"[train] best {targs.metric_for_best_model} = "
          f"{trainer.state.best_metric}")
    print(f"[train] model -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
