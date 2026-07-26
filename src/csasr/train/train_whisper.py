"""Stage 3: fine-tune Whisper on synthetic code-mixed speech (M6 / M7 / M8).

Keeps the paper's optimizer recipe: full fine-tune, AdamW, peak lr 2e-5,
effective batch 64, mixed precision, max 5k steps. No LoRA and no quantization,
which also means `predict_with_generate` works and we can select checkpoints on
a real MER rather than on validation loss.

Runs on two very different machines and *detects* which, rather than assuming:

* **Kaggle 2x T4** (Turing, sm75). No bf16, no TF32, no FlashAttention-2. So:
  fp16 autocast + GradScaler, gradient checkpointing on, whisper-small (244M,
  ~4 GB of optimizer state).
* **A100** (Ampere, sm80). bf16 and TF32 are both available and both taken --
  bf16 needs no GradScaler and cannot overflow the way fp16 does. Enough memory
  to full-fine-tune **whisper-large-v2** (1.54B, ~24.7 GB of state), which is
  the paper's own model, so on this path deviation **D2 does not apply** and
  absolute MER is directly comparable to the paper's Table 2.

Precision, TF32 and gradient checkpointing are each chosen from the detected
device, printed so the choice is auditable, and each overridable by flag.

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


#: fp32 params + fp32 grads + AdamW's two fp32 moments = 16 bytes per parameter.
#: Mixed precision does not shrink this: autocast keeps fp32 master weights.
BYTES_PER_PARAM = 16

#: Above this fraction of VRAM spent on optimizer state, activations stop fitting
#: and gradient checkpointing is worth its ~30-40% throughput cost.
#: large-v2 (24.7 GB) -> 0.62 on a 40 GB A100 (on), 0.31 on an 80 GB (off).
#: small (3.9 GB) -> 0.24 on a 16 GB T4 (off... but see _plan_hardware).
CHECKPOINT_VRAM_FRACTION = 0.35


def _plan_hardware(args, torch, n_params: int) -> dict:
    """Choose precision / TF32 / gradient checkpointing from the actual device.

    Returns a dict of Seq2SeqTrainingArguments kwargs. Every decision is printed,
    because a silent wrong guess here shows up as a NaN loss or an OOM 250 steps
    in, and neither points back at this function.
    """
    if not torch.cuda.is_available():
        print("[train] WARNING: no CUDA device; this will be unusably slow")
        return {"fp16": False, "bf16": False, "tf32": False,
                "gradient_checkpointing": args.gradient_checkpointing == "on"}

    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / 1e9
    cap = (props.major, props.minor)
    ampere = cap >= (8, 0)
    state = n_params * BYTES_PER_PARAM / 1e9

    # bf16 needs Ampere. torch reports capability directly; is_bf16_supported()
    # also returns True for some emulated paths, so require sm80 as well.
    if args.precision == "auto":
        bf16 = ampere and torch.cuda.is_bf16_supported()
        fp16 = not bf16
    else:
        bf16 = args.precision == "bf16"
        fp16 = args.precision == "fp16"
        if bf16 and not (ampere and torch.cuda.is_bf16_supported()):
            raise SystemExit(
                f"--precision bf16 requested but {props.name} is sm{cap[0]}{cap[1]}, "
                "which has no bf16. Turing (T4) must use fp16."
            )

    # TF32 is an Ampere matmul path; harmless to request elsewhere but does nothing.
    tf32 = ampere if args.tf32 == "auto" else args.tf32 == "on"
    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.gradient_checkpointing == "auto":
        ckpt = state > CHECKPOINT_VRAM_FRACTION * vram
    else:
        ckpt = args.gradient_checkpointing == "on"

    print(f"[train] device   {props.name}  sm{cap[0]}{cap[1]}  {vram:.0f} GB")
    print(f"[train] params   {n_params/1e6:,.0f} M  -> ~{state:.1f} GB of "
          f"fp32 param+grad+Adam state ({state/vram:.0%} of VRAM)")
    print(f"[train] precision {'bf16' if bf16 else 'fp16' if fp16 else 'fp32'}"
          f"   tf32 {'on' if tf32 else 'off'}"
          f"   grad-checkpointing {'on' if ckpt else 'off'}")
    if fp16 and ampere:
        print("[train] NOTE: fp16 on an Ampere card -- bf16 is strictly safer here "
              "(no GradScaler, no overflow). Drop --precision to auto-select it.")

    out = {"fp16": fp16, "bf16": bf16, "tf32": tf32, "gradient_checkpointing": ckpt}
    if ckpt:
        # Default reentrant checkpointing warns loudly and interacts badly with
        # frozen/unused params on recent torch. We full-fine-tune everything, so
        # the non-reentrant path is both quieter and correct here.
        out["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return out


def _apply_config_defaults(ap, argv):
    """Let `--config foo.yaml` supply defaults that explicit CLI flags override.

    The configs/ yaml files used to be decorative -- nothing read them, so they
    looked functional while silently having no effect. Loading them as argparse
    *defaults* (not as a post-parse overwrite) is what preserves the usual
    precedence: CLI > yaml > code default.
    """
    known, _ = ap.parse_known_args(argv)
    if not known.config:
        return
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(OmegaConf.load(known.config), resolve=True)
    valid = {a.dest for a in ap._actions}
    unknown = set(cfg) - valid
    if unknown:
        raise SystemExit(f"{known.config}: unknown keys {sorted(unknown)}")
    # argparse applies type= only to strings from the command line, so a yaml
    # `lr: 2.0e-5` must already be a float. It is -- yaml parses it natively.
    ap.set_defaults(**{k: v for k, v in cfg.items() if v is not None})
    print(f"[train] defaults from {known.config}")


def _prepare(batch, feature_extractor, tokenizer):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


def _build_parser() -> argparse.ArgumentParser:
    """Separate from main() so tests can assert the real configs/ files parse
    against the real parser, rather than against a hand-copied stand-in that
    drifts."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None,
                    help="yaml of defaults, e.g. configs/train_m7_large.yaml; "
                         "explicit flags still win")
    ap.add_argument("--model", default="openai/whisper-small")
    # NOT required=True: argparse's required check ignores set_defaults, so a
    # required --out would make `out:` in a config file silently useless -- the
    # same decorative-config trap this --config support exists to remove.
    # Validated by hand after parsing instead.
    ap.add_argument("--out", type=Path, default=None)

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
    # Guards the paper's recipe against silent drift: it is easy to raise
    # --batch-size for a bigger GPU and forget to lower --grad-accum, which
    # quietly changes the effective batch and therefore the learning dynamics.
    # Pass --effective-batch 0 to opt out deliberately.
    ap.add_argument("--effective-batch", type=int, default=64,
                    help="asserted == batch-size * grad-accum; 0 disables the check")
    # Eval runs predict_with_generate, so its memory profile is nothing like
    # training's: 225 generated tokens of KV cache per sequence. Reusing the
    # train batch size OOMs large-v2 at the FIRST eval, 250 steps in.
    ap.add_argument("--eval-batch-size", type=int, default=8)
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

    # ---- hardware: detected by default, never assumed -------------------
    ap.add_argument("--precision", choices=("auto", "fp16", "bf16"), default="auto",
                    help="auto = bf16 on Ampere+, fp16 on Turing")
    ap.add_argument("--tf32", choices=("auto", "on", "off"), default="auto",
                    help="auto = on for Ampere+ (no effect on Turing)")
    ap.add_argument("--gradient-checkpointing", choices=("auto", "on", "off"), default="auto",
                    help="auto = on when optimizer state exceeds ~35%% of VRAM")
    ap.add_argument("--attn-impl", choices=("auto", "sdpa", "eager"), default="auto",
                    help="auto = sdpa on CUDA, falling back to eager if unsupported")
    ap.add_argument("--optim", default="adamw_torch",
                    help="adamw_bnb_8bit cuts Adam state 12.3 GB -> ~3 GB on a 40 GB card")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = _build_parser()
    _apply_config_defaults(ap, argv)
    args = ap.parse_args(argv)

    if args.out is None:
        raise SystemExit("need --out, or an `out:` key in the --config yaml")
    args.out = Path(args.out)  # a yaml default arrives as str, bypassing type=

    if args.effective_batch and args.batch_size * args.grad_accum != args.effective_batch:
        raise SystemExit(
            f"batch-size {args.batch_size} x grad-accum {args.grad_accum} = "
            f"{args.batch_size * args.grad_accum}, but the paper's effective batch is "
            f"{args.effective_batch}. Adjust one of them, or pass --effective-batch 0 "
            "to override deliberately."
        )

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

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    processor = WhisperProcessor.from_pretrained(args.model, language=args.language, task="transcribe")
    tokenizer = processor.tokenizer
    tokenizer.set_prefix_tokens(language=args.language, task="transcribe")

    # sdpa is a real speedup on Ampere and supported for Whisper since
    # transformers 4.36, but the kwarg name and support matrix have both moved
    # across versions -- so try it and fall back rather than pinning a version.
    attn = args.attn_impl
    if attn == "auto":
        attn = "sdpa" if torch.cuda.is_available() else "eager"
    try:
        model = WhisperForConditionalGeneration.from_pretrained(
            args.model, attn_implementation=attn
        )
    except (TypeError, ValueError) as e:
        print(f"[train] attn_implementation={attn!r} rejected ({e}); using the default")
        model = WhisperForConditionalGeneration.from_pretrained(args.model)
        attn = "default"
    print(f"[train] attention {attn}")

    hw = _plan_hardware(args, torch, sum(p.numel() for p in model.parameters()))

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
    print(f"[train] effective batch {args.batch_size} x {args.grad_accum} = "
          f"{args.batch_size * args.grad_accum}   lr {args.lr:g}   "
          f"eval batch {args.eval_batch_size}")

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
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        optim=args.optim,
        **hw,  # fp16 / bf16 / tf32 / gradient_checkpointing, from _plan_hardware
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

    trainer = Seq2SeqTrainer(
        args=targs,
        model=model,
        train_dataset=train,
        eval_dataset=dev,
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    result = trainer.train()
    trainer.save_model(str(args.out))
    processor.save_pretrained(str(args.out))

    print(f"[train] finished at step {result.global_step:,} "
          f"(max_steps={args.max_steps}; early stopping may have fired)")
    print(f"[train] best {targs.metric_for_best_model} = "
          f"{trainer.state.best_metric}")
    print(f"[train] model -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
