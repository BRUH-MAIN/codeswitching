"""Stage 3: fine-tune Whisper on synthetic code-mixed speech (M6 / M7 / M8).

Keeps the paper's optimizer recipe by default: full fine-tune, AdamW, peak lr
2e-5, effective batch 64, mixed precision, max 5k steps.

`--ft-method lora|qlora` is the ablation path: the base model is frozen and only
low-rank adapters train. LoRA and QLoRA share one adapter config and differ only
in the precision the frozen base is held at (bf16/fp32 vs 4-bit NF4), so the
comparison isolates quantization. Ablation runs also pass `--generate-eval off`:
checkpoints are saved at fixed steps and selected afterwards by
`csasr.eval.select`, which decodes dev with the same recording-level decoder
the test set is scored under.

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

    peft = getattr(args, "ft_method", "full") != "full"
    if peft:
        # The frozen base carries no grads and no Adam moments, so the 16
        # bytes/param figure does not apply: only the weights themselves.
        state = n_params * 4 / 1e9
    if args.gradient_checkpointing == "auto":
        ckpt = (not peft) and state > CHECKPOINT_VRAM_FRACTION * vram
    else:
        ckpt = args.gradient_checkpointing == "on"

    print(f"[train] device   {props.name}  sm{cap[0]}{cap[1]}  {vram:.0f} GB")
    print(f"[train] params   {n_params/1e6:,.0f} M  -> ~{state:.1f} GB of "
          f"{'frozen fp32 weights' if peft else 'fp32 param+grad+Adam state'} "
          f"({state/vram:.0%} of VRAM)")
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
    # M8 mixes ~18k code-switched utterances with ~17k Common Voice monolingual
    # ones -- roughly 1:1, so on average half of every effective batch carries no
    # code-switch signal at all, vs. M7's 100%. Repeating the code-switched set
    # keeps every training example the model sees (all of --extra-hf still
    # appears once per epoch) while raising its share of gradient updates,
    # without touching lr/batch-size/steps -- the paper states those as fixed
    # across every model, so this is a data-composition knob, not a recipe
    # deviation.
    ap.add_argument("--oversample-train", type=int, default=1,
                    help="repeat --train-hf/-manifest this many times before "
                         "concatenating --extra-hf, so it isn't diluted 1:1 by "
                         "monolingual data (M8)")
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
    ap.add_argument("--select-by", choices=("mer", "score"), default="mer",
                    help="checkpoint-selection metric. 'score' = mer/(cba_total+1), "
                         "which penalizes a collapsed CBA that plain MER selection "
                         "would not catch (M8)")
    # ---- ablation: fine-tuning method -----------------------------------
    ap.add_argument("--ft-method", choices=("full", "lora", "qlora"), default="full",
                    help="full = the paper's recipe; lora = frozen base + adapters; "
                         "qlora = the same adapters on a 4-bit NF4 base")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-targets", nargs="+", default=["q_proj", "v_proj"],
                    help="module-name suffixes to adapt (encoder and decoder both)")
    # ---- ablation: fixed-step checkpoints, selected post hoc -------------
    ap.add_argument("--save-steps", type=int, default=None,
                    help="checkpoint interval; defaults to --eval-steps")
    ap.add_argument("--generate-eval", choices=("on", "off"), default="on",
                    help="off = dev LOSS only during training (no generate, no early "
                         "stopping, no best-model reload); checkpoints are then "
                         "selected by csasr.eval.select with the test decoder")
    # Twenty steps on 2% of the data already moved test MER and CBA well past
    # zero-shot in the smoke run, while the earlier 1000+ step runs ended up
    # below it -- so the interesting part of the curve is BEFORE the first
    # regular checkpoint, and has to be saved to be selectable.
    ap.add_argument("--save-at", type=int, nargs="*", default=None,
                    help="extra checkpoint steps on top of --save-steps; default "
                         "50 100 when --generate-eval off, none otherwise")
    ap.add_argument("--dev-max-utts", type=int, default=0,
                    help="seeded subsample of dev for the in-training eval; 0 = all")
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
    # Disk, not VRAM. See save_only_model below: a large-v2 checkpoint is 6.2 GB
    # of weights, or 18.5 GB if the optimizer state rides along.
    ap.add_argument("--save-total-limit", type=int, default=2)
    ap.add_argument("--save-optimizer", action="store_true",
                    help="keep AdamW state in checkpoints so training can resume; "
                         "triples checkpoint size (18.5 GB each for large-v2)")
    # Resuming is the default whenever --out already has a checkpoint (a SLURM
    # walltime cap can cut a run off mid-training, same story as a100/hf_sync.py
    # on the policyswitching side) -- --from-scratch is the explicit opt-out.
    ap.add_argument("--from-scratch", action="store_true",
                    help="ignore any existing checkpoint under --out and start at step 0 "
                         "(default: resume from the latest checkpoint there, if any)")
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
    from transformers.trainer_utils import get_last_checkpoint

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
    # low_cpu_mem_usage loads shard-by-shard into the final placement instead of
    # materialising a second full fp32 copy in host RAM. large-v2 is 6.2 GB, so
    # on a 16 GB SLURM allocation the naive path is a real OOM risk.
    load_kw = {"low_cpu_mem_usage": True}
    if args.ft_method == "qlora":
        from transformers import BitsAndBytesConfig

        # NF4 + double quantization, bf16 compute: the QLoRA paper's setting.
        # Only nn.Linear is quantized; the two conv layers, LayerNorms and
        # embeddings stay in floating point.
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
        load_kw["device_map"] = {"": 0}
    try:
        model = WhisperForConditionalGeneration.from_pretrained(
            args.model, attn_implementation=attn, **load_kw
        )
    except (TypeError, ValueError) as e:
        print(f"[train] attn_implementation={attn!r} rejected ({e}); using the default")
        model = WhisperForConditionalGeneration.from_pretrained(args.model, **load_kw)
        attn = "default"
    print(f"[train] attention {attn}")

    # num_parameters(), not a sum of numel(): a 4-bit layer stores its weights
    # packed, so numel() reports about half the real count and QLoRA would look
    # like it trains twice the share of the model that LoRA does.
    n_total = model.num_parameters()
    hw = _plan_hardware(args, torch, n_total)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None
    # NOT args.language. Training loss uses tokenizer.set_prefix_tokens(language=
    # args.language) above, which forces <|hi|> into the labels -- unaffected by
    # this. But predict_with_generate (compute_metrics below, i.e. every eval and
    # therefore load_best_model_at_end / EarlyStoppingCallback) reads this
    # attribute at generate() time, and the paper's test protocol -- and this
    # repo's own decode.py -- score under auto-detect (`language=None`), not
    # forced Hindi. Leaving this at "hi" silently picks "best" checkpoints under
    # a decoding condition that never matches how the model is actually scored.
    model.generation_config.language = None
    model.generation_config.task = "transcribe"

    if args.ft_method != "full":
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if args.ft_method == "qlora":
            # Freezes the base and casts the non-quantized params to fp32.
            # Gradient checkpointing is wired by the Trainer from `hw`, not here.
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=list(args.lora_targets), bias="none",
        ))
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] ft-method {args.ft_method}: {n_trainable/1e6:,.2f} M trainable "
          f"of {n_total/1e6:,.0f} M ({n_trainable/n_total:.2%})")

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

    if args.oversample_train > 1:
        before = len(train)
        train = concat([train] * args.oversample_train)
        print(f"[train] oversampled primary train set {args.oversample_train}x: "
              f"{before:,} -> {len(train):,} utts")

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

    if args.dev_max_utts and len(dev) > args.dev_max_utts:
        dev = dev.shuffle(seed=args.seed).select(range(args.dev_max_utts))

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
        m = mer(refs, hyps)
        return {
            "mer": m,
            "cba_he": c.he,
            "cba_eh": c.eh,
            "cba_total": c.total,
            # MER-only selection can lock in a checkpoint whose code-switch
            # accuracy has collapsed while MER still looks fine (confirmed for
            # M8: CBA-HE stuck under ~14% for the entire run, from the very
            # first eval, while MER kept improving). Dividing by cba_total+1
            # makes a near-zero-CBA checkpoint score much worse regardless of
            # MER, without CBA needing to already dominate a raw weighted sum
            # across two differently-scaled metrics. Only used when
            # --select-by score (see train_m8_large.yaml); M6/M7 keep the
            # plain "mer" selection that already produces the right ordering.
            "score": m / (c.total + 1.0),
        }

    # tqdm's carriage-return progress bar renders fine on a real terminal but
    # turns into unreadable repeated-line spam in a plain SLURM .out file, which
    # isn't a tty. logging_steps=25 below already gives periodic textual
    # progress; disable_tqdm just stops it from being drowned out by bar noise.
    disable_tqdm = not sys.stdout.isatty()
    print(f"[train] disable_tqdm={disable_tqdm} (stdout is{'' if not disable_tqdm else ' not'} a tty)")

    gen_eval = args.generate_eval == "on"
    save_steps = args.save_steps or args.eval_steps
    if not gen_eval and args.max_steps % save_steps:
        raise SystemExit(f"--max-steps {args.max_steps} is not a multiple of --save-steps "
                         f"{save_steps}: the final step would never be checkpointed")
    if gen_eval and save_steps % args.eval_steps:
        raise SystemExit(f"--save-steps {save_steps} must be a multiple of --eval-steps "
                         f"{args.eval_steps} when the best checkpoint is reloaded at the end")

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
        save_steps=save_steps,
        # 0 keeps every checkpoint, which post-hoc selection needs.
        save_total_limit=args.save_total_limit or None,
        # A Trainer checkpoint normally carries the AdamW state too -- 2 bytes of
        # moment per byte of parameter. For large-v2 that is 6.2 GB of weights
        # PLUS 12.3 GB of optimizer per checkpoint, so save_total_limit=2 quietly
        # needs ~37 GB of disk. We never resume mid-run (we early-stop), and
        # load_best_model_at_end only needs weights, so drop the optimizer:
        # 37 GB -> 12.4 GB. Set --save-optimizer if you actually need to resume.
        save_only_model=not args.save_optimizer,
        logging_steps=25,
        predict_with_generate=gen_eval,
        # Without generate the Trainer would otherwise gather full-vocabulary
        # logits for every dev utterance just to throw them away.
        prediction_loss_only=not gen_eval,
        generation_max_length=225,
        load_best_model_at_end=gen_eval,
        metric_for_best_model=args.select_by if gen_eval else None,
        greater_is_better=False if gen_eval else None,
        report_to=_report_to(args.report_to),
        disable_tqdm=disable_tqdm,
        seed=args.seed,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=args.dataloader_workers,
    )

    save_at = args.save_at if args.save_at is not None else ([] if gen_eval else [50, 100])
    save_at = sorted({s for s in save_at if 0 < s < args.max_steps and s % save_steps})
    extra_callbacks = []
    if save_at:
        from transformers import TrainerCallback

        class _SaveAt(TrainerCallback):
            def on_step_end(self, targs_, state, control, **kw):
                if state.global_step in save_at:
                    control.should_save = True
                return control

        extra_callbacks.append(_SaveAt())
        print(f"[train] extra checkpoints at steps {save_at}")

    trainer = Seq2SeqTrainer(
        args=targs,
        model=model,
        train_dataset=train,
        eval_dataset=dev,
        data_collator=collator,
        compute_metrics=compute_metrics if gen_eval else None,
        processing_class=processor,
        callbacks=([EarlyStoppingCallback(early_stopping_patience=args.patience)]
                   if gen_eval else []) + extra_callbacks,
    )

    # get_last_checkpoint returns None on a fresh/nonexistent --out, so this is
    # a no-op the first time a model is trained. Without --save-optimizer,
    # resumed checkpoints restore model weights and the trainer's global_step
    # (so the LR schedule and step count pick up correctly) but not Adam's
    # momentum/variance -- transformers silently re-initializes those rather
    # than erroring, since they're optional. That is an acceptable soft resume
    # for surviving a walltime cutoff; pass --save-optimizer up front for a
    # bit-exact one.
    resume_checkpoint = None
    if args.from_scratch:
        print("[train] --from-scratch: ignoring any existing checkpoint under --out")
    elif args.out.exists():
        resume_checkpoint = get_last_checkpoint(str(args.out))
    print(f"[train] {'resuming from ' + resume_checkpoint if resume_checkpoint else 'starting from step 0'}")

    result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    # With post-hoc selection the last checkpoint already IS the final model;
    # a second copy at the top level would cost another 6.2 GB for large-v2.
    if gen_eval:
        trainer.save_model(str(args.out))
    processor.save_pretrained(str(args.out))
    # Not written automatically by trainer.train() -- without this, the full
    # loss/metric history (log_history, used by plot_metrics.py) only survives
    # inside individual checkpoint-N/ dirs, which save_total_limit prunes away.
    trainer.save_state()

    # What the fine-tuning-method ablation compares besides accuracy.
    import json

    peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
    meta = {
        "model": str(args.model), "ft_method": args.ft_method,
        "params_total": n_total, "params_trainable": n_trainable,
        "trainable_pct": round(100 * n_trainable / n_total, 3),
        "peak_vram_gb": round(peak_gb, 1),
        "train_runtime_s": round(result.metrics.get("train_runtime", 0.0)),
        "global_step": result.global_step, "lr": args.lr,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "train_utts": len(train), "seed": args.seed,
        # A resumed run only timed (and only peaked) over its final leg.
        "resumed": bool(resume_checkpoint),
    }
    (args.out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[train] run_meta {meta}")

    if gen_eval:
        print(f"[train] finished at step {result.global_step:,} "
              f"(max_steps={args.max_steps}; early stopping may have fired)")
        print(f"[train] best {targs.metric_for_best_model} = "
              f"{trainer.state.best_metric}")
    else:
        print(f"[train] finished at step {result.global_step:,}; checkpoints every "
              f"{save_steps} steps await csasr.eval.select")
    print(f"[train] model -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
