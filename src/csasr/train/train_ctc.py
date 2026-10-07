"""Ablation: fine-tune a CTC encoder (MMS-1B) on the synthetic code-mixed speech.

The non-Whisper arm of the ASR-model ablation. `facebook/mms-1b-all` is a
wav2vec2 encoder with small per-language adapters and a per-language character
head. It differs from Whisper in the way that matters for this study: there is
NO autoregressive decoder, so nothing in the model is a language model that
synthetic *text* could teach. Whatever it gains has to come through the
acoustics, and the acoustics here are two TTS voices.

What is trained
---------------
The Hindi adapter and Hindi head are loaded, then the head is rebuilt for the
character set of the training text. Rows for characters the Hindi head already
has are copied across (its vocabulary does include the Latin letters, though it
was trained to use them rarely), so the model starts from the pretrained
solution; any character the head lacks starts from scratch.

* `full`  -- everything except the convolutional feature encoder.
* `lora`  -- frozen base, LoRA on the attention projections, plus the new head.
* `qlora` -- the same on a 4-bit NF4 base.

Checkpoints are saved at fixed steps and selected afterwards by
`csasr.eval.select --engine ctc`, exactly as for Whisper.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from ..normalize import normalize

__all__ = ["build_vocab", "ctc_text", "load_ctc_model", "main"]

SR = 16_000
PAD, UNK, DELIM = "[PAD]", "[UNK]", "|"
VOCAB_FILE = "vocab.json"
META_FILE = "ctc_meta.json"


def ctc_text(text: str) -> str:
    """The scorer's own normalization, so the model is trained to emit exactly
    the form it is scored in: no punctuation, lowercase Latin."""
    return normalize(text, "scoring")


def build_vocab(texts) -> dict[str, int]:
    """Character vocabulary. [PAD] is id 0 and doubles as the CTC blank."""
    chars = sorted({ch for t in texts for ch in ctc_text(t) if ch != " "})
    vocab = {PAD: 0, UNK: 1, DELIM: 2}
    for ch in chars:
        vocab.setdefault(ch, len(vocab))
    return vocab


def _tokenizer(vocab_path: Path):
    from transformers import Wav2Vec2CTCTokenizer

    return Wav2Vec2CTCTokenizer(str(vocab_path), unk_token=UNK, pad_token=PAD,
                                word_delimiter_token=DELIM)


def _fresh_model(base: str, source_lang: str, vocab: dict[str, int], quantization=None):
    """Base encoder + `source_lang` adapter, with the head rebuilt for `vocab`."""
    import torch
    from transformers import AutoTokenizer, Wav2Vec2ForCTC

    kw: dict[str, Any] = dict(target_lang=source_lang, ignore_mismatched_sizes=True,
                              ctc_loss_reduction="mean", ctc_zero_infinity=True,
                              layerdrop=0.0)
    if quantization is not None:
        kw.update(quantization_config=quantization, device_map={"": 0})
    model = Wav2Vec2ForCTC.from_pretrained(base, **kw)

    # The pretrained head is read from the adapter file rather than from
    # `model.lm_head`: under 4-bit loading that module holds packed NF4 codes,
    # not weights, and LoRA and QLoRA must start from the same head.
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    adapter = load_file(hf_hub_download(base, f"adapter.{source_lang}.safetensors"))
    old_w, old_b = adapter["lm_head.weight"].float(), adapter["lm_head.bias"].float()
    old_tok = AutoTokenizer.from_pretrained(base, target_lang=source_lang)
    old_vocab = old_tok.get_vocab()
    old_head = model.lm_head
    new_head = torch.nn.Linear(old_head.in_features, len(vocab))
    torch.nn.init.normal_(new_head.weight, std=0.02)
    torch.nn.init.zeros_(new_head.bias)
    copied = 0
    with torch.no_grad():
        for ch, new_id in vocab.items():
            # MMS spells its specials differently from ours.
            old_id = old_vocab.get({PAD: old_tok.pad_token, UNK: old_tok.unk_token}.get(ch, ch))
            if old_id is not None and old_id < old_w.shape[0]:
                new_head.weight[new_id] = old_w[old_id]
                new_head.bias[new_id] = old_b[old_id]
                copied += 1
    model.lm_head = new_head.to(next(model.parameters()).device)
    model.config.vocab_size = len(vocab)
    model.config.pad_token_id = vocab[PAD]
    # Otherwise from_pretrained() on our checkpoint would re-download the Hindi
    # adapter and overwrite the one we fine-tuned.
    model.target_lang = None
    print(f"[ctc] head rebuilt: {len(vocab)} chars, {copied} rows copied from the "
          f"'{source_lang}' head, {len(vocab) - copied} new")
    return model


def load_ctc_model(path: str | Path, *, quantization=None):
    """Load a checkpoint written by this script (full or adapter) with its processor."""
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2Processor

    path = Path(path)
    root = path if (path / META_FILE).exists() else path.parent
    meta = json.loads((root / META_FILE).read_text(encoding="utf-8"))
    vocab = json.loads((root / VOCAB_FILE).read_text(encoding="utf-8"))
    processor = Wav2Vec2Processor(
        Wav2Vec2FeatureExtractor.from_pretrained(meta["base"]), _tokenizer(root / VOCAB_FILE)
    )
    if (path / "adapter_config.json").exists():
        from peft import PeftModel

        model = _fresh_model(meta["base"], meta["source_lang"], vocab, quantization)
        model = PeftModel.from_pretrained(model, str(path)).merge_and_unload()
    else:
        model = Wav2Vec2ForCTC.from_pretrained(str(path))
    return model, processor


@dataclass
class CTCCollator:
    processor: Any

    def __call__(self, features: list[dict]) -> dict:
        batch = self.processor.feature_extractor.pad(
            [{"input_values": f["input_values"]} for f in features],
            return_tensors="pt", return_attention_mask=True,
        )
        labels = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt"
        )
        batch["labels"] = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        return batch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="facebook/mms-1b-all")
    ap.add_argument("--source-lang", default="hin", help="MMS adapter to start from")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-hf", default=None)
    ap.add_argument("--train-config", default="synth_t2")
    ap.add_argument("--train-manifest", type=Path, default=None)
    ap.add_argument("--subset-ids", type=Path, default=None)
    ap.add_argument("--dev-hf", default=None)
    ap.add_argument("--dev-config", default="dev")
    ap.add_argument("--dev-manifest", type=Path, default=None)
    ap.add_argument("--dev-max-utts", type=int, default=300)
    ap.add_argument("--ft-method", choices=("full", "lora", "qlora"), default="full")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-targets", nargs="+", default=["q_proj", "v_proj"])
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--save-at", type=int, nargs="*", default=[50, 100],
                    help="extra checkpoint steps, as in train_whisper")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dataset-fraction", type=float, default=1.0)
    ap.add_argument("--dataloader-workers", type=int, default=4)
    ap.add_argument("--from-scratch", action="store_true")
    args = ap.parse_args(argv)

    import torch
    from transformers import (
        Trainer,
        TrainingArguments,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2Processor,
    )
    from transformers.trainer_utils import get_last_checkpoint

    from ..data.loaders import apply_subset, load_hub_dataset, load_manifest_dataset

    token = os.environ.get("HF_TOKEN")

    def _load(hf, cfg, manifest):
        if hf:
            return load_hub_dataset(hf, cfg, token=token)
        if manifest:
            return load_manifest_dataset(manifest)
        return None

    train = _load(args.train_hf, args.train_config, args.train_manifest)
    dev = _load(args.dev_hf, args.dev_config, args.dev_manifest)
    if train is None or dev is None:
        raise SystemExit("need --train-hf/--train-manifest and --dev-hf/--dev-manifest")
    if args.subset_ids:
        train = apply_subset(train, args.subset_ids)
    if args.dataset_fraction < 1.0:
        train = train.shuffle(seed=args.seed).select(
            range(max(1, int(len(train) * args.dataset_fraction))))
    if args.dev_max_utts and len(dev) > args.dev_max_utts:
        dev = dev.shuffle(seed=args.seed).select(range(args.dev_max_utts))
    print(f"[ctc] train={len(train):,} utts   dev={len(dev):,} utts")

    # The vocabulary is fixed by the FIRST launch and reused on resume: rebuilding
    # it from a different data fraction would silently renumber the head.
    args.out.mkdir(parents=True, exist_ok=True)
    vocab_path = args.out / VOCAB_FILE
    if vocab_path.exists() and not args.from_scratch:
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    else:
        vocab = build_vocab(train["text"])
        vocab_path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    (args.out / META_FILE).write_text(json.dumps(
        {"base": args.model, "source_lang": args.source_lang, "ft_method": args.ft_method}),
        encoding="utf-8")

    processor = Wav2Vec2Processor(
        Wav2Vec2FeatureExtractor.from_pretrained(args.model), _tokenizer(vocab_path)
    )

    quant = None
    if args.ft_method == "qlora":
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16)
    model = _fresh_model(args.model, args.source_lang, vocab, quant)
    n_total = model.num_parameters()  # counts 4-bit layers at their real size

    if args.ft_method == "full":
        model.freeze_feature_encoder()
    else:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if args.ft_method == "qlora":
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=list(args.lora_targets), bias="none",
            modules_to_save=["lm_head"],  # the new head has to train too
        ))
        # The head was rebuilt, so config.vocab_size differs from the base's and
        # PEFT's "auto" concludes the embeddings were resized and tries to save
        # them -- but a wav2vec2 has no input embeddings and raises
        # NotImplementedError at the first checkpoint. The head is already
        # saved through modules_to_save; the Trainer offers no way to pass this.
        import functools

        model.save_pretrained = functools.partial(model.save_pretrained,
                                                  save_embedding_layers=False)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ctc] ft-method {args.ft_method}: {n_trainable/1e6:,.2f} M trainable of "
          f"{n_total/1e6:,.0f} M ({n_trainable/n_total:.2%})")

    # Featurized lazily: raw waveforms for 22 h would be a ~5 GB Arrow cache per
    # run, and unlike Whisper's log-mels there is nothing expensive to precompute.
    def _transform(batch):
        out = {"input_values": [], "labels": []}
        for audio, text in zip(batch["audio"], batch["text"]):
            out["input_values"].append(
                processor.feature_extractor(audio["array"], sampling_rate=SR).input_values[0])
            out["labels"].append(processor.tokenizer(ctc_text(text)).input_ids)
        return out

    train = train.with_transform(_transform)
    dev = dev.with_transform(_transform)

    ampere = torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (8, 0)
    if ampere:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    targs = TrainingArguments(
        output_dir=str(args.out),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        bf16=ampere,
        fp16=torch.cuda.is_available() and not ampere,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_steps=args.save_steps,
        save_total_limit=None,
        save_only_model=True,
        prediction_loss_only=True,
        logging_steps=25,
        report_to=[],
        disable_tqdm=not sys.stdout.isatty(),
        seed=args.seed,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=args.dataloader_workers,
    )
    save_at = sorted({s for s in args.save_at
                      if 0 < s < args.max_steps and s % args.save_steps})
    callbacks = []
    if save_at:
        from transformers import TrainerCallback

        class _SaveAt(TrainerCallback):
            def on_step_end(self, targs_, state, control, **kw):
                if state.global_step in save_at:
                    control.should_save = True
                return control

        callbacks.append(_SaveAt())
        print(f"[ctc] extra checkpoints at steps {save_at}")

    trainer = Trainer(args=targs, model=model, train_dataset=train, eval_dataset=dev,
                      data_collator=CTCCollator(processor), callbacks=callbacks)

    resume = None if args.from_scratch else get_last_checkpoint(str(args.out))
    print(f"[ctc] {'resuming from ' + resume if resume else 'starting from step 0'}")
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_state()

    peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
    meta = {
        "model": args.model, "ft_method": args.ft_method,
        "params_total": n_total, "params_trainable": n_trainable,
        "trainable_pct": round(100 * n_trainable / n_total, 3),
        "peak_vram_gb": round(peak_gb, 1),
        "train_runtime_s": round(result.metrics.get("train_runtime", 0.0)),
        "global_step": result.global_step, "lr": args.lr,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "train_utts": len(train), "seed": args.seed, "resumed": bool(resume),
    }
    (args.out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[ctc] run_meta {meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
