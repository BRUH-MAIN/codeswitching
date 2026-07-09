#!/usr/bin/env python
"""Generate the Kaggle notebooks from source, so they stay valid JSON and reviewable.

The notebooks are deliberately THIN: they pip-install this package and call into
`csasr`. Logic belongs in version-controlled, unit-tested modules -- not in
notebook cells, where it cannot be reviewed or tested.

    python scripts/build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = "https://github.com/BRUH-MAIN/codeswitching.git"
OUT = Path(__file__).resolve().parents[1] / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
            "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [], "isInternetEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ===========================================================================
# 01 - generate dataset (LLM text + TTS audio + push to Hub)
# ===========================================================================

NB01 = notebook([
    md(f"""
# Track 2 · Stage 1+2 — Generate the synthetic code-mixed corpus

Replication of Biswas et al., Interspeech 2025 (Track 2).

This notebook does **all** data generation and pushes the result to the Hub:

1. **Stage 1** — few-shot prompt Llama-3.1-8B for Hindi-English bigrams, filter them,
   expand each into four sentences (~16k sentences).
2. **Stage 2** — synthesize them with Indic Parler-TTS (Rohit / Divya), 22h of audio.
3. Push `synth_t2` + `t1_ids.json` to `RohanRamesh/hi-en-synth-cs`.

### Why one notebook can host both models
`parler-tts` hard-pins `transformers==4.46.1`. vLLM will not tolerate that pin, but
`transformers` 4.46.1 *does* support Llama-3.1, so the LLM runs under bitsandbytes NF4
in the same environment. (vLLM is also out on principle: its AWQ kernels need compute
capability ≥ 8.0 and the T4 is sm75.)

### Before you run
* Accept the licence for `ai4bharat/indic-parler-tts` (gated) and `meta-llama/Llama-3.1-8B-Instruct`.
* Add your HF **write** token to Kaggle Secrets as `HF_TOKEN`.
* Turn notebook internet **on**, accelerator **GPU T4 x2**.
* `STAGE` below lets you resume: `text`, `audio`, or `both`.

Runtime ≈ 2h (LLM) + 4–6h (TTS). Everything is cached/sharded, so a 12h timeout
costs at most the in-flight shard.
"""),

    md("## 0 · Install\n\n**Cell order matters.** The `transformers` downgrade silently no-ops if anything has already imported `transformers`. Do not import it above this cell."),
    code(f"""
!pip install -q "transformers==4.46.1" "datasets<4" bitsandbytes accelerate soxr
!pip install -q git+https://github.com/huggingface/parler-tts.git
!pip install -q git+{REPO}

import transformers
assert transformers.__version__ == "4.46.1", (
    f"transformers is {{transformers.__version__}}, expected 4.46.1 - "
    "something imported it before the downgrade took effect. Restart the kernel."
)
print("transformers", transformers.__version__)
"""),

    code("""
import os, gc, json, subprocess, sys
from pathlib import Path

# Keep 8.7 GB of model weights out of the 20 GB /kaggle/working budget.
os.environ["HF_HOME"] = "/kaggle/temp/hf"
Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)

from kaggle_secrets import UserSecretsClient
HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN

from huggingface_hub import login
login(token=HF_TOKEN)

STAGE      = "both"          # "text" | "audio" | "both"
REAL_REPO  = "RohanRamesh/mucs-he-cs"
SYNTH_REPO = "RohanRamesh/hi-en-synth-cs"
LLM        = "meta-llama/Llama-3.1-8B-Instruct"   # or unsloth/Meta-Llama-3.1-8B-Instruct (ungated)
NUM_SHARDS = 4

WORK = Path("/kaggle/working")
MAN  = WORK / "manifests"; MAN.mkdir(parents=True, exist_ok=True)
CACHE = WORK / "llm_cache"; CACHE.mkdir(parents=True, exist_ok=True)
AUDIO = WORK / "audio";    AUDIO.mkdir(parents=True, exist_ok=True)

def run(*args):
    print(">", " ".join(str(a) for a in args), flush=True)
    subprocess.run([sys.executable, "-m", *args], check=True)

!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),

    md("## 1 · Pull the in-domain transcripts (few-shot exemplars)\n\nTrack 2 never trains on real code-switched audio — we only need the *text* of the MUCS train split."),
    code("""
from datasets import load_dataset
from csasr.manifest import write_jsonl

train_text = load_dataset(REAL_REPO, "train_text", split="train", token=HF_TOKEN)
write_jsonl(MAN / "mucs_train.jsonl", [dict(r) for r in train_text])
print(f"{len(train_text):,} in-domain sentences for few-shot prompting")
print(train_text[0]["text"])
"""),

    md("## 2 · Stage 1a — generate bigrams\n\nPaper: 44,657 raw → 5,932 unique (13.3%)."),
    code("""
if STAGE in ("text", "both"):
    run("csasr.llm.gen_bigrams",
        "--train-manifest", MAN / "mucs_train.jsonl",
        "--out", MAN / "bigrams_raw.jsonl",
        "--cache", CACHE / "bigrams.jsonl",
        "--model", LLM, "--n-calls", "4466", "--batch-size", "16")
"""),

    md("## 3 · Stage 1b — filter\n\nDeterministic script filter, then an LLM translation check with 3-sample self-consistency.\nPaper: 5,932 unique → 5,477 valid (92.3%)."),
    code("""
if STAGE in ("text", "both"):
    run("csasr.llm.filter_bigrams",
        "--raw", MAN / "bigrams_raw.jsonl",
        "--out", MAN / "bigrams_valid.jsonl",
        "--cache", CACHE / "transcheck.jsonl",
        "--model", LLM, "--items-per-call", "20", "--n-samples", "3")
"""),

    md("## 4 · Stage 2a — expand bigrams into sentences\n\nFour per bigram (2 English-matrix, 2 Hindi-matrix). Paper: ~16,000 unique from a theoretical 21,908."),
    code("""
if STAGE in ("text", "both"):
    run("csasr.llm.gen_sentences",
        "--bigrams", MAN / "bigrams_valid.jsonl",
        "--out", MAN / "sentences.jsonl",
        "--cache", CACHE / "sentences.jsonl",
        "--model", LLM, "--batch-size", "16")
"""),

    md("### GATE 1 — yield ratios must track the paper\n\nA large divergence in the 13.3% dedup rate means the prompt or temperature is off. Decide here, not later."),
    code("""
from csasr.manifest import read_jsonl

raw   = list(read_jsonl(MAN / "bigrams_raw.jsonl"))
uniq  = {r["bigram"] for r in raw}
valid = list(read_jsonl(MAN / "bigrams_valid.jsonl"))
sents = list(read_jsonl(MAN / "sentences.jsonl"))

rows = [
    ("raw bigrams",    len(raw),   44_657, None),
    ("unique bigrams", len(uniq),   5_932, len(uniq) / max(len(raw), 1)),
    ("valid bigrams",  len(valid),  5_477, len(valid) / max(len(uniq), 1)),
    ("sentences",      len(sents), 16_000, None),
]
print(f"{'metric':<16}{'ours':>10}{'paper':>10}{'survival':>12}   paper survival")
for name, got, want, surv in rows:
    s = f"{surv:.1%}" if surv else "-"
    print(f"{name:<16}{got:>10,}{want:>10,}{s:>12}")
print("\\npaper survival: dedup 13.3%, filter 92.3%")

# Free the LLM before Parler-TTS claims the GPU.
import torch
gc.collect(); torch.cuda.empty_cache()
"""),

    md("## 5 · Push the text artifacts immediately\n\nCheckpointing to the Hub *before* the long TTS run means a session timeout never costs the LLM stage."),
    code("""
if STAGE in ("text", "both"):
    for man, cfg in [("bigrams_valid.jsonl", "bigrams"), ("sentences.jsonl", "sentences")]:
        run("csasr.data.push_to_hub", "--manifest", MAN / man,
            "--repo", SYNTH_REPO, "--config", cfg, "--text-only", "--token", HF_TOKEN)
"""),

    md("## 6 · Stage 2b — synthesize audio\n\nIndic Parler-TTS, Rohit (male) / Divya (female) 50/50. Emits 22.05 kHz → resampled to 16 kHz, written as int16 (float32 would be 7 GB).\n\nSharded: rerun this cell after a timeout and it skips whatever already exists."),
    code("""
if STAGE in ("audio", "both"):
    # Pull sentences back from the Hub if this is a fresh session.
    if not (MAN / "sentences.jsonl").exists():
        ds = load_dataset(SYNTH_REPO, "sentences", split="train", token=HF_TOKEN)
        write_jsonl(MAN / "sentences.jsonl", [dict(r) for r in ds])

    for shard in range(NUM_SHARDS):
        run("csasr.tts.synthesize",
            "--sentences", MAN / "sentences.jsonl",
            "--audio-dir", AUDIO,
            "--out", MAN / f"train_t2.shard{shard}.jsonl",
            "--shard", str(shard), "--num-shards", str(NUM_SHARDS),
            "--batch-size", "8")
"""),

    md("## 7 · Merge shards, build Train_T1 by reference\n\n### GATE 2 — `Train_T1 ⊆ Train_T2`, durations ≈ 8h / 22h"),
    code("""
import itertools
merged = list(itertools.chain.from_iterable(
    read_jsonl(MAN / f"train_t2.shard{i}.jsonl") for i in range(NUM_SHARDS)
))
write_jsonl(MAN / "train_t2.jsonl", merged)
hours = sum(r["dur"] for r in merged) / 3600
print(f"Train_T2: {len(merged):,} clips, {hours:.2f} h (paper: 22 h)")

run("csasr.tts.make_subset", "--t2", MAN / "train_t2.jsonl",
    "--out", WORK / "t1_ids.json", "--hours", "8.0")
"""),

    code("""
# Spot-listen 2 clips per voice before committing 5 GPU-hours of training to them.
import IPython.display as ipd, random
from csasr.tts.speakers import assign_speaker
random.seed(0)
for voice in ("Rohit", "Divya"):
    picks = [r for r in merged if assign_speaker(r["sent_id"]).name == voice][:200]
    for r in random.sample(picks, 2):
        print(f"[{voice}] {r['text']}")
        ipd.display(ipd.Audio(r["wav"]))
"""),

    md("## 8 · Push the synthetic corpus\n\nParquet + FLAC ≈ 1.4 GB (vs 2.5 GB as WAV). The round-trip check catches a corrupted upload *before* a 3h training run."),
    code("""
run("csasr.data.push_to_hub", "--manifest", MAN / "train_t2.jsonl",
    "--repo", SYNTH_REPO, "--config", "synth_t2", "--token", HF_TOKEN, "--verify")

from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj=str(WORK / "t1_ids.json"),
                    path_in_repo="t1_ids.json", repo_id=SYNTH_REPO,
                    repo_type="dataset", token=HF_TOKEN)
print("done ->", SYNTH_REPO)
"""),
])


# ===========================================================================
# 02 - train M6 / M7 / M8
# ===========================================================================

NB02 = notebook([
    md(f"""
# Track 2 · Stage 3 — Fine-tune Whisper (M6 / M7 / M8)

Pulls everything from the Hub. Nothing is generated here.

| model | training data | paper MER |
|---|---|---|
| M6 | Train_T1 (8h synthetic)        | 48.2 |
| M7 | Train_T2 (22h synthetic)       | 40.8 |
| M8 | Train_T2 + Common Voice mono   | 39.2 |

`whisper-small` (244M) is **fully fine-tuned** — no LoRA, no quantization. Optimizer
state is ~4 GB, which fits a 16 GB T4, so the paper's recipe survives intact:
AdamW, lr 2e-5, effective batch 64.

**T4 is Turing**: no bf16, no FlashAttention-2. `fp16=True` gives autocast + GradScaler.

Deviation D2: the paper used whisper-large-v2 (1.54B). Absolute MER will be far worse;
what we test is the **M6 → M7 → M8 ordering**.
"""),

    code(f"""
!pip install -q git+{REPO}
# datasets>=4 decodes Audio via torchcodec, which stock Kaggle images lack.
!pip install -q -U transformers accelerate evaluate jiwer
!pip install -q "datasets<4" librosa soundfile

import os, subprocess, sys
from pathlib import Path
os.environ["HF_HOME"] = "/kaggle/temp/hf"

from kaggle_secrets import UserSecretsClient
HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
from huggingface_hub import hf_hub_download, login
login(token=HF_TOKEN)

SYNTH = "RohanRamesh/hi-en-synth-cs"
REAL  = "RohanRamesh/mucs-he-cs"
MODEL = "openai/whisper-small"

def run(*args):
    print(">", " ".join(str(a) for a in args), flush=True)
    subprocess.run([sys.executable, "-m", *args], check=True)

t1_ids = hf_hub_download(SYNTH, "t1_ids.json", repo_type="dataset", token=HF_TOKEN)
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),

    md("## Smoke test first\n\n20 steps on 1% of the data proves the collator, the `<|hi|>` prefix tokens, and the label masking work — before committing a 3h session."),
    code("""
run("csasr.train.train_whisper", "--model", MODEL, "--out", "/kaggle/working/smoke",
    "--train-hf", SYNTH, "--train-config", "synth_t2",
    "--dev-hf", REAL, "--dev-config", "dev", "--hf-token", HF_TOKEN,
    "--max-steps", "20", "--eval-steps", "10", "--dataset-fraction", "0.01",
    "--batch-size", "4", "--grad-accum", "1")
"""),

    md("## M6 — Train_T1 (8h synthetic)"),
    code("""
run("csasr.train.train_whisper", "--model", MODEL, "--out", "/kaggle/working/m6",
    "--train-hf", SYNTH, "--train-config", "synth_t2", "--subset-ids", t1_ids,
    "--dev-hf", REAL, "--dev-config", "dev", "--hf-token", HF_TOKEN,
    "--lr", "2e-5", "--batch-size", "16", "--grad-accum", "4",
    "--max-steps", "5000", "--eval-steps", "250", "--patience", "4")
"""),

    md("## M7 — Train_T2 (22h synthetic)"),
    code("""
run("csasr.train.train_whisper", "--model", MODEL, "--out", "/kaggle/working/m7",
    "--train-hf", SYNTH, "--train-config", "synth_t2",
    "--dev-hf", REAL, "--dev-config", "dev", "--hf-token", HF_TOKEN,
    "--lr", "2e-5", "--batch-size", "16", "--grad-accum", "4",
    "--max-steps", "5000", "--eval-steps", "250", "--patience", "4")
"""),

    md("## M8 — Train_T2 + Common Voice monolingual (52h)\n\nStill one `<|hi|>` prompt for everything: language-*specific* prompting is Track 1's M4."),
    code("""
run("csasr.train.train_whisper", "--model", MODEL, "--out", "/kaggle/working/m8",
    "--train-hf", SYNTH, "--train-config", "synth_t2",
    "--extra-hf", f"{REAL}:cv_hi", f"{REAL}:cv_en",
    "--dev-hf", REAL, "--dev-config", "dev", "--hf-token", HF_TOKEN,
    "--lr", "2e-5", "--batch-size", "16", "--grad-accum", "4",
    "--max-steps", "5000", "--eval-steps", "250", "--patience", "4")
"""),

    md("## Push checkpoints so `03_eval` can find them"),
    code("""
from huggingface_hub import HfApi
api = HfApi()
for m in ("m6", "m7", "m8"):
    repo = f"RohanRamesh/whisper-small-cs-{m}"
    api.create_repo(repo, exist_ok=True, private=True, token=HF_TOKEN)
    api.upload_folder(folder_path=f"/kaggle/working/{m}", repo_id=repo, token=HF_TOKEN)
    print("pushed", repo)
"""),
])


# ===========================================================================
# 03 - eval (Gate 3 + decode)
# ===========================================================================

NB03 = notebook([
    md(f"""
# Track 2 · Stage 4 — Evaluation

## Run **GATE 3 first**, before any generation or training.

`whisper-large-v2` zero-shot on the test set should score **≈ 52.0 MER / 42.9 CBA-HE**.
Inference only — no training, ~3 GB, ~40 min on a T4.

That number is published in the paper, so reproducing it validates decoding,
normalization, word-level language ID, and both metrics end to end. It is **not**
our baseline — it is the calibration of the measuring instrument. A broken metric
makes every downstream result uninterpretable.

Then decode M6/M7/M8 and the `whisper-small` zero-shot baseline they are measured against.

Decoding uses `language=None` (Whisper auto-detect), reproducing WhisperX's "None"
option from the paper. That is the whole point: a model that has not learned
code-switching mis-detects the language and then transliterates or deletes.
"""),

    code(f"""
!pip install -q git+{REPO}
!pip install -q -U transformers jiwer
!pip install -q "datasets<4" librosa soundfile   # see 02_train: torchcodec

import os, subprocess, sys
os.environ["HF_HOME"] = "/kaggle/temp/hf"
from kaggle_secrets import UserSecretsClient
HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
from huggingface_hub import login; login(token=HF_TOKEN)

REAL = "RohanRamesh/mucs-he-cs"
OUT  = "/kaggle/working"

def run(*args):
    print(">", " ".join(str(a) for a in args), flush=True)
    subprocess.run([sys.executable, "-m", *args], check=True)
"""),

    md("## GATE 3 — calibrate the metric against a published number"),
    code("""
run("csasr.eval.decode", "--model", "openai/whisper-large-v2",
    "--test-hf", REAL, "--test-config", "test", "--hf-token", HF_TOKEN,
    "--out", f"{OUT}/hyp_largev2_zeroshot.jsonl", "--batch-size", "8")
"""),

    code("""
import json
from datasets import load_dataset
from csasr.manifest import write_jsonl
from csasr.eval.score import score

test = load_dataset(REAL, "test", split="train", token=HF_TOKEN)
write_jsonl(f"{OUT}/refs_test.jsonl",
            [{"utt_id": r["utt_id"], "text": r["text"]} for r in test])

for mode in ("word", "hybrid"):
    res = score(f"{OUT}/refs_test.jsonl", f"{OUT}/hyp_largev2_zeroshot.jsonl", mer_mode=mode)
    print(f"MER mode={mode:<7} -> MER {res['mer']:.1f}   CBA-HE {res['cba_he']:.1f}   CBA-EH {res['cba_eh']:.1f}")

print("\\npaper (large-v2 zero-shot): MER 52.0   CBA-HE 42.9   CBA-EH 36.x")
print("Whichever mode lands near 52.0 is the definition the authors used.")
"""),

    md("## whisper-small zero-shot — the actual baseline for M6/M7/M8"),
    code("""
run("csasr.eval.decode", "--model", "openai/whisper-small",
    "--test-hf", REAL, "--test-config", "test", "--hf-token", HF_TOKEN,
    "--out", f"{OUT}/hyp_small_zeroshot.jsonl", "--batch-size", "16")
"""),

    md("## Decode the fine-tuned models"),
    code("""
for m in ("m6", "m7", "m8"):
    run("csasr.eval.decode", "--model", f"RohanRamesh/whisper-small-cs-{m}",
        "--test-hf", REAL, "--test-config", "test", "--hf-token", HF_TOKEN,
        "--out", f"{OUT}/hyp_{m}.jsonl", "--batch-size", "16")
"""),

    md("## Results — reproduce the ordering of Table 2"),
    code("""
systems = [
    ("large-v2 zero-shot", "hyp_largev2_zeroshot.jsonl", 52.0),
    ("small zero-shot",    "hyp_small_zeroshot.jsonl",   None),
    ("M6 (T1, 8h)",        "hyp_m6.jsonl",               48.2),
    ("M7 (T2, 22h)",       "hyp_m7.jsonl",               40.8),
    ("M8 (T2 + mono)",     "hyp_m8.jsonl",               39.2),
]
rows = []
for name, f, paper_mer in systems:
    r = score(f"{OUT}/refs_test.jsonl", f"{OUT}/{f}")
    rows.append((name, r, paper_mer))

print(f"{'system':<20}{'MER':>8}{'paper':>8}{'CBA-HE':>9}{'CBA-EH':>9}")
for name, r, pm in rows:
    p = f"{pm:.1f}" if pm else "-"
    print(f"{name:<20}{r['mer']:>8.1f}{p:>8}{r['cba_he']:>9.1f}{r['cba_eh']:>9.1f}")

mers = [r["mer"] for _, r, _ in rows[2:]]
print("\\nM6 > M7 > M8 ordering reproduced:", mers == sorted(mers, reverse=True))

with open(f"{OUT}/table2.json", "w") as fh:
    json.dump([{"system": n, **r} for n, r, _ in rows], fh, indent=2)
"""),
])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, nb in [
        ("01_generate_dataset.ipynb", NB01),
        ("02_train.ipynb", NB02),
        ("03_eval.ipynb", NB03),
    ]:
        p = OUT / name
        p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {p}  ({len(nb['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
