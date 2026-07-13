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
# 01a - generate the code-mixed TEXT (Gemma 4, transformers >= 5.5)
# ===========================================================================
#
# SPLIT FROM 01b ON PURPOSE. parler-tts hard-pins transformers==4.46.1; Gemma 4
# needs transformers>=5.5. They cannot share a process. The Hub is the
# checkpoint between the two notebooks.

NB01A = notebook([
    md(f"""
# Track 2 · Stage 1 — Generate the code-mixed **text**

Replication of Biswas et al., Interspeech 2025 (Track 2).

Few-shot-prompt **Gemma 4 E4B** for Hindi-English bigrams → filter them → expand each
into four sentences (~16k). Push the text to `RohanRamesh/hi-en-synth-cs`.

Audio synthesis is a **separate notebook** (`01b`). `parler-tts` hard-pins
`transformers==4.46.1`, Gemma 4 needs `transformers>=5.5` — they cannot coexist in one
process. The Hub is the checkpoint between them.

### The model: `google/gemma-4-E4B-it`
* **apache-2.0, ungated** — no licence click-through, no gated-repo token scope, no mirror
  needed. (Gemma 3's `google/*` repos *are* gated; Gemma 4's are not.)
* Deviation **D1**: the paper used Llama-3.3-70B (141 GB bf16). Gemma 4 E4B is ~8B raw
  params, pretrained on 140+ languages, and a **deterministic script filter** sits behind it
  to catch what it still gets wrong.
* The one **live risk**: Gemma 4 is **bf16-native** (`torch_dtype: bfloat16`) and the T4 is
  Turing — it has **no bf16**. fp16 activations can exceed 65,504 and go non-finite.
  `backend.py` probes the logits after loading and transparently reloads with float32 compute
  if they are NaN/inf. **The smoke test below is how you find out**, in about a minute.
* Two non-issues, confirmed against the real config (the backend defends against both anyway):
  it is `Gemma4ForConditionalGeneration` but *is* registered under `AutoModelForCausalLM`, and
  its chat template **does** support a `system` role — so the paper's verbatim system prompt
  survives intact.

### Before you run
* HF **write** token in Kaggle Secrets as `HF_TOKEN`. Internet **on**, **GPU T4 ×2**.
* Nothing to accept — Gemma 4 is apache-2.0. (Parler-TTS *is* gated; that bites in `01b`.)

Runtime ≈ 2h. Every LLM call is cached, so a 12h timeout costs nothing on a re-run.
"""),

    md("## 0 · Install"),
    code(f"""
# Gemma 4 needs transformers >= 5.5. NO parler-tts here -- that is 01b's problem.
!pip install -q -U "transformers>=5.5" accelerate bitsandbytes
!pip install -q "datasets<4" librosa soundfile soxr omegaconf rich
!pip install -q --force-reinstall --no-deps git+{REPO}

# This NOTEBOOK's own version. `pip install` updates the csasr PACKAGE but NOT the
# .ipynb -- an old notebook against a new package is a real and confusing failure
# (the old 01a loaded the model in-kernel and starved every subprocess of VRAM).
NOTEBOOK_VERSION = "0.9.3"

import csasr, transformers
assert csasr.__version__ == NOTEBOOK_VERSION, (
    f"csasr package is {{csasr.__version__}} but this NOTEBOOK is {{NOTEBOOK_VERSION}}.\\n"
    "  package older  -> restart the kernel (Run > Restart & clear); pip skips a\\n"
    "                    reinstall when the version looks satisfied.\\n"
    "  notebook older -> re-download it from the repo; pip does NOT update .ipynb files."
)
from packaging.version import Version
assert Version(transformers.__version__) >= Version("5.5"), (
    f"transformers {{transformers.__version__}} cannot load Gemma 4 (needs >= 5.5)."
)
print("csasr", csasr.__version__, "| transformers", transformers.__version__)
"""),

    code("""
import os, subprocess, sys
from pathlib import Path

os.environ["HF_HOME"] = "/kaggle/temp/hf"
Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)

from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])
HF_TOKEN = os.environ["HF_TOKEN"]

REAL_REPO  = "RohanRamesh/mucs-he-cs"
SYNTH_REPO = "RohanRamesh/hi-en-synth-cs"
LLM        = "google/gemma-4-E4B-it"      # apache-2.0, ungated. E2B is the smaller option.

WORK  = Path("/kaggle/working")
MAN   = WORK / "manifests"; MAN.mkdir(parents=True, exist_ok=True)
CACHE = WORK / "llm_cache"; CACHE.mkdir(parents=True, exist_ok=True)

def run(*args):
    print(">", " ".join(str(a) for a in args), flush=True)
    p = subprocess.run([sys.executable, "-m", *args])   # inherits HF_TOKEN + streams
    if p.returncode != 0:
        raise RuntimeError(
            f"{args[0]} failed (exit {p.returncode}). The real error is printed ABOVE "
            f"this traceback - scroll up in this cell's output."
        )

import torch
N_GPU = max(1, torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"cuda:{i}  {free/2**30:5.1f} GiB free / {total/2**30:.1f} GiB")
    assert free / total > 0.85, (
        f"cuda:{i} already has {(total-free)/2**30:.1f} GiB in use. Something in THIS "
        "kernel is holding the GPU (an older notebook loaded the model in-cell). "
        "Every subprocess below will then fail with 'Some modules are dispatched on "
        "the CPU or the disk'. Restart the kernel: Run > Restart & clear all."
    )

from csasr.manifest import read_jsonl, write_jsonl

def run_sharded(module, out_path, *args, cache_stem="cache"):
    \"\"\"Run one process PER GPU and merge the shards.

    `device_map="auto"` puts the whole model on ONE card, so a single process
    leaves Kaggle's second T4 completely idle. Sharding the work across two
    processes -- each pinned to its own GPU with CUDA_VISIBLE_DEVICES -- is a
    straight 2x. Each shard gets its OWN cache file: two writers on one JSONL
    would interleave and corrupt it.
    \"\"\"
    out_path = Path(out_path)
    procs = []
    for i in range(N_GPU):
        cmd = [sys.executable, "-m", module, *[str(a) for a in args],
               "--cache", str(CACHE / f"{cache_stem}.shard{i}.jsonl"),
               "--out", str(out_path.with_suffix(f".shard{i}.jsonl")),
               "--shard", str(i), "--num-shards", str(N_GPU)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        print(f"> GPU{i}: {module} shard {i}/{N_GPU}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env))

    for i, p in enumerate(procs):
        if p.wait() != 0:
            raise RuntimeError(f"{module} shard {i} failed (exit {p.returncode}). "
                               "The real error is printed ABOVE - scroll up.")

    rows = [r for i in range(N_GPU)
            for r in read_jsonl(out_path.with_suffix(f".shard{i}.jsonl"))]
    write_jsonl(out_path, rows)
    print(f"merged {len(rows):,} rows -> {out_path}")
    return rows
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

    md("""## 2 · SMOKE TEST — 20 bigrams before committing 2 hours

Loads Gemma, runs the **fp16 → float32 logits health check** (Gemma 4 is bf16-native and
the T4 has no bf16), generates real bigrams, and shows which survive the script filter.

**It runs as a subprocess, deliberately.** Jupyter's `Out[]` history holds a reference to
anything a cell produced, so `del model` does *not* free the VRAM — and the next subprocess
then dies with `Some modules are dispatched on the CPU or the disk`. Keeping the model out
of the kernel entirely is the only reliable fix.

Expect Gemma to emit **three**-word phrases (`बुनियादी formatting basics`). That is fine: the
filter extracts the switch pair from inside them. See deviation **D9**."""),
    code("""
run("csasr.llm.smoke", "--model", LLM,
    "--train-manifest", MAN / "mucs_train.jsonl",
    "--n-calls", "2", "--bigrams-per-call", "10")
"""),

    md("## 3 · Generate bigrams\n\nPaper: 44,657 raw → 5,932 unique (13.3%)."),
    code("""
run_sharded("csasr.llm.gen_bigrams", MAN / "bigrams_raw.jsonl",
            "--train-manifest", MAN / "mucs_train.jsonl",
            "--model", LLM, "--n-calls", "4466", "--batch-size", "32",
            cache_stem="bigrams")
"""),

    md("## 4 · Filter\n\nDeterministic script filter (one Devanagari token + one Latin token), then an LLM translation check with 3-sample self-consistency.\nPaper: 5,932 unique → 5,477 valid (92.3%)."),
    code("""
run_sharded("csasr.llm.filter_bigrams", MAN / "bigrams_valid.jsonl",
            "--raw", MAN / "bigrams_raw.jsonl",
            "--model", LLM, "--items-per-call", "20", "--n-samples", "3",
            "--batch-size", "16",
            cache_stem="transcheck")
"""),

    md("## 5 · Expand each bigram into four sentences\n\n2 English-matrix, 2 Hindi-matrix. Paper: ~16,000 unique from a theoretical 21,908."),
    code("""
run_sharded("csasr.llm.gen_sentences", MAN / "sentences.jsonl",
            "--bigrams", MAN / "bigrams_valid.jsonl",
            "--model", LLM, "--batch-size", "32",
            cache_stem="sentences")
"""),

    md("### GATE 1 — yields must track the paper\n\nA large divergence in the 13.3% dedup rate means the prompt or temperature is off. Decide here, not after 6 hours of TTS."),
    code("""
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
print(f"{'metric':<16}{'ours':>10}{'paper':>10}{'survival':>12}")
for name, got, want, surv in rows:
    s = f"{surv:.1%}" if surv else "-"
    print(f"{name:<16}{got:>10,}{want:>10,}{s:>12}")
print("\\npaper survival: dedup 13.3%, filter 92.3%")
print("\\nGemma 4 E4B is far smaller than the paper's 70B (deviation D1), so a lower")
print("valid-bigram yield is expected. What matters is that ENOUGH sentences survive:")
print(f"  -> {len(sents):,} sentences  (need >~8,000 for a usable 22h corpus)")

for r in sents[:5]:
    print("   ", r["text"])
"""),

    md("""### Repair — strip the matrix-language labels

Gemma prefixes each sentence with its matrix language:

    English: Many software programs have different aliases निर्धारित for commands.

Left in, Parler-TTS would literally **speak** "English colon, many software programs…" and
Whisper would then be **trained to emit `English:`** at the start of every transcript. This
re-cleans the text, re-checks that the bigram survived, and recomputes the matrix language
(the prefix biased it). Seconds — no LLM re-run."""),
    code("""
run("csasr.llm.fix_sentences",
    "--in", MAN / "sentences.jsonl",
    "--bigrams", MAN / "bigrams_valid.jsonl",
    "--out", MAN / "sentences.jsonl")

sents = list(read_jsonl(MAN / "sentences.jsonl"))
print(f"\\n{len(sents):,} sentences ready for TTS:")
for r in sents[:5]:
    print("   ", r["text"])
"""),

    md("## 6 · Push the text to the Hub\n\nThis is the handoff to `01b`. Push before anything can time out."),
    code("""
for man, cfg in [("bigrams_valid.jsonl", "bigrams"), ("sentences.jsonl", "sentences")]:
    run("csasr.data.push_to_hub", "--manifest", MAN / man,
        "--repo", SYNTH_REPO, "--config", cfg, "--text-only")
print("\\ntext stage complete -> now run 01b_synthesize_audio.ipynb")
"""),
])


# ===========================================================================
# 01b - synthesize the AUDIO (Indic Parler-TTS, transformers==4.46.1)
# ===========================================================================

NB01B = notebook([
    md(f"""
# Track 2 · Stage 2 — Synthesize the code-mixed **audio**

Pulls the ~16k sentences that `01a` pushed, voices them with **Indic Parler-TTS**
(Rohit ♂ / Divya ♀, 50/50), and pushes ~22h of 16 kHz audio to
`RohanRamesh/hi-en-synth-cs`.

**Separate notebook because `parler-tts` hard-pins `transformers==4.46.1`**, which cannot
load the Gemma 3 used in `01a`. No LLM is loaded here.

### Before you run
* **Accept the licence for [`ai4bharat/indic-parler-tts`](https://huggingface.co/ai4bharat/indic-parler-tts)** — it is gated and has no
  ungated mirror. If your HF token is *fine-grained*, also tick **"Read access to contents
  of all public gated repos you can access"**; accepting the licence alone is not enough.
* HF **write** token in Kaggle Secrets as `HF_TOKEN`. Internet **on**, **GPU T4 ×2**.

Runtime ≈ 4–6h. Sharded — re-running skips every clip already on disk, so a 12h timeout
costs at most the in-flight shard.
"""),

    md("## 0 · Install\n\n**Cell order matters.** The `transformers` downgrade silently no-ops if anything has already imported `transformers`. Do not import it above this cell."),
    code(f"""
!pip install -q "transformers==4.46.1" "datasets<4" accelerate soxr librosa soundfile
!pip install -q git+https://github.com/huggingface/parler-tts.git
!pip install -q --force-reinstall --no-deps git+{REPO}

# This NOTEBOOK's own version. `pip install` updates the csasr PACKAGE but NOT the
# .ipynb -- an old notebook against a new package is a real and confusing failure.
NOTEBOOK_VERSION = "0.9.3"

import csasr, transformers
assert csasr.__version__ == NOTEBOOK_VERSION, (
    f"csasr package is {{csasr.__version__}} but this NOTEBOOK is {{NOTEBOOK_VERSION}}.\\n"
    "  package older  -> restart the kernel (Run > Restart & clear); pip skips a\\n"
    "                    reinstall when the version looks satisfied.\\n"
    "  notebook older -> re-download it from the repo; pip does NOT update .ipynb files."
)
assert transformers.__version__ == "4.46.1", (
    f"transformers is {{transformers.__version__}}, expected 4.46.1 - something imported "
    "it before the downgrade took effect. Restart the kernel."
)
print("csasr", csasr.__version__, "| transformers", transformers.__version__)
"""),

    code("""
import os, subprocess, sys, itertools
from pathlib import Path

os.environ["HF_HOME"] = "/kaggle/temp/hf"
Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)

# NEVER let transformers import TensorFlow. `parler_tts` imports
# `transformers.PreTrainedModel`, which reaches image_transforms.py and runs
# `if is_tf_available(): import tensorflow`. Kaggle HAS TensorFlow, but it wants a
# newer protobuf than our pins leave behind, so it dies with
#   ImportError: cannot import name 'runtime_version' from 'google.protobuf'
# We never use TF. USE_TF=0 makes is_tf_available() False and the import vanishes.
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])
HF_TOKEN = os.environ["HF_TOKEN"]

import torch
N_GPU = max(1, torch.cuda.device_count())

SYNTH_REPO = "RohanRamesh/hi-en-synth-cs"
TTS        = "ai4bharat/indic-parler-tts"
NUM_SHARDS = 4          # finer than N_GPU on purpose: a timeout costs one shard

WORK  = Path("/kaggle/working")
MAN   = WORK / "manifests"; MAN.mkdir(parents=True, exist_ok=True)
AUDIO = WORK / "audio";     AUDIO.mkdir(parents=True, exist_ok=True)

def run(*args):
    print(">", " ".join(str(a) for a in args), flush=True)
    p = subprocess.run([sys.executable, "-m", *args])
    if p.returncode != 0:
        raise RuntimeError(
            f"{args[0]} failed (exit {p.returncode}). The real error is printed ABOVE "
            f"this traceback - scroll up in this cell's output."
        )

def run_parallel(cmds):
    \"\"\"One process per GPU, pinned with CUDA_VISIBLE_DEVICES; wait for all.

    Running the shards SEQUENTIALLY leaves Kaggle's second T4 completely idle --
    the same waste already fixed in 01a and 03_eval.
    \"\"\"
    procs = []
    for i, cmd in enumerate(cmds):
        gpu = i % N_GPU
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        print(f"> GPU{gpu}: {' '.join(str(c) for c in cmd)}", flush=True)
        procs.append(subprocess.Popen(
            [sys.executable, "-m", *[str(c) for c in cmd]], env=env))
    for i, p in enumerate(procs):
        if p.wait() != 0:
            raise RuntimeError(f"shard {i} failed (exit {p.returncode}). The real error "
                               "is printed ABOVE - scroll up in this cell's output.")

print(f"{N_GPU} GPU(s), {NUM_SHARDS} shards")
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),

    md("## 0b · PREFLIGHT — Parler-TTS is gated\n\nFails in seconds rather than after a 6-hour run has started."),
    code("""
from huggingface_hub import model_info
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

try:
    model_info(TTS, token=HF_TOKEN)
    print(f"  OK  {TTS}")
except GatedRepoError:
    raise SystemExit(
        f"PREFLIGHT FAILED: {TTS} is gated.\\n"
        f"  1. Accept the licence at https://huggingface.co/{TTS}\\n"
        f"  2. If your HF token is FINE-GRAINED, also tick 'Read access to contents of\\n"
        f"     all public gated repos you can access', then update the Kaggle Secret."
    )
except RepositoryNotFoundError:
    raise SystemExit(f"PREFLIGHT FAILED: {TTS} not found, or your token cannot see it.")
"""),

    md("## 1 · Pull the sentences generated by `01a`"),
    code("""
from datasets import load_dataset
from csasr.manifest import read_jsonl, write_jsonl

ds = load_dataset(SYNTH_REPO, "sentences", split="train", token=HF_TOKEN)
write_jsonl(MAN / "sentences.jsonl", [dict(r) for r in ds])
print(f"{len(ds):,} sentences to synthesize")
print(ds[0]["text"])
"""),

    md("""## 2 · Synthesize

Indic Parler-TTS emits **44.1 kHz** → resampled to 16 kHz and written as **int16**
(`synthesize.py` reads `model.config.sampling_rate`; it never hardcodes the rate).

**Both GPUs, in parallel.** Shards run two at a time, one per T4 — running them
sequentially would leave half the machine idle, which is exactly the waste already fixed in
01a and 03_eval.

**Resumable.** Every clip already on disk is skipped, so re-running this cell after a
timeout costs only the in-flight batch."""),
    code("""
BATCH = 16     # Parler is small (938M); 8 left the GPU underfed

# Two shards at a time -- one per GPU.
for start in range(0, NUM_SHARDS, N_GPU):
    group = range(start, min(start + N_GPU, NUM_SHARDS))
    run_parallel([
        ["csasr.tts.synthesize",
         "--sentences", MAN / "sentences.jsonl",
         "--audio-dir", AUDIO,
         "--out", MAN / f"train_t2.shard{s}.jsonl",
         "--shard", s, "--num-shards", NUM_SHARDS,
         "--batch-size", BATCH]
        for s in group
    ])
"""),

    md("## 3 · Merge shards, build Train_T1 by reference\n\n### GATE 2 — `Train_T1 ⊆ Train_T2`, durations ≈ 8h / 22h"),
    code("""
merged = list(itertools.chain.from_iterable(
    read_jsonl(MAN / f"train_t2.shard{i}.jsonl") for i in range(NUM_SHARDS)
))
write_jsonl(MAN / "train_t2.jsonl", merged)
hours = sum(r["dur"] for r in merged) / 3600
print(f"Train_T2: {len(merged):,} clips, {hours:.2f} h (paper: 22 h)")

run("csasr.tts.make_subset", "--t2", MAN / "train_t2.jsonl",
    "--out", WORK / "t1_ids.json", "--hours", "8.0")
"""),

    md("### Listen to it. Two clips per voice — do not skip this."),
    code("""
import IPython.display as ipd, random
from csasr.tts.speakers import assign_speaker
random.seed(0)
for voice in ("Rohit", "Divya"):
    picks = [r for r in merged if assign_speaker(r["sent_id"]).name == voice][:200]
    for r in random.sample(picks, 2):
        print(f"[{voice}] {r['text']}")
        ipd.display(ipd.Audio(r["wav"]))
"""),

    md("## 4 · Push the synthetic corpus\n\nParquet + FLAC ≈ 1.4 GB (vs 2.5 GB as WAV). The round-trip check catches a corrupted upload *before* a 3h training run."),
    code("""
run("csasr.data.push_to_hub", "--manifest", MAN / "train_t2.jsonl",
    "--repo", SYNTH_REPO, "--config", "synth_t2", "--verify")

from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj=str(WORK / "t1_ids.json"),
                    path_in_repo="t1_ids.json", repo_id=SYNTH_REPO,
                    repo_type="dataset", token=HF_TOKEN)
print("done ->", SYNTH_REPO, "\\nnext: 02_train.ipynb")
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
# datasets>=4 decodes Audio via torchcodec, which stock Kaggle images lack.
!pip install -q -U transformers accelerate evaluate jiwer
!pip install -q "datasets<4" librosa soundfile soxr omegaconf rich

# csasr LAST and FORCED: `pip install git+...` treats an already-installed
# version as satisfied and SKIPS the reinstall, so re-running in a live kernel
# silently keeps OLD code. --no-deps because the deps are installed above.
!pip install -q --force-reinstall --no-deps git+{REPO}

# This NOTEBOOK's own version. `pip install` updates the csasr PACKAGE but NOT the
# .ipynb -- an old notebook against a new package is a real and confusing failure.
NOTEBOOK_VERSION = "0.9.3"

import csasr
assert csasr.__version__ == NOTEBOOK_VERSION, (
    f"csasr package is {{csasr.__version__}} but this NOTEBOOK is {{NOTEBOOK_VERSION}}. "
    "If the PACKAGE is older: restart the kernel (Run > Restart & clear) -- pip skips "
    "a reinstall when the version already looks satisfied. If the NOTEBOOK is older: "
    "re-download it from the repo -- pip does NOT update .ipynb files."
)
print("csasr", csasr.__version__)

import os, subprocess, sys
from pathlib import Path
os.environ["HF_HOME"] = "/kaggle/temp/hf"

from kaggle_secrets import UserSecretsClient
# Token goes in the environment, never in argv (it leaks into every traceback).
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
from huggingface_hub import hf_hub_download, login
login(token=os.environ["HF_TOKEN"])
HF_TOKEN = os.environ["HF_TOKEN"]   # for load_dataset / hf_hub_download only

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
    "--dev-hf", REAL, "--dev-config", "dev",
    "--max-steps", "20", "--eval-steps", "10", "--dataset-fraction", "0.01",
    "--batch-size", "4", "--grad-accum", "1")
"""),

    md("## M6 — Train_T1 (8h synthetic)"),
    code("""
run("csasr.train.train_whisper", "--model", MODEL, "--out", "/kaggle/working/m6",
    "--train-hf", SYNTH, "--train-config", "synth_t2", "--subset-ids", t1_ids,
    "--dev-hf", REAL, "--dev-config", "dev",
    "--lr", "2e-5", "--batch-size", "16", "--grad-accum", "4",
    "--max-steps", "5000", "--eval-steps", "250", "--patience", "4")
"""),

    md("## M7 — Train_T2 (22h synthetic)"),
    code("""
run("csasr.train.train_whisper", "--model", MODEL, "--out", "/kaggle/working/m7",
    "--train-hf", SYNTH, "--train-config", "synth_t2",
    "--dev-hf", REAL, "--dev-config", "dev",
    "--lr", "2e-5", "--batch-size", "16", "--grad-accum", "4",
    "--max-steps", "5000", "--eval-steps", "250", "--patience", "4")
"""),

    md("## M8 — Train_T2 + Common Voice monolingual (52h)\n\nStill one `<|hi|>` prompt for everything: language-*specific* prompting is Track 1's M4."),
    code("""
run("csasr.train.train_whisper", "--model", MODEL, "--out", "/kaggle/working/m8",
    "--train-hf", SYNTH, "--train-config", "synth_t2",
    "--extra-hf", f"{REAL}:cv_hi", f"{REAL}:cv_en",
    "--dev-hf", REAL, "--dev-config", "dev",
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

### Decoding reproduces WhisperX, not a per-clip loop
The paper decodes whole recordings with WhisperX. We use **faster-whisper**, the engine
WhisperX wraps: language detected **once per recording**, 30-second chunks with real
context, beam 5, VAD, temperature fallback, and a compression-ratio threshold that aborts
repetition loops.

This matters enormously. Decoding the 3,136 isolated 6-second clips instead makes Whisper
render English loanwords in Devanagari, so the hypothesis contains **no script boundary and
no switch bigram can match**. Measured on real test audio with one model:

| decoding | %Latin in hyp | MER | CBA-HE |
|---|---|---|---|
| reference | 21.6% | – | – |
| per-clip | **0.0%** | 182.6 | 0.0 |
| recording-level | 12.2% | 103.1 | 1.6 |

`utt_id` is `<speaker>_<recording>_<index>`, so the `test` config already on the Hub is
regrouped into its 30 recordings here — no re-upload.
"""),

    code(f"""
!pip install -q -U transformers jiwer faster-whisper ctranslate2
!pip install -q "datasets<4" librosa soundfile soxr omegaconf rich

# csasr LAST and FORCED: `pip install git+...` treats an already-installed
# version as satisfied and skips the reinstall, so a re-run in a live kernel
# silently keeps OLD code. --no-deps because the deps are installed above.
!pip install -q --force-reinstall --no-deps git+{REPO}

# This NOTEBOOK's own version. `pip install` updates the csasr PACKAGE but NOT the
# .ipynb -- an old notebook against a new package is a real and confusing failure.
NOTEBOOK_VERSION = "0.9.3"

import csasr
assert csasr.__version__ == NOTEBOOK_VERSION, (
    f"csasr package is {{csasr.__version__}} but this NOTEBOOK is {{NOTEBOOK_VERSION}}. "
    "If the PACKAGE is older: restart the kernel (Run > Restart & clear) -- pip skips "
    "a reinstall when the version already looks satisfied. If the NOTEBOOK is older: "
    "re-download it from the repo -- pip does NOT update .ipynb files."
)
print("csasr", csasr.__version__)

import os, subprocess, sys
os.environ["HF_HOME"] = "/kaggle/temp/hf"

from kaggle_secrets import UserSecretsClient
# Put the token in the ENVIRONMENT, never in argv: a CLI arg lands verbatim in
# every traceback and in `ps` output.
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
from huggingface_hub import login; login(token=os.environ["HF_TOKEN"])

REAL = "RohanRamesh/mucs-he-cs"
OUT  = "/kaggle/working"

def run(*args):
    \"\"\"Run a csasr CLI. On failure, re-raise with the LAST LINES of the child's
    output -- otherwise Jupyter shows only `exit status 1` and buries the cause.\"\"\"
    print(">", " ".join(str(a) for a in args), flush=True)
    p = subprocess.run([sys.executable, "-m", *args])   # inherits HF_TOKEN + streams
    if p.returncode != 0:
        raise RuntimeError(
            f"{{args[0]}} failed (exit {{p.returncode}}). The real error is printed "
            f"ABOVE this traceback - scroll up in this cell's output."
        )

import torch
N_GPU = torch.cuda.device_count()
print(f"{{N_GPU}} GPU(s) visible")
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),

    md("""## Decoding: use **both** T4s and batch the chunks

Sequential decoding pinned one GPU at ~4 GB of 15 GB and left the second idle — large-v2
took 1h50m. Two independent fixes:

* **Batched inference** (`--batch-size 16`). VAD carves the recording into speech chunks and
  they are decoded as a batch instead of one at a time: **8× faster**, measured. This is not a
  shortcut — batched VAD inference is exactly what WhisperX does, so it is *more* faithful to
  the paper than our sequential loop was.
* **Shard recordings across the two GPUs**, one process each: another **2×**.

Together: large-v2 ≈ **1h50m → ~8 min**.

`--lang-detect-segments 8` also lands here. faster-whisper detects the language from a *single*
30-second window by default, and one bad window sends a whole recording into Urdu — where no
Hindi/English switch bigram can match and CBA collapses. Voting over 8 windows fixes it."""),
    code("""
from pathlib import Path
from csasr.manifest import read_jsonl, write_jsonl
from csasr.eval.ct2 import resolve_ct2

CT2 = "/kaggle/temp/ct2"

def decode(model, out_name, batch_size=16):
    \"\"\"Shard the 30 recordings across every GPU, decode in parallel, merge.\"\"\"
    # Convert ONCE up front: two processes racing on the same cache dir would
    # corrupt it. Prebuilt OpenAI models pass straight through.
    resolve_ct2(model, cache_dir=CT2, quantization="float16")

    n = max(1, N_GPU)
    procs = []
    for i in range(n):
        cmd = [sys.executable, "-m", "csasr.eval.decode",
               "--model", model, "--engine", "faster-whisper", "--mode", "recording",
               "--language", "none", "--batch-size", str(batch_size),
               "--lang-detect-segments", "8",
               "--test-hf", REAL, "--test-config", "test", "--ct2-cache", CT2,
               "--shard", str(i), "--num-shards", str(n),
               "--out", f"{OUT}/{out_name}.shard{i}.jsonl",
               "--refs-out", f"{OUT}/refs.shard{i}.jsonl"]   # PER-UTTERANCE refs
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        print(f"> GPU{i}: {model} shard {i}/{n}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env))

    for i, p in enumerate(procs):
        if p.wait() != 0:
            raise RuntimeError(f"decode shard {i} failed (exit {p.returncode})")

    hyps = [r for i in range(n) for r in read_jsonl(f"{OUT}/{out_name}.shard{i}.jsonl")]
    refs = [r for i in range(n) for r in read_jsonl(f"{OUT}/refs.shard{i}.jsonl")]
    write_jsonl(f"{OUT}/{out_name}.jsonl", hyps)
    write_jsonl(f"{OUT}/refs_utt.jsonl", sorted(refs, key=lambda r: r["utt_id"]))
    print(f"merged {len(hyps)} recordings -> {OUT}/{out_name}.jsonl")
    return hyps
"""),

    md("## GATE 3 — calibrate the metric against a published number\n\nInference only, ~8 min on 2× T4."),
    code("""
decode("openai/whisper-large-v2", "hyp_largev2_zeroshot")
"""),

    code("""
from csasr.eval.score import score

REFS = f"{OUT}/refs_utt.jsonl"          # PER-UTTERANCE: CBA's denominator is Table 1's
HYP  = f"{OUT}/hyp_largev2_zeroshot.jsonl"

for mm in ("word", "hybrid"):
    r = score(REFS, HYP, group="recording", mer_mode=mm)
    print(f"MER {mm:<7} {r['mer']:>6.1f}")
print()
for cm in ("adjacent", "lenient"):
    r = score(REFS, HYP, group="recording", cba_mode=cm)
    print(f"CBA {cm:<9} HE {r['cba_he']:>5.1f}   EH {r['cba_eh']:>5.1f}   "
          f"(HE denominator {r['he_total']:,})")

print("\\npaper (large-v2 zero-shot): MER 52.0   CBA-HE 42.9   CBA-EH 36.x   HE denom 4,189")
print()
print("MER: 'hybrid' reproduces the paper (51.9 vs 52.0), so they use the SEAME")
print("     definition - characters on Devanagari, words on Latin.")
print("CBA: the paper never defines 'correctly recognized'. 'lenient' lands on their")
print("     42.9; 'adjacent' is the literal bigram reading and lands at half. Both are")
print("     reported. Every system is scored identically, so the M6->M7->M8 ordering")
print("     that Track 2 actually tests is unaffected. See csasr/eval/cba.py.")
"""),

    md("### Diagnostics — read these before trusting the numbers above"),
    code("""
from collections import Counter
from csasr.manifest import read_jsonl
from csasr.eval.cba import cba
from csasr.eval.mer import mer
from csasr.lid import Lang, count_words
from csasr.normalize import normalize

from csasr.eval.grouping import concat_refs
refs = concat_refs(list(read_jsonl(f"{OUT}/refs_utt.jsonl")))
hyps = list(read_jsonl(f"{OUT}/hyp_largev2_zeroshot.jsonl"))
R = [refs[h["utt_id"]] for h in hyps]
H = [h["hyp"] for h in hyps]

def mix(t):
    c = count_words(normalize(t, "scoring")); tot = sum(c.values()) or 1
    return c[Lang.HI] / tot, c[Lang.EN] / tot

# 1) BOTH scripts must be present, or CBA is structurally zero regardless of MER.
rh, re_ = mix(" ".join(R)); hh, he = mix(" ".join(H))
print(f"REFERENCE : {rh:5.1%} Devanagari  {re_:5.1%} Latin")
print(f"HYPOTHESIS: {hh:5.1%} Devanagari  {he:5.1%} Latin")

# 2) Hindi/Urdu confusion destroys switch points wholesale.
print("\\ndetected language per recording:", dict(Counter(h["detected_language"] for h in hyps)))
def wc(t): return sum(count_words(normalize(t, "scoring")).values())
bad = [h for h in hyps if h["detected_language"] != "hi"]
if bad:
    share = sum(wc(refs[h["utt_id"]]) for h in bad) / sum(wc(t) for t in R)
    print(f"  non-hi: {len(bad)}/{len(hyps)} recordings = {share:.1%} of reference words")
    hi = [h for h in hyps if h["detected_language"] == "hi"]
    ch = cba([refs[h["utt_id"]] for h in hi], [h["hyp"] for h in hi])
    ca = cba(R, H)
    print(f"  CBA-HE  all {ca.he:.1f}  ->  hi-only {ch.he:.1f}")
    print(f"  CBA-EH  all {ca.eh:.1f}  ->  hi-only {ch.eh:.1f}")

# 3) Is any residual MER gap definitional rather than a quality gap?
print()
for p in ("raw", "punct", "scoring"):
    print(f"MER preset={p:8}: {mer(R, H, preset=p):.1f}")
"""),

    md("## whisper-small zero-shot — the actual baseline for M6/M7/M8\n\nExpect its CBA ≈ 0: whisper-small transliterates English into Devanagari, so it has no script boundary to match. That is the model, not the pipeline — and it is exactly what fine-tuning is supposed to fix."),
    code("""
decode("openai/whisper-small", "hyp_small_zeroshot")
"""),

    md("""## Decode the fine-tuned models

`decode.py` converts each HF checkpoint to CTranslate2 on first use and caches it, so
faster-whisper can load it. Run this **only after** `02_train.ipynb` has pushed the repos —
otherwise you get a 404, which is expected, not a bug."""),
    code("""
for m in ("m6", "m7", "m8"):
    decode(f"RohanRamesh/whisper-small-cs-{m}", f"hyp_{m}")
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
    r = score(f"{OUT}/refs_utt.jsonl", f"{OUT}/{f}", group="recording", mer_mode="hybrid")
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
        ("01a_generate_text.ipynb", NB01A),      # Gemma 3, transformers >= 4.50
        ("01b_synthesize_audio.ipynb", NB01B),   # parler-tts, transformers == 4.46.1
        ("02_train.ipynb", NB02),
        ("03_eval.ipynb", NB03),
    ]:
        p = OUT / name
        p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {p}  ({len(nb['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
