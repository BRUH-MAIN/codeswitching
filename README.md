# Hindi-English Code-Switched ASR — Track 2 Replication

Replication of **Track 2** of Biswas et al., *"Adapting Whisper for low-resource
Hindi-English Code-Mix speech with on-the-fly Augmentation & LLM-Synthesised Data"*
(Interspeech 2025). [`paper/biswas25_interspeech.pdf`](paper/biswas25_interspeech.pdf)

Track 2's claim: **you do not need real code-switched training audio.** An LLM
few-shot-prompted for Hindi-English bigrams, expanded into sentences and voiced
by a TTS model, is enough to adapt Whisper to a code-switching domain.

| Model | Training data | Paper MER ↓ | CBA-HE ↑ | CBA-EH ↑ |
|---|---|---|---|---|
| Large-V2 zero-shot | — | 52.0 | 42.9 | 36.x |
| **M6** | Train_T1 (8h synthetic) | 48.2 | 45.5 | 45.x |
| **M7** | Train_T2 (22h synthetic) | 40.8 | 52.3 | 55.x |
| **M8** | Train_T2 + Common Voice mono | 39.2 | 55.4 | 56.x |

---

## Pipeline

```
LOCAL (CPU)                          KAGGLE (T4)                        HF HUB
-----------                          -----------                        ------
OpenSLR 104  ─┐
              ├─► prepare_mucs ─► GATE 0 ─► push ──────────────────► mucs-he-cs
Common Voice ─┘   (text-only train,                                  (train_text,
 (streamed)        4h dev, 5.2h test)                                 dev, test,
                                                                      cv_hi, cv_en)
                                          01_generate_dataset.ipynb        │
                                          ├─ gen_bigrams   (Llama-3.1-8B)  │
                                          ├─ filter_bigrams                │
                                          ├─ gen_sentences ─► GATE 1       │
                                          ├─ synthesize    (Indic Parler)  │
                                          └─ make_subset   ─► GATE 2 ──► hi-en-synth-cs
                                                                       (synth_t2,
                                          02_train.ipynb                t1_ids.json)
                                          └─ whisper-small full FT ──► whisper-small-cs-m{6,7,8}

                                          03_eval.ipynb
                                          ├─ GATE 3 (large-v2 zero-shot ≈ 52.0 MER)
                                          └─ decode (language=None)
score.py (CPU) ◄──────────────────────────────────────────────── hypotheses
     └─► results/table2.md
```

**Track 2 never trains on real code-switched audio.** The 89.8h MUCS train split is
used only for (a) its transcripts, as few-shot exemplars, and (b) a 4h dev slice for
checkpoint selection. The other ~86h are downloaded, read, and deleted — they never
reach the Hub.

---

## The gates

Each is cheap relative to what it protects. **Run them in this order.**

| Gate | What it checks | Cost |
|---|---|---|
| **0** | `verify_table1.py` reproduces the paper's Table 1 word and bigram counts | CPU, seconds |
| **1** | Bigram yield ratios track the paper (13.3% dedup, 92.3% filter) | free (end of Stage 1) |
| **2** | `Train_T1 ⊆ Train_T2`; durations ≈ 8h / 22h; spot-listen | free (end of Stage 2) |
| **3** | `whisper-large-v2` zero-shot ≈ **52.0 MER / 42.9 CBA-HE** | ~40 min on a T4 |

Gate 3 is inference-only and validates decoding, normalization, language ID, and
both metrics against a *published* number. It is not our baseline — it is the
calibration of the measuring instrument. **Run it before generating anything.**

### Gate 0 status: PASSED

**Test split** — the split every reported metric (MER, CBA) is computed on:

```
metric              got      paper   rel err  status
words_h          28,219     28,215     0.01%  PASS
words_e           9,152      9,627     4.93%  info
total            37,557     37,842     0.75%  PASS
he_cs             4,125      4,189     1.53%  PASS
eh_cs             5,058      5,176     2.28%  PASS
hours              5.18       5.20     0.42%  PASS
```

`words_e` runs ~5% low, and it is *explained, not a bug*: the paper's Total (37,842)
exceeds this corpus's entire raw whitespace-token count (37,611), so their tokenizer
splits tokens ours does not. The 186 bare digit tokens (`334`, `1204`, …) are the
likely cause — expanded to words they would close most of the gap. We keep digits as
`OTHER` because the paper's own bigram filter requires each bigram to contain "both
Hindi and English characters", and a digit has neither.

**Train split — does not reproduce, and no tokenization can make it.**

```
metric              got      paper   rel err
hours             89.55      89.80     0.27%  PASS
words_h         468,104    445,762     5.01%  info
words_e         159,330    160,694     0.85%  info
he_cs            77,366     85,761     9.79%  info
eh_cs            89,019     90,802     1.96%  info
```

We searched the tokenizer space (punctuation stripping × script-boundary splitting ×
digit handling × de-duplication). The best train fit is still 3.75% off, and the
**maximum HE bigram count any variant produces is 80,860 against the paper's 85,761**
— reached only by the most switch-generating setting (digits counted as English).
Splitting tokens can only *add* adjacencies; it cannot invent Hindi→English pairs
that are absent from the text. So the released OpenSLR 104 transcripts contain fewer
Hindi→English adjacencies than the paper's train row implies. The
[challenge page](https://navana-tech.github.io/MUCS2021/data.html) confirms the
Feb 2021 revision is the only release and offers no separate transcript download.

This costs the replication nothing: **Track 2 uses the train split only for few-shot
exemplar text and a 4h dev slice**, neither of which depends on its exact word counts.
Gate 0 therefore asserts the paper's counts on `test` and only integrity invariants
(duration, both languages present, no surviving `MIXED` tokens, code-switching present)
on `train`.

---

### The eval path, verified end-to-end on real audio

`whisper-tiny` zero-shot with `language=None` on 16 real test utterances. It
auto-detects English and *transliterates* the Hindi instead of transcribing it —
the exact failure the paper describes:

| Reference | Hypothesis |
|---|---|
| `प्रस्तुति document` | "**prostitute** document" |
| `बुनियादी formatting` | "buneadee formatting" |
| `इस spoken` | "is spoken" |

`MER 125.0` (insertions can push an error rate past 100%), `CBA-HE 0.0`,
`CBA-EH 0.0` — all 61 switch points destroyed. Scoring sanity on the full 3,136
utterances: a perfect hypothesis gives MER 0 / CBA 100, an empty one MER 100 /
CBA 0, and deleting every Hindi word gives MER 75.3 / CBA 0.

---

## Three things that would have silently corrupted this

**1. Whisper's `BasicTextNormalizer` shreds Devanagari.**
It replaces every character in Unicode category `M*` with a space, and Devanagari
vowel signs, the virama, and the nukta are all combining marks:

```
दस्तावेज़  ->  ['दस', 'त', 'व', 'ज']      # one word becomes four (+65.7% Hindi words)
```

[`normalize.py`](src/csasr/normalize.py) strips only `P*` and `S*`. The danda `।` is
`Po`, so it is still removed correctly. Gate 0 keeps `whisper_basic` around purely as
a regression guard that must fail.

**2. The danda lives inside the Devanagari Unicode block.**
`।` is U+0964, between U+0900 and U+097F. A naive block-range test classifies bare
punctuation as a Hindi word. [`lid.py`](src/csasr/lid.py) additionally requires
category `L*`/`M*`.

**3. CBA needs multiset semantics.**
The paper counts 4,189 HE bigram *tokens* against 2,347 unique *types*. A set-based
intersection would under-count the denominator and inflate the score.

We also split dropped-space typos (`दायाँclick`, `करेंspoken`) at the script boundary —
they hide real switch points inside one whitespace token, and splitting them brings
Hindi word counts to within 4 of the paper's.

**4. `datasets >= 4.0` cannot decode audio without `torchcodec`,** which is absent
from stock Kaggle images. Pinned to the `3.x` line, which decodes via
`librosa` + `soundfile`. This would have failed *during training*, not at import.

---

## Deviations from the paper

| # | Paper | Here | Why |
|---|---|---|---|
| **D1** | Llama-3.3-70B-Instruct | **Llama-3.1-8B-Instruct**, NF4 | 70B is 141 GB in bf16. Nearest same-family model with official Hindi support. |
| **D2** | whisper-large-v2 (1.54B) | **whisper-small** (244M) | Fits a T4. *(There is no `whisper-small-v2`; `v2`/`v3` exist only for `large`.)* |
| **D3** | Full FT, AdamW, lr 2e-5, batch 64 | **unchanged** | A consequence of D2: whisper-small full FT fits, so no LoRA substitution is needed. |
| **D4** | WhisperX, `language=None` | `transformers.generate(language=None)` | WhisperX needs a CTranslate2 conversion; same decode condition. |
| **D5** | Common Voice (official) | `fsicoli/common_voice_17_0` | Mozilla moved CV to the Data Collective in Oct 2025; the HF repo is now an empty stub. |
| **D6** | 4h dev from real train | same, **plus** a synthetic-only dev logged alongside | The paper's dev set is real in-domain audio, a mild leak against its "no real data" claim. |

D1 and D2 both cost quality, so **absolute MER will not match Table 2**. The claim
under test is the **ordering and relative gains**: `zero-shot > M6 > M7 > M8`.

Also note: the paper's few-shot examples were mangled by PDF extraction into
`pr-tEt document; bEnyAdF formatting; isspoken`. All three are recoverable verbatim
from the first line of the MUCS test transcripts — `pr-tEt` is **प्रस्तुति**
("presentation"), which occurs in the corpus, not प्रतीत, which never does. See
[`prompts.py`](src/csasr/llm/prompts.py).

---

## Disk

Audio is 16 kHz / 16-bit / mono = **115.2 MB per hour**.

| Artifact | Hours | Size |
|---|---|---|
| Train_T2 synthetic, WAV (on Kaggle) | 22 | **2.53 GB** |
| Train_T2 as FLAC (on the Hub) | 22 | ~1.39 GB |
| Train_T1 | 8 | **0 extra** — `t1_ids.json`, by reference |

Never let Parler-TTS's float32 output hit disk: that would be 7 GB. Write int16.

Local peak ≈ 25 GB; **≈ 2.1 GB after deleting the real training audio.**
Kaggle NB 01 peaks ≈ 12.6 GB — set `HF_HOME=/kaggle/temp/hf` so 8.7 GB of model
weights stay out of the 20 GB `/kaggle/working` budget.

> **Common Voice English `train` is 45 GB of audio across 28 shards with a 363 MB
> TSV.** `prepare_cv.py` never touches it: it walks the splits smallest-first
> (`en/dev` alone holds ~50h with a 4.9 MB TSV) and streams the tars over HTTP,
> aborting the moment the 15h target is met. It also bypasses the mirror's
> loading script, which declares a `variant` column its own TSVs lack.

---

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[local,dev]"
pytest -q
```

Local work is CPU-only — no torch needed. The GPU stages run on Kaggle.

### Stage 0 (local)

```bash
python -m csasr.data.download --splits test train
python -m csasr.data.prepare_mucs --root data/mucs/test --out manifests/mucs_test.jsonl --cut-audio
python -m csasr.data.prepare_mucs --root data/mucs/train --out manifests/mucs_train.jsonl \
       --dev-hours 4 --dev-out manifests/mucs_dev.jsonl
python scripts/verify_table1.py            # <- GATE 0, blocks everything
python -m csasr.data.prepare_cv --langs hi en --hours 15
python -m csasr.data.push_to_hub --manifest manifests/mucs_train.jsonl \
       --repo RohanRamesh/mucs-he-cs --config train_text --text-only --verify
```

### Stages 1-4 (Kaggle)

Upload [`notebooks/`](notebooks/). Requirements: GPU **T4 x2**, internet **on**, and
an HF **write** token in Kaggle Secrets as `HF_TOKEN`. Accept the licences for
`ai4bharat/indic-parler-tts` and `meta-llama/Llama-3.1-8B-Instruct` first.

Regenerate the notebooks with `python scripts/build_notebooks.py` — they are thin
wrappers that `pip install` this package and call into `csasr`, so all logic stays
unit-tested rather than buried in cells.

### Compute budget

| | |
|---|---|
| Local | ~2-3h, CPU + network |
| Kaggle | ~13-15 GPU-h, inside the 30 h/week quota |

NB 01 (LLM ~2h + TTS ~4-6h) is the only notebook near the 12h session cap. It is
sharded, Hub-checkpointed, and has a `STAGE` flag — every stage is idempotent and
resumable, because a stage that cannot resume cannot finish.

---

## Layout

```
src/csasr/
  lid.py            word-level script language ID   <- everything rests on this
  normalize.py      punctuation-only normalization  <- NOT BasicTextNormalizer
  manifest.py       JSONL contract between stages
  data/             download, prepare_mucs, prepare_cv, push_to_hub, loaders
  llm/              backend, cache (resumable), prompts (verbatim), gen_*/filter_*
  tts/              speakers (Rohit/Divya), synthesize (sharded), make_subset
  train/            collator, train_whisper (full FT)
  eval/             decode (language=None), mer, cba (multiset), score
scripts/
  verify_table1.py  GATE 0
  build_notebooks.py
notebooks/          01_generate_dataset, 02_train, 03_eval  (Kaggle)
configs/            data, llm, tts, train_m{6,7,8}
```

## Licence / attribution

MUCS 2021 (OpenSLR 104) is **CC BY-SA 4.0**. Derived text and audio inherit
share-alike. `RohanRamesh/mucs-he-cs` is kept private as a derivative of a challenge
corpus.
