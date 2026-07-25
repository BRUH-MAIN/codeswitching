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
                                          01a_generate_text.ipynb          │
                                          │  transformers>=4.50 + Gemma-3-4B│
                                          ├─ gen_bigrams                    │
                                          ├─ filter_bigrams                │
                                          └─ gen_sentences ─► GATE 1 ──────┼─► hi-en-synth-cs
                                                                           │   (bigrams,
                                          01b_synthesize_audio.ipynb       │    sentences)
                                          │  transformers==4.46.1 + parler │
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
| **1.5** | `audit_sentences.py` — label leaks, script mix, switch density, domain overlap, projected TTS hours | CPU, seconds |
| **3** | `whisper-large-v2` zero-shot ≈ **52.0 MER / 42.9 CBA-HE** | ~15 min on a T4 |

Gate 3 is inference-only and validates decoding, normalization, language ID, and
both metrics against a *published* number. It is not our baseline — it is the
calibration of the measuring instrument. **Run it before generating anything.**

### Gate 3 caught a real bug, which is the whole point

The first Gate 3 run returned **MER 75.6 / CBA-HE 14.7** against the paper's
52.0 / 42.9. Since CBA does not depend on the MER definition, a 3× miss there
meant the *hypotheses* were wrong, not the metric.

Cause: we were decoding the 3,136 isolated 6-second clips, while the paper
decodes whole recordings with WhisperX. Without surrounding context Whisper
renders English loanwords in Devanagari, so a hypothesis contains **no script
boundary and no switch bigram can match**. Per-utterance language detection also
flipped to Urdu on some clips, and short zero-padded inputs triggered `तो तो तो …`
repetition loops. Measured on real test audio, same model, same audio span:

| decoding | %Latin in hypothesis | MER | CBA-HE |
|---|---|---|---|
| reference | 21.6% | — | — |
| per-clip (wrong) | **0.0%** | 182.6 | 0.0 |
| recording-level | 12.2% | 103.1 | 1.6 |

`decode.py` now defaults to `--engine faster-whisper --mode recording`:
faster-whisper is the engine WhisperX wraps, so we get one language detection per
recording, 30-second chunks with context, beam 5, VAD, temperature fallback, and
a compression-ratio threshold that aborts repetition loops. `utt_id` encodes the
recording, so the `test` config already on the Hub is regrouped in place — no
re-upload. Scoring is recording-level (`score.py --group recording`, or use the
refs file `decode.py` emits): MER is a corpus-level ratio, so concatenation is
equivalent, and CBA correctly *gains* the bigrams that straddle segment joins.

This cost one 40-minute inference run and saved 6–8 hours of TTS plus 5 hours of
training measured against a broken ruler.

### Decoding speed: use the whole machine

The first working Gate 3 run took **1h50m** for large-v2 while pinning one GPU at 4 GB of
15 GB and leaving Kaggle's second T4 completely idle. Two independent fixes, both in
`decode.py`:

| lever | effect |
|---|---|
| `--batch-size 16` (VAD chunks decoded as a batch, not one at a time) | **8×** — measured, whisper-small: 253s → 32s on 181s of audio |
| `--num-shards N` + `CUDA_VISIBLE_DEVICES` (one process per GPU) | **2×** on Kaggle's 2× T4 |

Together large-v2 drops to roughly **8 minutes**.

Batching is not a shortcut: **batched VAD inference is precisely what WhisperX does** — it is
the feature that makes WhisperX "70× realtime". Our original sequential loop was the *less*
faithful option. It does cost a little accuracy (whisper-small: MER 75.3 → 80.6), and the
penalty shrinks as the model gets stronger (whisper-tiny, which hallucinates freely, was far
worse). Every system is decoded identically, so the M6→M7→M8 comparison is unaffected. Pass
`--batch-size 1` to fall back to sequential.

`--lang-detect-segments 8` ships alongside. faster-whisper detects the language from a
**single 30-second window** by default; Whisper confuses Hindi with Urdu constantly (same
spoken language, different script), and one bad window sends an entire recording into
Perso-Arabic, where no Hindi↔English switch bigram can match and CBA collapses. Voting over
8 windows fixes it. The first Gate 3 run reported `detected languages: {'ur', 'hi'}`, which
is the likely cause of its CBA-HE shortfall (29.2 against the paper's 42.9).

### GATE 3 result — MER reproduces; CBA is ambiguous in the paper

Run on 2× T4, large-v2 zero-shot, decoded recording-level with faster-whisper:

| metric | ours | paper |
|---|---|---|
| MER (word) | 54.8 | — |
| **MER (hybrid)** | **51.9** | **52.0** |
| CBA-HE (`adjacent`) | 20.1 | 42.9 |
| **CBA-HE (`lenient`)** | **43.8** | **42.9** |
| CBA-EH (`lenient`) | 40.9 | ~36 |
| HE denominator | 4,125 | 4,189 |

**The MER definition is settled.** `hybrid` — characters on Devanagari, words on Latin,
the SEAME rule from Zhang et al. — lands on 51.9 against a published 52.0. The `word`
reading does not. That ambiguity is closed empirically.

**A real bug was found and fixed.** Our CBA denominator was computed from *concatenated*
recordings, which fabricated ~1,000 HE bigrams straddling segment joins that no reference
utterance contains — inflating the denominator 23% and halving CBA. The denominator must
come from the reference *utterances* (Table 1's count). `cba_grouped()` now does this,
applying the multiset cap once per recording rather than per utterance.

**The CBA matching rule is genuinely unrecoverable from the paper.** "Correctly recognized"
is never defined. With hypotheses whose MER already matches theirs to 0.2%:

* `adjacent` (both words correct *and* adjacent — the literal reading of "bigram") → 20.1
* `lenient` (both words recognized somewhere) → 43.8, essentially their 42.9
* edit-distance alignment → 33.6 / 35.0, but it inverts the HE > EH ordering that holds in
  every row of their Table 2, so it is unlikely to be theirs

We report **both** and default to `adjacent`. This does not affect the experiment: every
system is scored identically, and the M6 → M7 → M8 ordering Track 2 actually tests is
preserved under either rule. Never mix modes across systems.

**What is verified, and what is not.** Recording-level decoding demonstrably fixes
MER (182.6 → 65.6 on whisper-small, same audio), stabilises language detection to
a single `hi` per recording, and eliminates the repetition loops. Whether it
restores CBA-HE to the paper's 42.9 has **not** been verified — only a
`whisper-large-v2` run can settle that, and it must be run on a GPU.

whisper-small is a poor proxy for the script question: it transliterates every
English word into Devanagari (`impress` → `इंप्रस`), so its CBA is structurally
zero under *every* decoder setting. An ablation over `vad_filter` ×
`condition_on_previous_text` moved %Latin only between 0.7% and 2.9%, never CBA.
large-v2 does emit Latin — it scored CBA-HE 14.7 even under the broken per-clip
protocol — so its zero-shot CBA should rise substantially once context is
restored. Expect our own `whisper-small` zero-shot baseline row to have CBA ≈ 0;
that is the model, not the pipeline, and fine-tuning on mixed-script targets is
precisely what fixes it.

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

**5. Gemma 4 cannot share a process with Parler-TTS, and the T4 has no bf16.**

* `parler-tts` hard-pins `transformers==4.46.1`; **Gemma 4 needs `>=5.5`** (its config says
  `transformers_version: 5.5.0.dev0`). Irreconcilable — hence `01a_generate_text` (Gemma 4)
  and `01b_synthesize_audio` (Parler-TTS) are **separate notebooks**, with the Hub as the
  checkpoint between them.
* **The live risk:** Gemma 4's config declares `torch_dtype: bfloat16`, and the T4 is Turing
  — it has **no bf16**. fp16 activations can exceed 65,504 and go non-finite.
  [`backend.py`](src/csasr/llm/backend.py) probes the logits after loading and transparently
  reloads with float32 compute if they are NaN/inf. `01a`'s smoke-test cell surfaces this in
  about a minute, before you spend two hours.
* Two things I *expected* to be problems and verified are not: Gemma 4 is
  `Gemma4ForConditionalGeneration` but **is** registered under `AutoModelForCausalLM`, and its
  chat template **does** support a `system` role — so the paper's verbatim system prompt
  survives intact. (Gemma 2/3 fail both; the backend defends against them anyway.)

**6. `pip install git+...` does NOT pick up new code in a live kernel.** pip sees the
package version is already satisfied and silently skips the reinstall — so a Kaggle
session that has run once keeps executing **old code** even after you push. The symptom
is baffling: `error: unrecognized arguments: --shard`, from a repo that demonstrably
has `--shard`. The notebooks now `--force-reinstall --no-deps` and assert
`csasr.__version__`, so a stale kernel fails loudly at cell 1. **Bump the version in
`pyproject.toml` whenever Kaggle must pick up a change.**

**6. Never pass an HF token as a command-line argument.** `decode.py`,
`train_whisper.py`, and `push_to_hub.py` read `HF_TOKEN` from the environment.
An earlier version took `--hf-token`, and the value was reproduced verbatim in a
Kaggle traceback (it is also visible in `ps` output). If you ever see a token in
a log, revoke it at [hf.co/settings/tokens](https://huggingface.co/settings/tokens).

---

## Deviations from the paper

| # | Paper | Here | Why |
|---|---|---|---|
| **D1** | Llama-3.3-70B-Instruct | **Gemma-4-26B-A4B-it** GGUF via **llama.cpp** (`unsloth/gemma-4-26B-A4B-it-GGUF`, apache-2.0, ungated) | 70B is 141 GB in bf16. This is a 25.2B/3.8B-active MoE, Q4_K_M ≈ 16 GB, `tensor_split` across both T4s (one process, both GPUs). Measured ~2h26m for 4,466 calls. Replaced the dense Gemma-4-E4B: see D12. |
| **D2** | whisper-large-v2 (1.54B) | **whisper-small** (244M) | Fits a T4. *(There is no `whisper-small-v2`; `v2`/`v3` exist only for `large`.)* |
| **D3** | Full FT, AdamW, lr 2e-5, batch 64 | **unchanged** | A consequence of D2: whisper-small full FT fits, so no LoRA substitution is needed. |
| **D4** | WhisperX, `language=None` | **faster-whisper** (the engine WhisperX wraps), recording-level, `language=None` | Same decoding algorithm and heuristics, minus WhisperX's forced alignment, which we don't need because we score at recording level. Our fine-tuned checkpoints are converted to CTranslate2 on first use and cached. |
| **D5** | Common Voice (official) | `fsicoli/common_voice_17_0` | Mozilla moved CV to the Data Collective in Oct 2025; the HF repo is now an empty stub. |
| **D6** | 4h dev from real train | same, **plus** a synthetic-only dev logged alongside | The paper's dev set is real in-domain audio, a mild leak against its "no real data" claim. |
| **D7** | 15h Common Voice Hindi | **11.87h** | CV 17 Hindi holds only ~20.6h across `dev`+`test`+`train`+`other`; after the 1–30s duration filter and clips missing from the per-split TSVs, 11.87h is all that exists. English is the full 15.00h. Affects M8 only. |
| **D8** | Per-utterance MER / CBA | **recording-level** MER / CBA | Forced by D4. MER is a corpus-level ratio, so concatenation is equivalent. CBA's denominator still comes from the reference *utterances* (Table 1's count). |
| **D10** | Train_T2 = 22h | **~25.1h** | Resolved by D13. At 15,000 calls the corpus reaches 18,054 sentences ≈ **25.1h**, *exceeding* the paper's 22h, so the M6-vs-M7 contrast (8h vs 25h) is at least as strong as theirs. Gate 1 prints projected hours and warns below 16h. |
| **D11** | Synthetic matches the domain | **Off-distribution** | ~**47% Latin** against the real corpus's **25%**, and ~**2.0** switch points per sentence against **3.15**. English token overlap is a healthy 88%, so the *words* come from the right domain — but the *mixture* is more English-heavy and switches less. |
| **D12** | — | **Model swap did NOT fix D11** | Replacing the dense Gemma-4-E4B with the 26B-A4B MoE moved %Latin 46.9→47.2 and switch density 1.96→2.02 — i.e. **not at all**. It *did* improve fluency (median 8→12 words) and filter survival (37.7%→85.2%, near the paper's 92.3%). Conclusion: D11 is driven by the **prompt**, not model capacity. Recorded because it is a negative result worth reporting. |
| **D13** | 4,466 bigram calls | **15,000** | The 26B repeats itself: 4,466 calls gave only **3,458 unique** bigrams vs the paper's 5,932. **Resolved** — at 15,000 calls every Gate 1 figure now *exceeds* the paper: 7,219 unique (vs 5,932), 6,114 valid (vs 5,477), 18,054 sentences (vs 16,000), ~25.1h (vs 22h). Dedup survival still falls with scale (7.7% → 4.8%), i.e. sub-linear returns: 3.4× the calls bought 2.1× the unique bigrams. Cost ~6h26m from a cold cache. |
| **D14** | Insertional CS (Hindi frame, English terms) | **Alternational CS** (English clauses alternating with Hindi) | The sharpest form of D11, and *faithful to the paper's own prompt*. Real MUCS is **88% Hindi-matrix**, mean English run **1.70 words**, **7%** of English runs ≥4 words. Ours is **58.5% Hindi-matrix**, mean English run **3.92 words**, **48.2%** ≥4 words. But §3.2.2's prompt explicitly orders *"Make 2 sentences with English as the main language and 2 sentences with Hindi as the main language"* — a 50/50 split, against the paper's own 88/12 test set. We follow the prompt verbatim, so the mismatch is inherited, not introduced. Measured by [`scripts/grammar_probe.py`](scripts/grammar_probe.py). Expected to cost absolute MER (Whisper's decoder is an LM, and we train it on the wrong register) while leaving the M6→M7→M8 *ordering* intact. **Leading suspect if M6/M7 underperform.** |
| **D9** | LLM obeys "a couple of words" | **We extract the switch pair from longer phrases** | The paper's 70B emitted true bigrams. Gemma 4 reliably appends a third word (`बुनियादी formatting basics`). Demanding exactly two tokens threw away **10/10** of a real Gemma-4 sample even though **8** carried a valid switch point. The *prompt* stays verbatim; only `script_filter` is more forgiving. `--strict-bigrams` restores the paper-faithful behaviour. |

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

Upload [`notebooks/`](notebooks/) and run them in order: **01a → 01b → 02 → 03**.
Requirements: GPU **T4 x2**, internet **on**, and an HF **write** token in Kaggle Secrets
as `HF_TOKEN`.

Only one licence to accept: [`ai4bharat/indic-parler-tts`](https://huggingface.co/ai4bharat/indic-parler-tts) (gated, no mirror). The LLM
uses an **ungated** Gemma 3 mirror. If your token is *fine-grained*, it also needs
**"Read access to contents of all public gated repos you can access"**.

`01a` and `01b` are separate because `parler-tts` pins `transformers==4.46.1` while
Gemma 3 needs `>=4.50` — see gotcha 5.

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
notebooks/          01a_generate_text, 01b_synthesize_audio, 02_train, 03_eval
configs/            data, llm, tts, train_m{6,7,8}
```

## Licence / attribution

MUCS 2021 (OpenSLR 104) is **CC BY-SA 4.0**. Derived text and audio inherit
share-alike. `RohanRamesh/mucs-he-cs` is kept private as a derivative of a challenge
corpus.
