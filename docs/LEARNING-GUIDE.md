# End-to-end learning guide

A teaching document for this repository. It assumes you know Python and
general ML, but **not** speech recognition, code-switching linguistics, or
Whisper specifically. Everything else is built up from scratch.

Read it top to bottom once. After that, each part stands alone as a
reference.

* [`README.md`](../README.md) — the lab notebook: results and gotchas as they were found.
* [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — the map: what each file does.
* **This document** — the *why*: concepts, formulae, and every place we
  deliberately diverged from the paper.

Deviations from the paper are marked inline like this:

> **⚠ DEVIATION D2** — what the paper did / what we did / why / what it costs.

---

## Table of contents

1. [The claim under test](#1-the-claim-under-test)
2. [Foundations: code-switching](#2-foundations-code-switching)
3. [Foundations: how Whisper works](#3-foundations-how-whisper-works)
4. [The two primitives: script LID and normalization](#4-the-two-primitives-script-lid-and-normalization)
5. [The metrics: MER and CBA](#5-the-metrics-mer-and-cba)
6. [Phase 0 — real data](#6-phase-0--real-data)
7. [Phase 1 — LLM text generation](#7-phase-1--llm-text-generation)
8. [Phase 2 — TTS voice synthesis](#8-phase-2--tts-voice-synthesis)
9. [Phase 3 — fine-tuning](#9-phase-3--fine-tuning)
10. [Phase 4 — evaluation](#10-phase-4--evaluation)
11. [The gates](#11-the-gates)
12. [Results and the verdict](#12-results-and-the-verdict)
13. [Complete deviation table](#13-complete-deviation-table)
14. [Questions you should be able to answer](#14-questions-you-should-be-able-to-answer)

---

## 1. The claim under test

### The paper

Biswas et al., *"Adapting Whisper for low-resource Hindi-English Code-Mix
speech with on-the-fly Augmentation & LLM-Synthesised Data"* (Interspeech
2025). It has two tracks; **we replicate Track 2 only**.

**Track 2's claim, stated plainly:** you do not need real code-switched
training audio to make an ASR model good at code-switching. You can

1. prompt a large language model to invent code-switched *text*, then
2. have a text-to-speech model *speak* that text, and
3. fine-tune the ASR model on that entirely synthetic audio,

and it will get substantially better at transcribing **real** code-switched
speech.

That claim matters because real code-switched audio is genuinely scarce.
Transcribing it is expensive and needs bilingual annotators. If synthetic
data works, the bottleneck disappears.

### The experiment

Three fine-tuned models, differing **only** in training data. Everything else
— base model, optimizer, learning rate, batch size, schedule — is held
identical, so any difference between them is attributable to data alone.

| Model | Training data | Tests |
|---|---|---|
| **M6** | Train_T1: ~8h synthetic | Does *any* synthetic data beat zero-shot? |
| **M7** | Train_T2: ~22–25h synthetic (T1 ⊂ T2) | Does *more* synthetic data help further? |
| **M8** | Train_T2 + ~27h real monolingual (Common Voice hi + en) | Does real-but-out-of-domain audio help on top? |

**"Zero-shot"** means the base model with no fine-tuning at all — the
starting point every fine-tuned model must beat to have done anything.

The paper's reported result:

| Model | MER ↓ | CBA-HE ↑ | CBA-EH ↑ |
|---|---|---|---|
| large-v2 zero-shot | 52.0 | 42.9 | ~36 |
| M6 | 48.2 | 45.5 | ~45 |
| M7 | 40.8 | 52.3 | ~55 |
| M8 | 39.2 | 55.4 | ~56 |

MER ↓ = lower is better (it is an error rate). CBA ↑ = higher is better (it
is an accuracy). Both defined precisely in [Part 5](#5-the-metrics-mer-and-cba).

> **⚠ The headline deviations (D1, D2)** — the paper used a 70B-parameter LLM
> and fine-tuned `whisper-large-v2` (1.54B params). Neither fits free-tier
> hardware. We used a smaller LLM and a much smaller Whisper. **Absolute MER
> therefore cannot match Table 2.** What we set out to test is the
> **ordering**: `zero-shot > M6 > M7 > M8` (each better than the last).
> Whether that survived is [Part 12](#12-results-and-the-verdict) — and the
> honest answer is *no*, for reasons that turn out to be about our compute
> budget rather than the paper's idea.

---

## 2. Foundations: code-switching

### What it is

**Code-switching** is alternating between two languages inside a single
conversation, sentence, or even phrase. It is normal, rule-governed
bilingual behaviour, not sloppiness. In Hindi-English (colloquially
"Hinglish") it is pervasive in Indian technical and educational speech:

```
लिबर ऑफिस impress में एक प्रस्तुति document बनाना
(in LibreOffice Impress, creating a presentation document)
```

Hindi supplies the grammar; English supplies the technical nouns.

### Terms you need

**Matrix language** — the language supplying the sentence's grammatical
frame (word order, inflection, function words). **Embedded language** — the
one contributing inserted material. In the example above the matrix is
Hindi, the embedded is English.

**Insertional code-switching** — single embedded words or short phrases
dropped into an otherwise-monolingual frame. `प्रस्तुति document बनाना`
("create a presentation document") is insertional: one English noun inside
Hindi syntax.

**Alternational code-switching** — the sentence genuinely switches languages
mid-stream, with each language contributing whole clauses:
`I opened the file और फिर उसे save किया`.

This distinction becomes a real problem later — see
[D14](#d14-insertional-vs-alternational).

**Switch point** — the boundary between two adjacent words of different
languages. Both metrics in this project are built on switch points.

**Switch bigram** — the pair of words *at* a switch point. We classify them
by direction:

* **HE** = Hindi→English (`प्रस्तुति document`)
* **EH** = English→Hindi (`document बनाना`)

Direction matters because they are not symmetric in difficulty or frequency:
the MUCS test set has 4,189 HE and 5,176 EH bigrams.

### Why ASR finds this hard

A monolingual ASR system has one language model biasing its output toward
plausible word sequences *in that language*. At a switch point, that bias is
actively wrong — it pushes the decoder to hallucinate a same-language
continuation.

Whisper specifically has a `<|language|>` token that conditions the whole
transcript. Forced into Hindi, it tends to **transliterate** English words
into Devanagari (`impress` → `इंप्रस`) rather than writing them in Latin
script. That single behaviour destroys the CBA metric completely, because
CBA is defined by script boundaries. If everything comes out in Devanagari,
there are no boundaries, and no switch bigram can ever match. We measured
exactly this — see [Part 10](#10-phase-4--evaluation).

---

## 3. Foundations: how Whisper works

Whisper is an encoder-decoder Transformer trained by OpenAI on 680,000 hours
of multilingual audio. Understanding four things about it explains most
design decisions downstream.

### 3.1 Audio → log-mel spectrogram

Whisper does not consume raw audio. It consumes a **log-mel spectrogram** —
a 2-D image-like array of (frequency band × time).

The pipeline, with this project's actual verified parameters:

```
1. Resample to 16,000 Hz mono                        (sampling_rate = 16000)
2. Short-Time Fourier Transform (STFT):
      window n_fft = 400 samples  = 25 ms
      hop    = 160 samples        = 10 ms
3. Power spectrum:   P[k,t] = |STFT[k,t]|²
4. Mel filterbank:   M[m,t] = Σ_k mel_m[k] · P[k,t]     (80 mel bands)
5. Log compression:  L[m,t] = log10(max(M[m,t], 1e-10))
6. Normalize to roughly [-1, 1]
```

**Why "mel"?** The mel scale spaces frequency bands the way human hearing
does — finely at low frequencies, coarsely at high ones. It compresses a
~200-bin linear spectrum into 80 perceptually-motivated bands.

**Why log?** Loudness is perceived logarithmically, and log compresses a
huge dynamic range into something a neural net can handle.

**The critical consequence — the fixed 30-second window.** Whisper always
pads or truncates its input to exactly 30 seconds:

```
30 s × 16,000 Hz ÷ 160 hop = 3,000 frames
Every input is therefore exactly 80 × 3000, regardless of clip length.
```

This is why a 4-second synthetic clip costs the *same* encoder compute as a
30-second one — most of it is spent on padding. It is the single biggest
reason training is slower than you'd expect for a small model, and it comes
up again in [Part 9](#9-phase-3--fine-tuning).

### 3.2 The Transformer, briefly

The **encoder** turns the 80×3000 spectrogram into a sequence of 1,500
audio-representation vectors (it downsamples by 2 via strided convolution).
The **decoder** autoregressively generates text tokens, attending both to
its own previous outputs (self-attention) and to the encoder output
(cross-attention).

The core operation, scaled dot-product attention:

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

* `Q` (query), `K` (key), `V` (value) — learned linear projections of the input.
* `d_k` — dimension per attention head; dividing by `√d_k` keeps the
  softmax out of its saturated region where gradients vanish.

The decoder is, functionally, **a language model conditioned on audio**.
That framing explains a lot: if you fine-tune it on text of the wrong
register, you damage its language modelling — which is exactly the risk
[D14](#d14-insertional-vs-alternational) describes.

### 3.3 Special tokens

Whisper's decoder output always begins with a control prefix:

```
<|startoftranscript|> <|hi|> <|transcribe|> <|notimestamps|> … text … <|endoftext|>
```

* `<|hi|>` — the language token. **This project always uses `<|hi|>` for
  training** (per the paper's Table 2), for all of M6/M7/M8.
* `<|transcribe|>` — as opposed to `<|translate|>` (Whisper can translate
  to English; we never want that).
* `<|notimestamps|>` — suppresses timestamp tokens.

At *evaluation* we pass `language=None`, meaning Whisper must **detect** the
language itself. That is deliberate and is the whole point of the test: a
model that has not learned code-switching mis-detects, then transliterates
or deletes.

### 3.4 Model sizes

Verified from the actual configs:

| Model | Enc/Dec layers | `d_model` | Heads | FFN dim | Params |
|---|---|---|---|---|---|
| `whisper-base` | 6 / 6 | 512 | 8 | 2,048 | 74M |
| `whisper-small` | 12 / 12 | 768 | 12 | 3,072 | 244M |
| `whisper-large-v2` | 32 / 32 | 1,280 | 20 | 5,120 | 1.54B |

Transformer compute scales roughly as `layers × d_model²`, so:

```
base vs small:  (6/12) × (512/768)²  ≈ 0.22   → small is ~4.5× the FLOPs of base
```

Measured wall-clock was ~3× (not 4.5×), because per-step overhead —
dataloading, optimizer step, the fixed-size encoder pass — does not scale
with model size.

> **⚠ DEVIATION D2 — base model size.**
> **Paper:** `whisper-large-v2` (1.54B).
> **Us:** `whisper-small` (244M) originally, then `whisper-base` (74M) for
> the final runs.
> **Why:** large-v2 full fine-tuning does not fit a 16 GB T4 at all. The
> later base→small drop was a deadline/quota decision, see
> [D15](#123-why-it-did-not-reproduce--the-leading-explanation).
> **Cost:** absolute MER is far worse than the paper's. This is the dominant
> reason our numbers are not comparable to theirs in absolute terms.

---

## 4. The two primitives: script LID and normalization

Everything downstream — Gate 0's counts, the bigram filter, both metrics —
rests on these two modules. Get them wrong and every number in the project
is quietly wrong.

### 4.1 Language ID by script — [`src/csasr/lid.py`](../src/csasr/lid.py)

**LID** = Language IDentification. Normally this is a statistical model. Here
it is deterministic and dependency-free, because Hindi and English use
**disjoint scripts**:

* Hindi → Devanagari (Unicode block U+0900–U+097F)
* English → Latin

So a word's language is decided by looking at its characters. `word_lang()`
returns one of four labels:

| Label | Meaning |
|---|---|
| `HI` | contains Devanagari, no Latin |
| `EN` | contains Latin, no Devanagari |
| `MIXED` | contains **both** inside one whitespace token |
| `OTHER` | neither — digits, punctuation, symbols |

**Trap #1: the danda is inside the Devanagari block.** The Hindi full stop
`।` is U+0964, which *is* in the Devanagari range — but it is punctuation,
not a letter. Verified:

```
danda U+0964 category: Po        (P = punctuation)
```

A naive block-range test would classify a bare `।` as a Hindi word and
inflate every Hindi word count. `is_devanagari()` therefore additionally
requires Unicode category `L*` (letter) or `M*` (combining mark).

**Trap #2: dropped-space typos hide real switch points.** The MUCS
transcripts contain tokens like `दायाँclick` where a space was lost. As one
token it is `MIXED` and contributes no switch point.
`split_script_boundaries()` splits exactly where script changes. Verified:

```python
tokenize('यह दायाँclick और document है')
# → ['यह', 'दायाँ', 'click', 'और', 'document', 'है']

cs_bigrams(...)
# → [('दायाँ','click','HE'), ('click','और','EH'),
#    ('और','document','HE'), ('document','है','EH')]
```

Without the split, the first HE switch point would have been invisible.
Doing this brings our Hindi word count to within 4 of the paper's.

**Switch bigram extraction** (`cs_bigrams`) walks adjacent tagged pairs and
emits one only when the language flips HI↔EN. `OTHER` tokens never open or
close a switch point — a digit is neither Hindi nor English, and the paper's
own filter requires each bigram to contain "both Hindi and English
characters".

### 4.2 Normalization — [`src/csasr/normalize.py`](../src/csasr/normalize.py)

**Normalization** means stripping surface variation (punctuation, case) that
shouldn't count as a transcription error, before scoring.

The obvious choice would be Whisper's own `BasicTextNormalizer`. **It is
catastrophic for Devanagari.** It removes every character in Unicode
categories `M`, `S`, `P` — and Devanagari vowel signs, the virama, and the
nukta are all *combining marks* (`Mn`/`Mc`). Verified on a real word:

```
दस्तावेज़  ("document")  — 9 codepoints:
  U+0926 Lo  DEVANAGARI LETTER DA
  U+0938 Lo  DEVANAGARI LETTER SA
  U+094D Mn  DEVANAGARI SIGN VIRAMA        ← deleted by BasicTextNormalizer
  U+0924 Lo  DEVANAGARI LETTER TA
  U+093E Mc  DEVANAGARI VOWEL SIGN AA      ← deleted
  U+0935 Lo  DEVANAGARI LETTER VA
  U+0947 Mn  DEVANAGARI VOWEL SIGN E       ← deleted
  U+091C Lo  DEVANAGARI LETTER JA
  U+093C Mn  DEVANAGARI SIGN NUKTA         ← deleted

scoring        → ['दस्तावेज़']              one word, intact
whisper_basic  → ['दस', 'त', 'व', 'ज']     one word became FOUR
```

That would inflate Hindi word counts by ~66%, break Gate 0, and silently
corrupt both metrics. Our `scoring` preset strips only `P*` (punctuation)
and `S*` (symbols) and **always preserves marks**. The danda is category
`Po`, so it is still correctly removed as punctuation.

`whisper_basic` is kept in the codebase **only** as a Gate 0 regression
guard that must visibly fail. If someone ever "fixes" `normalize.py` by
reaching for the standard normalizer, Gate 0 fails loudly.

The four presets:

| Preset | Strips | Use |
|---|---|---|
| `raw` | whitespace collapse only | debugging |
| `punct` | `P*`, `S*`; keeps case | sentence validation |
| `scoring` | `P*`, `S*`, lowercases | **default for MER/CBA** |
| `whisper_basic` | `M*`, `S*`, `P*` | regression guard — never for scoring |

---

## 5. The metrics: MER and CBA

### 5.1 Edit distance, the foundation of both error rates

ASR accuracy is measured by **Levenshtein (edit) distance** — the minimum
number of single-unit insertions, deletions, and substitutions to turn the
hypothesis into the reference. Dynamic programming gives the counts:

* **S** — substitutions (wrong word in the right place)
* **D** — deletions (reference word missing from hypothesis)
* **I** — insertions (hypothesis word not in reference)

### 5.2 WER → MER

**WER** (Word Error Rate), the standard ASR metric:

```
WER = (S + D + I) / N × 100%
```

where `N` = number of units in the **reference**.

Two things surprise people:

1. **WER is a corpus-level ratio, not an average of per-utterance rates.**
   You sum all errors across the corpus and divide by all reference units.
   This is why concatenating utterances before scoring is mathematically
   equivalent — a fact this project relies on in [Part 10](#10-phase-4--evaluation).
2. **WER can exceed 100%.** Nothing bounds `I`. A model that hallucinates
   freely inserts without limit. We measured `whisper-tiny` zero-shot at
   **MER 125%** — a real result, not a bug.

**MER** (Mixed-language Error Rate) is WER adapted for code-mixed text. The
paper cites Zhang et al.'s SEAME metric for Mandarin-English, where errors
are counted at **character** level on the Mandarin side and **word** level
on the English side — because Mandarin is not whitespace-segmented, so
"words" aren't well-defined.

Hindi **is** whitespace-segmented. So the definition is genuinely ambiguous
for Hindi-English, and the paper does not say which it used. We implement
both ([`src/csasr/eval/mer.py`](../src/csasr/eval/mer.py)):

| Mode | Devanagari units | Latin units |
|---|---|---|
| `word` | whole words | whole words |
| `hybrid` | **individual characters** | whole words |

Worked example, verified:

```
ref = 'यह document है'          hyp = 'यह डॉक्यूमेंट है'
                                (English word transliterated into Devanagari)

word units (ref)  : ['यह', 'document', 'है']              → N = 3
hybrid units (ref): ['य','ह','document','ह','ै']          → N = 5

MER word   =  33.33     (1 substitution out of 3 words)
MER hybrid = 200.0      (the transliteration explodes into many char units)
```

That 200% is a good illustration of both facts above: error rates exceed
100%, and the choice of unit changes the number enormously.

**Which is right?** Settled empirically by Gate 3 — whichever mode puts
zero-shot `whisper-large-v2` nearest the paper's published 52.0. Result:

```
MER word   = 54.8
MER hybrid = 51.9      ← paper reports 52.0
```

`hybrid` it is. This is a nice example of using a published number to
resolve a definitional ambiguity instead of guessing.

### 5.3 CBA — Code-Switch Bigram Accuracy

MER measures overall transcription quality. It does **not** specifically
measure whether the model handled the *switch points* — the thing this
entire project is about. CBA does.

The paper defines it loosely as "correctly recognized bigrams at switch
points relative to their total count in the test set". Formally, as
implemented in [`src/csasr/eval/cba.py`](../src/csasr/eval/cba.py):

```
                  Σ_g  Σ_b  min( ref_count_g(b), hyp_count_g(b) )
CBA_HE  =  100 × ─────────────────────────────────────────────────
                            Σ_g  Σ_b  ref_count_g(b)
```

* `g` — a group (in our case one recording).
* `b` — a distinct HE switch bigram.
* `ref_count_g(b)` — how many times `b` occurs in group `g`'s **reference**.
* `hyp_count_g(b)` — how many times the hypothesis offers `b`.
* `CBA_EH` is the same with EH bigrams.

Three subtleties, each of which caused a real bug here:

**(a) Multiset, not set.** Table 1 of the paper counts 4,189 HE bigram
*tokens* against only 2,347 unique *types*. A bigram occurring three times
must contribute 3 to the denominator and may contribute up to 3 to the
numerator. Hence `min(ref_count, hyp_count)` rather than a set intersection
— a set-based version would under-count the denominator and inflate the
score.

**(b) The denominator must come from reference *utterances*, not
concatenated recordings.** We decode whole recordings (see
[Part 10](#10-phase-4--evaluation)), but if you concatenate a recording's
utterances *before* extracting reference bigrams, you fabricate switch pairs
straddling the joins — pairs no actual reference sentence contains. This
inflated our denominator by 23% and **halved** the measured CBA before it
was caught. `cba_grouped()` exists precisely to extract the denominator
per-utterance while matching against a per-recording hypothesis.

**(c) "Correctly recognized" is never defined by the paper.** Two readings
are defensible, and we implement both:

| Mode | Requirement | Gate 3 result | Paper |
|---|---|---|---|
| `adjacent` | both words correct **and still adjacent** | HE 20.1 | — |
| `lenient` | both words appear **anywhere** in the hypothesis | HE 43.8 | 42.9 |

`lenient` reproduces the paper's number almost exactly; `adjacent` — the
literal reading of the word "bigram" — lands at about half. We **default to
`adjacent`** because it is the honest literal reading, and **report both**.

Crucially, this choice does not affect the experiment: every system is
scored identically, so the M6→M7→M8 ordering is preserved under either rule.
The rule you must never do is mix modes across systems.

---

## 6. Phase 0 — real data

CPU-only, runs on a laptop. Goal: get the real corpora into a verified,
normalized shape and publish them to the Hugging Face Hub.

### 6.1 The corpus

**MUCS 2021** (Multilingual and Code-Switching ASR Challenges for Low
Resource Indian Languages), Hindi-English subtask, distributed as
**OpenSLR 104** under CC BY-SA 4.0.

| Split | Duration | Use here |
|---|---|---|
| train | 89.8 h | **text only** — few-shot exemplars + a 4h dev slice |
| test | 5.2 h | the only split any reported metric is computed on |

**The central rule of Track 2: we never train on real code-switched audio.**
The 89.8h train split is downloaded, its transcripts read, and the audio
deleted. Only (a) its text, as LLM prompt exemplars, and (b) a 4-hour dev
slice for checkpoint selection, are ever used.

> **⚠ DEVIATION D6 — the dev set.** The paper selects checkpoints on a 4h dev
> slice of *real* audio, which is a mild leak against its own "no real data"
> claim. We do the same (to stay faithful) but log a synthetic-only dev
> alongside for comparison.

### 6.2 Kaldi format — [`prepare_mucs.py`](../src/csasr/data/prepare_mucs.py)

MUCS ships in **Kaldi** layout (Kaldi being the long-dominant classical ASR
toolkit). Three plain-text files:

```
text      <utt_id> <transcript…>              what was said
segments  <utt_id> <reco_id> <start> <end>    where in the recording
wav.scp   <reco_id> <path>                    which audio file
```

Note the two-level structure: a long **recording** contains many short
**utterances**. This matters enormously at evaluation time
([Part 10](#10-phase-4--evaluation)).

`scan()` *discovers* these files by name rather than hardcoding paths,
because the train and test tarballs are laid out differently. `wav.scp`
paths point at the original packagers' machines, so they are remapped by
basename onto whatever `.wav` files actually exist locally.

### 6.3 The manifest contract — [`manifest.py`](../src/csasr/manifest.py)

Every stage of this pipeline **reads a JSONL manifest and writes a JSONL
manifest**. No stage ever hands another stage a live Python object.

One line per utterance:

```json
{"utt_id": "…", "text": "…", "dur": 6.02, "wav": "path.wav", "speaker": "…"}
```

Two properties fall out of this one rule, and both are load-bearing:

1. **Resumability** — a stage can be killed and restarted without losing
   upstream work.
2. **Environment isolation** — stages run in mutually incompatible Python
   environments (see [Part 8](#8-phase-2--tts-voice-synthesis)) because the
   only thing crossing the boundary is a file.

### 6.4 Common Voice — [`prepare_cv.py`](../src/csasr/data/prepare_cv.py)

Mozilla's **Common Voice** is a crowdsourced monolingual read-speech corpus.
Used only by M8, as auxiliary monolingual audio.

> **⚠ DEVIATION D5 — the source.** Mozilla moved Common Voice to the Mozilla
> Data Collective in Oct 2025 and the official HF repo is now an empty stub.
> We use the CC0 mirror `fsicoli/common_voice_17_0`.

> **⚠ DEVIATION D7 — Hindi hours.** The paper uses 15h each. CV17 Hindi
> holds only ~20.6h total across all splits; after the 1–30s duration filter
> only **11.87h** survives. English is the full 15.00h.

This module contains a nice piece of engineering worth understanding.
Loading CV naively is infeasible: English `train` alone is **45 GB across 28
tar shards** with a 363 MB metadata TSV. Instead it:

* walks splits **smallest-metadata-first** (`dev → test → train → other`)
  — `en/dev` alone holds ~50h behind a 4.9 MB TSV;
* streams the tars over HTTP with Python's `tarfile` in stream mode;
* **aborts the moment the hour target is met**.

A few hundred MB is actually transferred. A per-speaker cap
(`--max-per-speaker`) prevents one prolific contributor dominating, which
matters because a streamed tar is consumed in order, making this a *prefix*
sample rather than a uniform one.

### 6.5 Publishing — [`push_to_hub.py`](../src/csasr/data/push_to_hub.py)

Manifests become Parquet datasets on the Hub, audio FLAC-encoded (lossless,
roughly halves size: 22h is 2.53 GB as WAV, ~1.39 GB as FLAC).

**Audio arithmetic worth memorizing:**

```
16,000 samples/s × 16 bits × 1 channel = 256 kbit/s = 32 kB/s
→ 115.2 MB per hour of 16 kHz 16-bit mono PCM
```

This is why `synthesize.py` writes int16 and never float32 — float32 at 44.1
kHz would make a 22h corpus ~7 GB instead of ~2.5 GB.

`verify_round_trip()` re-downloads what was just pushed and checks row count,
duration, and sample rate — cheap insurance against committing a multi-hour
training run to a corrupted upload.

---

## 7. Phase 1 — LLM text generation

Runs on Kaggle GPU. Goal: invent thousands of natural Hindi-English
code-switched sentences using only an LLM and the real corpus's *transcripts*
as style exemplars.

### 7.1 The two-stage prompt design

The paper's method, which we follow verbatim:

**Stage 1a — generate bigrams.** Few-shot-prompt the LLM with five real MUCS
sentences and ask for ten Hindi-English switch bigrams.

**Few-shot prompting** = showing the model examples of the desired output
inside the prompt, rather than fine-tuning it. The examples come from the
real corpus, which is how domain vocabulary transfers into the synthetic
data without ever using the audio.

**Stage 2a — expand each bigram into sentences.** For each surviving bigram,
ask for four natural sentences using it — two with English as matrix
language, two with Hindi.

The prompts in [`prompts.py`](../src/csasr/llm/prompts.py) are transcribed
**verbatim** from the paper (§3.2.1, §3.2.2). Do not paraphrase them; the
yield statistics are only comparable if the prompt is identical.

A nice piece of forensics lives here: the paper's three few-shot examples
were mangled by PDF text extraction into `pr-tEt document; bEnyAdF
formatting; isspoken`. All three are recoverable from the first line of the
MUCS test transcripts — and `pr-tEt` is **प्रस्तुति** ("presentation"),
which occurs in the corpus, *not* प्रतीत ("appears"), which occurs zero
times.

### 7.2 The model

> **⚠ DEVIATION D1 — the LLM.**
> **Paper:** Llama-3.3-70B-Instruct.
> **Us:** `unsloth/gemma-4-26B-A4B-it-GGUF` via llama.cpp.
> **Why:** 70B is 141 GB in bf16. Ours is a 25.2B-total / **3.8B-active**
> Mixture-of-Experts, ~16 GB quantized to Q4_K_M.

Terms:

* **MoE (Mixture of Experts)** — only a subset of the network's parameters
  ("experts") activate per token. 25.2B total but 3.8B active means it costs
  roughly a 4B model to run while having far more stored knowledge. This is
  why 4,466 generation calls took only ~2h26m.
* **GGUF** — llama.cpp's model file format.
* **Quantization / Q4_K_M** — storing weights at ~4 bits instead of 16,
  trading a little accuracy for a ~4× memory reduction. This is what makes
  a 25B model fit at all.
* **`tensor_split`** — splits *one* model across both T4s. Note: one
  process, two GPUs — **not** two independent shards.

Why not vLLM: its AWQ kernels require compute capability ≥ 8.0 (Ampere);
Kaggle's T4 is sm75.

### 7.3 Filtering — [`filter_bigrams.py`](../src/csasr/llm/filter_bigrams.py)

Two filters in order, matching the paper's description:

**1. Script filter** (deterministic, free). Extract the first
Hindi↔English script-crossing adjacent pair. Kills bigrams that are entirely
one language.

> **⚠ DEVIATION D9 — extraction vs validation.** The paper's 70B obeyed "a
> couple of words" and emitted true two-word bigrams. Our smaller model
> reliably appends a third (`बुनियादी formatting basics`). Demanding exactly
> two tokens discarded **10/10** of a real sample even though 8 carried a
> valid switch point. So `script_filter()` *extracts* the switch pair from
> whatever comes back. The **prompt stays verbatim**; only our parser is more
> forgiving. `--strict-bigrams` restores paper-faithful behaviour.

**2. Translation check** (LLM-based). Rejects pairs that are just a word and
its own translation (`दस्तावेज़ document` — Hindi "document" followed by
English "document"). Such a pair is not a real code-switch, it is a gloss.

This uses **self-consistency**: sample the judge 3× at temperature 0.7 and
take a majority vote, because a smaller judge is noisier than the paper's
70B. Unparseable or tied votes **keep** the bigram — never silently drop
data.

### 7.4 Validation of generated sentences

Every returned sentence must (a) contain the bigram as *adjacent* tokens and
(b) carry both scripts. Sentences failing either are discarded, reproducing
the paper's own shortfall honestly rather than padding counts.

`matrix_lang()` assigns the sentence's matrix language by **word-count
majority**, not by trusting the model's own labelling of its output.

### 7.5 The label-leak bug — [`fix_sentences.py`](../src/csasr/llm/fix_sentences.py)

Worth studying as a category of failure. Gemma prefixes many sentences with
a literal matrix-language label:

```
English: Many software programs have different aliases निर्धारित for commands.
```

If `"English: "` survives into the corpus it gets **spoken aloud by the TTS**
and then **learned by Whisper as real transcript text** — quietly poisoning
the whole synthetic dataset with a phantom token that appears in no real
speech. `fix_sentences.py` re-cleans an already-generated corpus in place
(strip label, re-validate against the bigram, recompute matrix language,
dedup) in seconds rather than re-running hours of generation.

### 7.6 Resumability — [`cache.py`](../src/csasr/llm/cache.py)

Every LLM request is hashed on `(model, messages, sampling params, sample
index)` and its response appended to a JSONL the instant it returns. A
re-run skips anything already done.

This is the single mechanism that makes an 8–12h generation run survivable
on a platform that kills sessions at 12 hours. **A stage that cannot resume
cannot finish.**

### 7.7 The distribution problems

These are the most intellectually interesting deviations in the project,
because they are *measured negative results*.

> **⚠ DEVIATION D11 — the synthetic text is off-distribution.**
> Measured against real MUCS: ~**47% Latin** vs the real corpus's **25%**,
> and ~**2.0** switch points/sentence vs **3.15**. English *vocabulary*
> overlap is a healthy 88% — the words come from the right domain — but the
> *mixture* is more English-heavy and switches less.

> **⚠ DEVIATION D12 — capacity did not fix D11.** Swapping the dense
> Gemma-4-E4B for the 26B-A4B MoE moved %Latin 46.9→47.2 and switch density
> 1.96→2.02 — i.e. not at all. It *did* improve fluency (median 8→12 words)
> and filter survival (37.7%→85.2%). **Conclusion: D11 is driven by the
> prompt, not model size.** A genuine negative result worth reporting.

> **⚠ DEVIATION D13 — call count.** The paper used 4,466 bigram calls. At
> that count our model repeats itself, yielding only 3,458 unique bigrams vs
> the paper's 5,932. Raised to **15,000 calls**, at which every Gate 1 figure
> *exceeds* the paper: 7,219 unique (vs 5,932), 6,114 valid (vs 5,477),
> 18,054 sentences (vs 16,000), ~25.1h (vs 22h). Dedup survival falls with
> scale (7.7%→4.8%) — sub-linear returns: 3.4× the calls bought 2.1× the
> unique bigrams.

#### D14: insertional vs alternational

The sharpest form of D11, and the most linguistically interesting.

Measured by [`grammar_probe.py`](../scripts/grammar_probe.py), which computes
**contiguous same-language run lengths** — a 1-word English run is an
insertion, a 5-word run is an alternation:

| | Real MUCS | Our synthetic |
|---|---|---|
| Hindi-matrix sentences | **88%** | 58.5% |
| Mean English run | **1.70 words** | 3.92 words |
| English runs ≥4 words | **7%** | 48.2% |

Real tutorial speech is **insertional**: a Hindi frame with short English
technical terms. Ours is **alternational**: whole English clauses.

**But this is inherited, not introduced.** The paper's own §3.2.2 prompt
explicitly orders *"Make 2 sentences with English as the main language and 2
sentences with Hindi as the main language"* — a 50/50 split, against the
paper's own test set which is 88/12 Hindi-matrix. We follow the prompt
verbatim, so we inherit the mismatch.

**Why it should matter:** Whisper's decoder is a language model. Training it
on the wrong register damages its language modelling on the right one. This
was flagged in the README as the **leading suspect if M6/M7 underperform** —
before we had results.

---

## 8. Phase 2 — TTS voice synthesis

Goal: turn every validated sentence into spoken audio, then carve out the
paper's two training sets.

### 8.1 Why this is a separate notebook

`parler-tts` hard-pins `transformers==4.46.1`. Gemma 4 needs
`transformers>=5.5`. **These cannot coexist in one Python process.** Hence
`01a_generate_text.ipynb` and `01b_synthesize_audio.ipynb` are separate
Kaggle sessions with the Hub as the checkpoint between them. This is the
clearest possible justification for the manifest contract from
[6.3](#63-the-manifest-contract--manifestpy).

### 8.2 Description-conditioned TTS

`ai4bharat/indic-parler-tts` is unusual: instead of a fixed speaker ID, you
describe the voice in **natural language**:

```
"Rohit speaks at a moderate pace with a neutral tone.
 The recording is very clear audio, close-sounding, with minimal background noise."
```

The literal phrase *"very clear audio"* is documented on the model card as
improving output quality — it is not decoration. The model auto-detects
language from the prompt text, which is exactly what we want for code-mixed
input.

The paper uses "one Indian male and one Indian female voice"; the model
card's recommended Hindi pair is Rohit and Divya.

**A subtle correctness detail:** speaker assignment uses a **SHA-1 digest**
of the sentence id, deliberately not Python's built-in `hash()`. `hash()` is
salted per-process (`PYTHONHASHSEED`), so a run resumed after a timeout
would reassign voices mid-corpus, producing inconsistent speaker/sentence
pairings across the resume boundary.

### 8.3 Three engineering details

**Sample rate read at runtime.** The model card claims 22.05 kHz; the model
actually emits **44.1 kHz**. `synthesize.py` reads
`model.config.sampling_rate` rather than trusting documentation, then
resamples to 16 kHz.

**Trailing-silence trim.** Batched generation pads shorter clips with codec
silence. Untrimmed, this inflates every clip's measured duration and
therefore the corpus's total hours. Clips outside 0.5–30s after trimming are
dropped as TTS collapse (44 of 18,054 — 0.24%).

**Length-sorted batching.** Parler-TTS decodes autoregressively over audio
codec frames, so batching similar-length sentences minimizes wasted padding.

### 8.4 Train_T1 ⊂ Train_T2 — [`make_subset.py`](../src/csasr/tts/make_subset.py)

The paper: *"Train_T1 (8 hours) and Train_T2 (22 hours), where Train_T1 is a
subset of Train_T2."*

We store T1 **by reference** — a JSON list of utterance ids applied with
`.filter()` at load time — rather than copying ~1 GB of audio that already
exists in T2.

Selection **round-robins across bigram groups** so the 8h subset covers as
many distinct switch points as possible, rather than over-sampling a few
bigrams that happened to yield many sentences. Result: 6,069 clips / 8.00h
covering **100%** of T2's 5,387 distinct bigrams.

Three assertions guard the one thing that must never silently break: every
T1 id exists in T2, no duplicates, and T1 is a *strict* subset.

---

## 9. Phase 3 — fine-tuning

### 9.1 What fine-tuning a seq2seq ASR model means

**Teacher forcing.** During training the decoder is fed the *ground-truth*
previous tokens rather than its own predictions, so all positions can be
computed in parallel and errors don't compound. The loss is token-level
cross-entropy:

```
L = − (1/T) Σ_{t=1..T}  log P( y_t | y_<t , X )
```

* `X` — the audio (encoder input)
* `y_t` — the correct token at position `t`
* `y_<t` — all preceding ground-truth tokens
* `T` — number of target tokens

Minimizing this maximizes the likelihood the model assigns to the correct
transcript.

### 9.2 The collator — [`collator.py`](../src/csasr/train/collator.py)

Two details that silently corrupt training if missed. Both are tested in
[`tests/test_collator.py`](../tests/test_collator.py).

**(a) Label padding must be `-100`.** PyTorch's `CrossEntropyLoss` has
`ignore_index=-100` by default. Padding labels with the tokenizer's actual
pad token would train the model to *emit padding as real output*.

A subtlety specific to Whisper: its pad token and EOS token are the **same
id** (50257). So "no pad id in labels" is *not* the invariant — every
sequence legitimately ends with 50257. The real invariant is that `-100`
appears only as a trailing run.

**(b) Strip the leading decoder-start token.** The tokenizer emits
`<|startoftranscript|><|hi|><|transcribe|><|notimestamps|> …`, but the
model *also* prepends `decoder_start_token_id` itself when it shifts labels
right to build `decoder_input_ids`. Leaving both shifts the sequence by one
and the model learns to predict its own BOS token instead of the first real
word.

Note the input side needs no padding logic at all — log-mel features are
*always* 80×3000 ([3.1](#31-audio--log-mel-spectrogram)), so it is a stack,
not a pad.

### 9.3 The optimizer recipe

The paper's recipe, kept intact:

| Setting | Value | What it means |
|---|---|---|
| Optimizer | AdamW | Adam with *decoupled* weight decay |
| Learning rate | 2e-5 | peak LR |
| Warmup | 200 steps | linear ramp 0 → peak |
| Effective batch | 64 | see below |
| Max steps | 5,000 (paper) | hard ceiling |

**AdamW** maintains per-parameter first and second moment estimates:

```
m_t = β₁ m_{t-1} + (1−β₁) g_t                (momentum)
v_t = β₂ v_{t-1} + (1−β₂) g_t²               (variance)
θ_t = θ_{t-1} − lr · m̂_t / (√v̂_t + ε) − lr · λ · θ_{t-1}
                                              └── decoupled weight decay
```

Practically: it stores **two extra float32 buffers per parameter**, so
optimizer state is ~2× the model size on top of weights and gradients. This
is what makes "does full fine-tuning fit?" a real question.

**Gradient accumulation** achieves a large effective batch on small VRAM:

```
effective_batch = per_device_batch × grad_accum × n_devices
                = 16 × 4 × 1  = 64      (single GPU)
                = 16 × 4 × 2  = 128     (Kaggle's 2× T4, auto-DataParallel)
```

Gradients are summed over `grad_accum` micro-batches before one optimizer
step. Mathematically ≈ one big batch; memory cost of one small one.

**Warmup** exists because Adam's variance estimates are unreliable in the
first few steps; a full-size LR then can destabilize pretrained weights.

> **⚠ DEVIATION D3 — full fine-tuning, no LoRA.** The paper full-fine-tunes,
> and so do we. **LoRA** (Low-Rank Adaptation) freezes the base weights and
> trains small low-rank matrices `ΔW = BA` instead, cutting optimizer state
> enormously. We did **not** need it: whisper-small's ~4 GB of optimizer
> state fits a 16 GB T4. Using LoRA would have added a second deviation from
> the paper's methodology for no benefit — and note it would *not* have
> closed the gap to the paper's numbers, since that gap is driven by model
> capacity (D2), not by training method.

### 9.4 Hardware constraints

**fp16 vs bf16.** Kaggle's T4 is **Turing** architecture, which has **no
bfloat16 support**. So `fp16=True`, which enables autocast plus a
`GradScaler`.

* **fp16** — 10-bit mantissa, 5-bit exponent. Max ~65,504. Precise but
  narrow range; gradients can underflow to zero.
* **bf16** — 8-bit mantissa, 8-bit exponent — same range as fp32. Less
  precise, far more robust. Not available on Turing.
* **GradScaler** — multiplies the loss by a large factor before backward so
  small gradients stay representable in fp16, then unscales before the
  optimizer step. Without it, fp16 training silently stalls.

This same limitation bites in the LLM phase: Gemma's config declares
`torch_dtype: bfloat16`, so [`backend.py`](../src/csasr/llm/backend.py)
probes the logits after loading and transparently reloads in float32 if they
come back NaN/inf.

**Gradient checkpointing** (`gradient_checkpointing=True`) trades compute for
memory: instead of storing every intermediate activation for the backward
pass, it stores a few and **recomputes** the rest. Costs ~30–40% extra
compute, roughly halves activation memory. It is what makes full fine-tuning
fit — and part of why LoRA's speedup would be smaller than expected here
(LoRA's memory savings would let you *disable* checkpointing, which is where
its real speed win would come from).

### 9.5 Checkpoint selection

```
eval_strategy   = "steps",  eval_steps = 250
metric          = "mer"  (greater_is_better = False)
load_best_model_at_end = True
EarlyStoppingCallback(patience = 4)
```

**Early stopping**: if the dev metric fails to improve for `patience`
consecutive evaluations, stop. This is why `max_steps=5000` is a *ceiling*,
not a prediction — a point that caused real confusion mid-project when tqdm
displayed a 56-hour ETA (it extrapolates to `max_steps`, knowing nothing
about early stopping).

`predict_with_generate=True` means dev evaluation runs **real autoregressive
decoding** and scores actual MER, rather than just measuring validation
loss. Slower, but it selects checkpoints on the metric we care about.

Dev decoding uses `language=None` so checkpoint selection tracks the same
auto-detect condition the test set is scored under.

### 9.6 The featurization deadlock (a real bug worth knowing)

`os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")` sits at the very
top of `train_whisper.py`, **before any transformers import**. Why:

`datasets.map(num_proc=N)` **forks** the process. The parent has already
initialized HuggingFace's Rust fast tokenizer, which owns a thread pool. A
forked child inherits the pool's state but not its threads — and deadlocks.
The symptom is diabolical: the progress bar sits at `0/N` **with zero CPU
usage** forever. It looks like "slow", it is actually "hung permanently".

Fixed by that env var plus `--num-proc` defaulting to 1. Featurization is
just numpy log-mel plus FLAC decode, so one worker costs only a few minutes.

### 9.7 Our addition: Hub-backed resumable checkpoints

Not in the paper — a response to Kaggle killing sessions with no warning and
`/kaggle/working` dying with them. Implemented in
[`hub_checkpoint.py`](../src/csasr/train/hub_checkpoint.py).

Every `save_steps`, the Trainer's **complete** checkpoint is pushed to a
private Hub repo:

```
model.safetensors   optimizer.pt   scheduler.pt
rng_state.pth       scaler.pt      trainer_state.json
```

Note it is not just weights — optimizer momentum, LR schedule position, RNG
state and `global_step` are all required to resume *identically* rather than
merely restart from the same weights.

**Two checkpoints are tracked per run**, and the reason is subtle:

* `last-checkpoint` — what a resumed run continues *from*.
* `best-checkpoint` — the best-dev-MER one, which `load_best_model_at_end`
  reloads at the end. These are frequently **different** checkpoints, and
  `trainer_state.json` records the best one's path as a path on the *dead
  session's* disk. On a fresh machine that path doesn't exist, so recovering
  it requires its own Hub copy.

**Cleanup happens only after the final model is saved** — deliberately *not*
from an `on_train_end` callback, which fires *before* the final save. Deleting
that early would destroy the one thing a crash during `save_model()` could be
recovered from.

Each push calls `super_squash_history()`, because `upload_folder` creates a
new commit each time and old commits keep their LFS blobs — unsquashed, a
long run would accumulate tens of GB of dead history.

**This was validated in production, not just in tests:** a session was killed
mid-M6, and the next run on a different machine logged
`[train] resuming 'm6' from Hub checkpoint: …` and continued from step 200.

---

## 10. Phase 4 — evaluation

### 10.1 The single most important design decision

**Decode whole recordings, not isolated utterance clips.**

The paper decodes with **WhisperX**, which transcribes a *full recording*:
one language detection, 30-second chunks with real surrounding context.

Decoding the 3,136 isolated ~6-second test clips instead is a different and
much harder task — and it silently destroys what CBA measures. Without
context, Whisper renders English loanwords in Devanagari, so the hypothesis
contains **no script boundary** and no switch bigram can match, for any
model. Measured on real test audio, same model, same audio span:

| decoding | %Latin in hypothesis | MER | CBA-HE |
|---|---|---|---|
| reference | 21.6% | — | — |
| per-clip | **0.0%** | 182.6 | 0.0 |
| recording-level | 12.2% | 103.1 | 1.6 |

This was caught by Gate 3 failing, and it is the clearest demonstration in
the project of why the gates exist. Cost: one 40-minute inference run.
Saved: 6–8 hours of TTS plus 5 hours of training measured against a broken
ruler.

`grouping.py` makes this possible with no extra metadata: MUCS utterance ids
are `<speaker>_<recording>_<index>` and the trailing index is chronological,
verified against the corpus's own `segments` file for all 30 test recordings.

> **⚠ DEVIATION D4 / D8 — engine and granularity.** We use **faster-whisper**
> (the engine WhisperX *wraps*) at recording level, rather than WhisperX
> itself. Same decoding algorithm and heuristics, minus WhisperX's forced
> alignment, which we don't need because we score at recording level. MER is
> a corpus-level ratio so concatenation is equivalent; CBA's denominator
> still comes from reference *utterances*
> ([5.3b](#53-cba--code-switch-bigram-accuracy)).

### 10.2 The decoding heuristics

`decode.py` applies OpenAI's full stack:

* **Beam search** (`beam_size=5`) — keep the 5 highest-probability partial
  hypotheses rather than greedily taking the best token each step.
* **Temperature fallback** `(0.0, 0.2, …, 1.0)` — if a decode looks
  degenerate, retry with progressively more randomness.
* **Compression-ratio threshold (2.4)** — if the output text gzip-compresses
  too well, it is repetitive (`तो तो तो तो…`) and the decode is retried. A
  cheap, elegant loop detector.
* **VAD** (Voice Activity Detection) — segments out actual speech, skipping
  silence.
* **`log_prob_threshold`, `no_speech_threshold`** — confidence gates.

**`--lang-detect-segments 8`** deserves special mention. faster-whisper
detects language from a **single** 30-second window by default. Whisper
confuses Hindi with Urdu constantly — they are effectively the same spoken
language in different scripts — and one bad window sends an entire recording
into Perso-Arabic script, where no Hindi↔English bigram can possibly match
and CBA collapses. Voting over 8 windows fixes it. (Our final eval still
shows `{'hi': 29, 'ur': 1}` — one recording out of 30, 2.9% of reference
words.)

**Batching** (`--batch-size 16`) decodes VAD chunks as a batch rather than
one at a time: measured **8× speedup** (253s → 32s on 181s of audio). This is
not a shortcut — batched VAD inference is precisely what makes WhisperX "70×
realtime", so our original sequential loop was the *less* faithful option.

### 10.3 CTranslate2 — [`ct2.py`](../src/csasr/eval/ct2.py)

faster-whisper runs **CTranslate2** models (a C++ inference engine with
quantization and fused kernels), not raw HuggingFace checkpoints. OpenAI's
sizes have prebuilt conversions; our fine-tuned checkpoints are converted
once and cached.

A real bug lived here and is worth knowing as a *class* of bug: current
`transformers` renamed the file `save_pretrained()` writes from
`preprocessor_config.json` to `processor_config.json`. `ct2.py` had the old
name hardcoded, so converting *our own* checkpoints crashed — and it went
unnoticed for the whole project because Gate 3 and the zero-shot baseline
both use *prebuilt* conversions and never touch that code path. The first
thing that ever exercised it was decoding M6.

The fix checks both names and writes whichever exists under the name
faster-whisper's loader actually reads. That name matters: the file carries
`n_mels`, and while the default (80) is correct for base/small/medium, it
would be silently **wrong** for large-v3 (128 mels).

---

## 11. The gates

A **gate** is a cheap check that must pass before an expensive stage runs.
The design principle: *each gate costs far less than what it protects.*

| Gate | Checks | Cost |
|---|---|---|
| **0** | Table 1 word/bigram counts reproduce | seconds, CPU |
| **1** | Bigram yield ratios track the paper | free |
| **1.5** | Sentence quality, label leaks, projected hours | seconds, CPU |
| **2** | T1 ⊂ T2, durations, spot-listening | free |
| **2b** | T1 ids resolve against the *published* parquet | seconds |
| **3** | large-v2 zero-shot ≈ 52.0 MER | ~15 min GPU |

### Gate 0 — reproduce Table 1

The paper publishes exact word and bigram counts. Reproducing them pins down
tokenization, normalization, and LID **simultaneously**. Test split results:

```
metric        got      paper   rel err  status
words_h    28,219     28,215     0.01%  PASS
words_e     9,152      9,627     4.93%  info
total      37,557     37,842     0.75%  PASS
he_cs       4,125      4,189     1.53%  PASS
eh_cs       5,058      5,176     2.28%  PASS
hours        5.18       5.20     0.42%  PASS
```

`words_e` runs ~5% low, and this is **explained, not a bug**: the paper's
Total (37,842) exceeds this corpus's entire raw whitespace-token count
(37,611), so their tokenizer splits tokens ours does not — most likely
expanding the 186 bare digit tokens into words. We keep digits as `OTHER`
because the paper's own bigram filter requires each bigram to contain "both
Hindi and English characters", and a digit has neither.

**Train does not reproduce, and no tokenization can make it.** We searched
the tokenizer space; the maximum HE bigram count *any* variant produces is
80,860 against the paper's 85,761. Splitting tokens can only *add*
adjacencies — it cannot invent Hindi→English pairs absent from the text. So
the released OpenSLR 104 transcripts simply contain fewer than the paper's
train row implies. This costs the replication nothing, because Track 2 uses
train only for few-shot text and a dev slice.

### Gate 3 — calibrate the instrument

The most conceptually important gate. It runs `whisper-large-v2` **zero-shot**
— no fine-tuning — and compares against the paper's own published zero-shot
row.

**It is not our baseline. It is the calibration of the measuring
instrument.** If decoding, normalization, LID, or either metric's definition
were subtly wrong, this is where it surfaces — because we'd fail to
reproduce a *known* number using the *same* model the paper used.

Our final Gate 3:

```
MER hybrid  51.9    (paper 52.0)   ✓
CBA lenient HE 43.8 (paper 42.9)   ✓
HE denominator 4,125 (paper 4,189) ✓
```

It also caught the per-clip decoding bug described in
[10.1](#101-the-single-most-important-design-decision) — the first run
returned MER 75.6 / CBA-HE 14.7. Since CBA does not depend on the MER
definition, a 3× miss on *both* meant the hypotheses were wrong, not the
metric.

**Note the distinction that confuses people:** Gate 3 uses large-v2 forever,
regardless of what we fine-tune, because it exists to reproduce a *published
number*. The separate "base zero-shot" row is the actual before/after
baseline for M6/M7/M8.

---

## 12. Results and the verdict

### 12.1 What we got

Final evaluation, real MUCS test set, recording-level decoding, hybrid MER,
`adjacent` CBA:

```
system                   MER   paper   CBA-HE   CBA-EH
large-v2 zero-shot      51.9    52.0     20.1     17.6
base zero-shot          95.2       -      0.0      0.0
M6 (T1, 8h)             56.8    48.2      7.1     17.0
M7 (T2, 22h)            57.2    40.8      6.9     15.3
M8 (T2 + mono)          60.6    39.2      1.0      3.0

M6 > M7 > M8 ordering reproduced: False
```

### 12.2 The honest verdict

**The ordering did not reproduce.** MER got *worse* M6→M7→M8
(56.8 → 57.2 → 60.6), the opposite of the paper's 48.2 → 40.8 → 39.2.

Two things are nonetheless clearly true:

1. **Fine-tuning on synthetic data works.** `base zero-shot` scores **95.2
   MER / 0.0 CBA** — the un-fine-tuned model is useless at this task,
   transliterating everything into Devanagari. Every fine-tuned model is
   dramatically better (56.8–60.6, with non-zero CBA). The synthetic data
   demonstrably taught the model to produce mixed-script output.
2. **The measuring instrument is sound.** Gate 3 reproduces the paper to
   0.2%. This is a real result, not an artifact.

### 12.3 Why it did not reproduce — the leading explanation

> **⚠ DEVIATION D15 — whisper-base and the step cap.**
> **Paper / our original plan:** whisper-small, `max_steps=5000`,
> `eval_steps=250`, `patience=4`.
> **Final runs:** `whisper-base` (74M), `max_steps=800`, `eval_steps=200`,
> `patience=3`.
> **Why:** Kaggle's free tier turned out to be a **~6h/day** allowance, not
> the assumed 30h/week pool, against a hard deadline.

This is almost certainly the dominant confound, and the mechanism is worth
understanding precisely.

**The step cap was fixed at 800 while the datasets grew.** Steps, not
epochs, were capped. Kaggle allocated 2× T4, so HF Trainer auto-split the
batch across them:

```
effective_batch = 16 (per device) × 4 (grad accum) × 2 (GPUs) = 128
epochs_seen     ≈ (max_steps × effective_batch) / dataset_size
```

Checked against what the Trainer actually logged (`train_runtime ×
train_samples_per_second` confirms 128 samples/step to within 0.1%):

| | clips | predicted epochs | **logged epoch** |
|---|---|---|---|
| **M6** | 6,069 | 16.87 | **16.67** |
| **M7** | 18,010 | 5.69 | **5.675** |
| **M8** | ~35,069 | 2.92 | **2.921** |

**So the models with more data got proportionally less training on it.** M8
— the model the paper says should be *best* — saw its data **2.9 times**,
while M6 saw its data **16.7 times**. The experiment as run does not isolate
the variable it intends to: "more data" was confounded with "~6× less
exposure per example".

**M8's extra problem.** Its CBA collapsed (7.1/17.0 → 1.0/3.0), a sharper
drop than MER alone explains. M8 adds ~27h of **monolingual** Common Voice
into a run with only 800 total steps. A large share of that fixed budget is
spent on examples containing *no code-switching at all*, actively diluting
exposure to the exact skill CBA measures. The paper had 5,000 steps to
absorb both.

**Secondary suspects**, both pre-registered in the README before results
existed:

* **D14** (alternational vs insertional register) — we trained Whisper's
  decoder-as-LM on the wrong register. Flagged in advance as the leading
  suspect if M6/M7 underperformed.
* **D2** (whisper-base, 74M) — a model 20× smaller than the paper's may
  simply lack capacity to exploit additional data, compressing any real
  M6/M7/M8 differences below noise.

### 12.4 What would make this a fair test

In rough order of expected value:

1. **Scale steps with dataset size** so every model sees comparable epochs
   — or cap by *epochs* rather than steps. This is the confound; fix it
   first.
2. **Restore whisper-small** (or larger) so there is capacity to exploit
   more data.
3. **Fix D14** by adjusting the sentence-generation prompt toward an 88/12
   Hindi-matrix split matching the real corpus, and measuring with
   `grammar_probe.py`. Note this deviates from the paper's verbatim prompt —
   which is itself a defensible experiment about the paper's method.
4. **Ablate M8's monolingual data** — try Hindi-only, or a smaller
   proportion, to test the dilution hypothesis directly.

**None of this makes the paper wrong.** It makes *our run* an underpowered
test of it. Reporting that distinction accurately is the scientifically
honest outcome, and is more useful than a result that accidentally agreed.

---

## 13. Complete deviation table

| # | Paper | Ours | Why | Impact |
|---|---|---|---|---|
| **D1** | Llama-3.3-70B | Gemma-4-26B-A4B MoE (Q4_K_M, llama.cpp) | 70B = 141 GB bf16 | Text quality ↓ |
| **D2** | whisper-large-v2 (1.54B) | whisper-small (244M) → base (74M) | Doesn't fit a T4 | **Absolute MER ↑↑** |
| **D3** | Full FT, AdamW, lr 2e-5, batch 64 | unchanged | small/base FT fits | none |
| **D4** | WhisperX | faster-whisper (engine WhisperX wraps) | Don't need forced alignment | none |
| **D5** | Common Voice official | `fsicoli/common_voice_17_0` mirror | Mozilla emptied the repo Oct 2025 | none |
| **D6** | 4h real dev slice | same + synthetic dev logged | Paper's dev is a mild "no real data" leak | none |
| **D7** | 15h CV Hindi | 11.87h | Only ~20.6h exists in CV17 | M8 only |
| **D8** | Per-utterance MER/CBA | Recording-level | Forced by D4 | none (MER is a ratio) |
| **D9** | LLM emits true bigrams | Extract switch pair from longer phrases | Smaller model appends a 3rd word | none (prompt verbatim) |
| **D10** | Train_T2 = 22h | ~25.1h | Resolved by D13 | contrast *stronger* |
| **D11** | Synthetic matches domain | 47% Latin vs 25%; 2.0 vs 3.15 switches | Prompt-driven | quality ↓ |
| **D12** | — | Model swap did **not** fix D11 | Negative result: it's the prompt, not capacity | documented |
| **D13** | 4,466 bigram calls | 15,000 | Smaller model repeats itself | now exceeds paper |
| **D14** | Insertional CS | Alternational CS | **Inherited from paper's own prompt** | suspect for MER |
| **D15** | 5,000 steps, whisper-small | 800 steps, whisper-base | ~6h/day quota + deadline | **primary confound** |

---

## 14. Questions you should be able to answer

Use these to check your understanding. Each links back to the relevant part.

**Concepts**
1. What is code-switching, and what is the difference between insertional and alternational? → [Part 2](#2-foundations-code-switching)
2. Why can we identify Hindi vs English by script alone, and where does that break? → [4.1](#41-language-id-by-script--srccsasrlidpy)
3. Why does a 4-second clip cost the same encoder compute as a 30-second one? → [3.1](#31-audio--log-mel-spectrogram)

**Metrics**
4. Write the MER formula. Why can it exceed 100%? → [5.2](#52-wer--mer)
5. Why does CBA need multiset rather than set semantics? → [5.3](#53-cba--code-switch-bigram-accuracy)
6. Two hypotheses have identical MER but very different CBA. How? → [5.3](#53-cba--code-switch-bigram-accuracy) + [2](#2-foundations-code-switching)
7. Why is CBA's denominator taken per-utterance when we decode per-recording? → [5.3b](#53-cba--code-switch-bigram-accuracy)

**Traps**
8. What does `BasicTextNormalizer` do to `दस्तावेज़`, and why does it matter? → [4.2](#42-normalization--srccsasrnormalizepy)
9. Why must label padding be `-100` and not the pad token id? → [9.2](#92-the-collator--collatorpy)
10. Why is `TOKENIZERS_PARALLELISM=false` set before any import? → [9.6](#96-the-featurization-deadlock-a-real-bug-worth-knowing)
11. Why SHA-1 instead of `hash()` for speaker assignment? → [8.2](#82-description-conditioned-tts)

**Design**
12. Why are `01a` and `01b` separate notebooks? → [8.1](#81-why-this-is-a-separate-notebook)
13. Why does every stage read and write JSONL instead of passing objects? → [6.3](#63-the-manifest-contract--manifestpy)
14. Why decode whole recordings instead of clips? What did per-clip decoding do to CBA? → [10.1](#101-the-single-most-important-design-decision)
15. Why does Gate 3 use large-v2 even though we fine-tune base? → [Part 11](#11-the-gates)
16. Why track *two* Hub checkpoints per run instead of one? → [9.7](#97-our-addition-hub-backed-resumable-checkpoints)

**Results**
17. Did the paper's claim reproduce? What *did* reproduce? → [12.2](#122-the-honest-verdict)
18. Compute epochs-seen for M6/M7/M8 and explain why that invalidates the comparison. → [12.3](#123-why-it-did-not-reproduce--the-leading-explanation)
19. Why did M8's CBA collapse specifically, more than its MER? → [12.3](#123-why-it-did-not-reproduce--the-leading-explanation)
20. Would LoRA have closed the gap to the paper's numbers? Why not? → [9.3](#93-the-optimizer-recipe) + [D2](#34-model-sizes)
