# Ablation study — progress and handover

Last updated: **8 Oct 2026**. Presentation: **15 Oct 2026**.
Branch `ablation`. **All GPU work is done**: job 12578 (the three new-corpus runs plus
the report) completed 7 Oct 22:35 with exit 0. Only the slides remain.

Read this first in a new session. The study design is in the README's "Ablation
study" section; the cluster's quirks are in `../.claude/skills/slurm-scripts/SKILL.md`.

## Status in one paragraph

The ASR-model × fine-tuning-method grid is **complete** (12 runs + 4 zero-shot rows),
and it changes the story: with the checkpoint chosen early, **every fine-tuned model
beats zero-shot**, most by a wide margin. The earlier replication failure was
over-training, not a broken pipeline. The LLM / prompt axis is **complete** too: the
three new corpora trained after the host-memory fix. Swapping Gemma-4-26B for 31B changes
little (47.8 vs 45.2 MER), but fixing the script mix does a lot: the matched prompt (32.6)
and Sarvam-30B (32.0) both beat the paper's M7 (40.8). Sarvam-105B produced no usable
text and was **dropped by decision (6 Oct)**; the study uses three LLMs.

| Piece | State |
|---|---|
| Zero-shot rows (4 models) | done |
| Whisper small / medium / large-v2 × full / LoRA / QLoRA on the existing Gemma-4-26B corpus | done (9 runs) |
| LoRA learning-rate check (whisper-small, lr 1e-4) | done |
| MMS-1B full and LoRA | done |
| Text + 22 h audio: Gemma-4-31B, Sarvam-30B, Gemma-4-31B matched prompt | done |
| Training on those three corpora | done (job 12578 after the loader fix) |
| Sarvam-105B | **dropped (decision 6 Oct)**: failed the bigram smoke gate |
| Report and figures | final (`results/ablation/`), regenerated 8 Oct |
| Slides | **not started** |

## Results so far

Test set: MUCS, 30 recordings, hybrid MER, lenient CBA. Intervals are 95% bootstrap
over recordings; Δ is paired against the same model zero-shot. Full table:
`results/ablation/report.md`. Training text: existing Gemma-4-26B-A4B corpus cut to 22 h.

| Model | Method | MER | Δ vs zero-shot | CBA-HE | % Latin | Selected step |
|---|---|---|---|---|---|---|
| Whisper small | zero-shot | 67.2 | – | 8.2 | 4.1 | – |
| | full | 49.5 | −17.7 [−21.2, −14.6] | 52.1 | 21.0 | 50 |
| | LoRA | 55.1 | −12.0 [−14.5, −10.0] | 31.4 | 13.7 | 100 |
| | QLoRA | 54.2 | −12.9 [−15.5, −10.8] | 33.1 | 14.8 | 100 |
| Whisper medium | zero-shot | 55.2 | – | 43.4 | 16.8 | – |
| | full | 42.3 | −12.9 [−19.6, −7.4] | 63.6 | 24.7 | 50 |
| | LoRA | 42.1 | −13.1 [−19.7, −7.6] | 65.5 | 25.6 | 50 |
| | QLoRA | **40.7** | −14.5 [−21.2, −8.8] | **68.5** | 26.9 | 50 |
| Whisper large-v2 | zero-shot | 51.8 | – | 44.1 | 17.0 | – |
| | full (anchor) | 45.2 | −6.6 [−10.0, −4.1] | 51.2 | 20.4 | 50 |
| | LoRA | 45.0 | −6.8 [−10.4, −4.2] | 54.0 | 22.0 | 50 |
| | QLoRA | 45.4 | −6.4 [−9.9, −3.8] | 52.8 | 21.1 | 50 |
| MMS-1B | zero-shot | 50.0 | – | 0.1 | 0.8 | – |
| | full | 45.6 | −4.4 [−8.9, −0.3] | 3.4 | 23.3 | 50 |
| | LoRA | 44.8 | −5.2 [−11.5, +0.6] | 9.5 | 30.6 | 50 |

LLM / prompt axis (Whisper large-v2, full, 22 h each; zero-shot 51.8):

| Corpus | MER | Δ vs zero-shot | CBA-HE | % Latin | Selected step |
|---|---|---|---|---|---|
| Gemma-4-26B-A4B (existing) | 45.2 | −6.6 [−10.0, −4.1] | 51.2 | 20.4 | 50 |
| Gemma-4-31B | 47.8 | −4.0 [−7.3, −1.7] | 45.7 | 17.2 | 50 |
| Gemma-4-31B, matched prompt | **32.6** | −19.1 [−25.5, −13.5] | 79.1 | 31.7 | 50 |
| Sarvam-30B | **32.0** | −19.8 [−26.3, −13.9] | 79.6 | 31.9 | 50 |

Reference: 24.3% Latin. Paper: zero-shot 52.0, M7 (large-v2, 22 h synthetic) 40.8 / CBA-HE 52.3.
LoRA check: whisper-small LoRA at lr 1e-4 gives 53.6 (selected step 200) against 55.1 at lr 1e-3.

### What the dev curves show

Dev MER by checkpoint, large-v2 full (same decoder as test):

| Step | 0 | 50 | 100 | 200 | 400 | 600 | 800 | 1000 | 1200 |
|---|---|---|---|---|---|---|---|---|---|
| MER | 53.2 | **46.2** | 46.6 | 58.8 | 63.0 | 57.8 | 58.5 | 58.6 | 57.8 |
| % Latin | 16.9 | 19.1 | 20.2 | 10.4 | 4.9 | 9.6 | 8.7 | 8.5 | 9.9 |

Every run has this shape (`results/ablation/fig_dev_curves.png`): a sharp gain at
step 50–100, then by step 200–400 the model is worse than zero-shot and stays there,
with the Latin share collapsing to the 9% of the synthetic Hindi-matrix text. Step 50
is 3,200 utterances (0.19 epoch) with the learning rate still in warm-up (5e-6).

### Findings worth a slide

1. **The replication failure was over-training.** The M6/M7/M8 runs evaluated first at step 250 and trained to 2,000+, entirely inside the degraded regime.
2. **Early-stopped, synthetic data helps every model**, significantly, in MER and in switch-point accuracy.
3. **Whisper medium + QLoRA (40.7 MER, CBA-HE 68.5) matches the paper's M7 MER (40.8) and beats its CBA (52.3)**, on a smaller model. Large-v2 reaches 45.2, short of the paper's 40.8.
4. **Fine-tuning method barely matters at medium and large**: full, LoRA and QLoRA are within each other's intervals, while QLoRA needs a third of the memory (large-v2: 11 GB against 34 GB). At small, full fine-tuning is clearly better.
5. **MMS-1B improves in MER but not at switch points** (CBA stays under 10%): it starts writing Latin script, but with no decoder language model it does not get the words right.
6. **Script mix, not the LLM, drives the result.** Gemma-4-31B writes the same
   9%-Latin Hindi-matrix text as 26B and trains to about the same place (47.8 vs 45.2). The
   matched prompt and Sarvam-30B both reach about 23% Latin and about 32 MER, 13 points
   better than the paper-prompt corpora and 8.5 better than the paper's M7 (40.8).
   Their CBA-HE is 79 against 52 for M7. Sarvam gets there with the paper-verbatim
   prompt, so it is not a prompt-leak effect.
7. **Text statistics:**

| Corpus | % Latin in Hindi-matrix sentences | % Hindi-matrix | Test switch-bigram coverage | LLM calls |
|---|---|---|---|---|
| MUCS test (target) | 20.0 | 87.1 | 100 | – |
| Gemma-4-26B-A4B (existing) | 9.3 | 58.5 | 28.9 | 15,000 |
| Gemma-4-31B | 8.9 | 53.0 | 34.4 | 2,000 |
| Gemma-4-31B, matched prompt | 22.8 | 99.7 | 47.5 | – |
| Sarvam-30B | 22.6 | 67.9 | 49.4 | 15,000 |

   The matched prompt does what it was written to do. Sarvam-30B gets the script mix
   right on its own but needed 7.5× the calls, and about a third of its otherwise
   valid sentences were dropped: 21% as leaked reasoning, romanised Hindi or markup,
   14% as output beyond the four sentences asked for. The matched-prompt corpus
   lost 0.4% to the same filter.

### Caveats to state alongside the numbers

- **Step 50 is the earliest checkpoint saved**, so the optimum sits on the edge of the grid; the true best could be earlier or between 50 and 100. See "Optional" below.
- **Selection uses 4 h of real dev audio**, as the paper does. That is a mild leak against "no real data" (README deviation D6).
- **The matched prompt uses real MUCS *train* transcripts as few-shot examples** (not dev or test, so no test leakage). It does use real in-domain text, which stretches the "no real data" framing; say so next to the 32.6. Sarvam-30B's 32.0 uses the paper prompt and needs no such caveat.
- **Sarvam-30B and matched-prompt are within noise of each other** (32.0 vs 32.6); both are clearly separated from the other two corpora.
- **One seed per cell.** Differences inside the intervals (e.g. LoRA vs QLoRA at large) are not findings.
- **LoRA's learning rate interacts with the warm-up.** At a fixed step LoRA and full fine-tuning have had different effective updates; the small-model gap (49.5 vs 54–55) is partly this, as the lr 1e-4 row shows.
- **Peak VRAM is comparable within a model size only**: small runs at batch 64 without gradient checkpointing (64 GB), medium and large with it.
- **The s30 row has no train time or peak VRAM** (shown as `n/a †`). That run was trained to step 1,200 before job 12578; the job only resumed from checkpoint-1200, so `run_meta.json` recorded a 28 s final leg. The original timing was not recoverable. Use the g31 run (2.06 h, 33.9 GB) as the reference for the same data volume and settings. `ablation_report.py` now prints `n/a †` for any run with `resumed: true`.
- **"Train h" is the full 1,200 steps**; the selected step-50 model costs about 5 minutes of training.
- **The existing Gemma-4-26B corpus was not re-filtered**; 1.1% of its sentences would fail the sentence filter the new corpora went through.

## What failed, and why

### 1. Training on the new corpora: host out-of-memory — FIXED 6 Oct, rerun done (job 12578)

`g31`, `s30` and `g31m` × whisper-large-v2 × full each died about 17 minutes in, during
`featurize train` (`RuntimeError: One of the subprocesses has abruptly died during map
operation`; SLURM reports `oom_kill`, job limit 48 GB). Both jobs hit it.

Cause (confirmed 6 Oct: a `map(num_proc=2)` on the old loader's dataset had 0 cache files,
on the new loader's 2): the new corpora
are read from a local manifest, and `load_manifest_dataset` in
`src/csasr/data/loaders.py` builds the dataset with `Dataset.from_list`, which is
in-memory. `datasets.map` on an in-memory dataset keeps its output in RAM instead of
an Arrow cache file: 16–22k clips × 0.96 MB of log-mel features, duplicated across 4
workers and the parent. The existing corpus never hit this because it loads from the
Hub's parquet, which is disk-backed.

Fix: make the manifest dataset disk-backed, e.g.

```python
ds = load_dataset("json", data_files=str(path), split="train")
ds = ds.rename_column("wav", "audio").select_columns(["audio", "text", "utt_id"])
return ds.cast_column("audio", Audio(sampling_rate=SR))
```

Applied in `src/csasr/data/loaders.py`, keeping the old `wav` filter and the empty-manifest
error. 49 tests pass (`test_tts_and_data`, `test_pipeline_integration`, `test_ablation`), and
the real g31 manifest loads disk-backed at 333 MB peak RSS. **Verified in job 12578:** no OOM; all three runs completed. `train_ctc.py` is not affected (it
featurizes lazily). The feature caches (~15-21 GB per corpus) go to the HF datasets cache.

### 2. Sarvam-105B: no text — DROPPED by decision on 6 Oct (kept as a finding)

The model loaded (8 slots, 64 GB) and answered in ~1.4 s, but for the paper's bigram
prompt it returned 36 monolingual phrases per call — English-only (`document
management`, `file transfer`) then Hindi-only, some transliterated (`यूएसबी फ्लैश
ड्राइव`). 0 of 143 phrases carried a switch point, so the smoke gate stopped the run.
Thinking was disabled, as for every other LLM.

Note the gate saw one draw repeated four times (`smoke.py` sends the same prompt with
the same seed), so this is one sample, not four. Options, cheapest first:

- probe it with ~20 different prompts (5 min of GPU) to see whether the failure is general;
- allow it to think (`--reasoning-budget` above 0 and a larger `max_tokens`), which changes the condition relative to the other LLMs and must be reported as such;
- report it as a finding (does not follow the bigram instruction in no-thinking mode at Q4_K_M) and present three LLMs.

## Next steps, in order

1. ~~Apply and test the loader fix~~ — done.
2. ~~Run the three blocked training runs~~ — **done, job 12578** (finished 7 Oct 22:35, exit 0). Command used:
   ```bash
   cd /dist_home/d_palmani/c-08/codeswitching
   STAGES="llm report" LLMS="g31 s30" MATCHED_LLM=g31 sbatch scripts/ablate.sh
   ```
   Finished units are skipped; the audio is already on disk. Submitting the same line
   twice lets two GPUs split the three runs.
3. ~~Decide Sarvam-105B~~ — dropped. Say on the slides that it was tried and excluded: with
   thinking disabled at Q4_K_M it returned monolingual phrases (0 of 143 had a switch point).
   `ablate.sh` no longer includes it by default; `LLMS="s105"` still opts in,
   but the weights are deleted, so it would re-download 60 GB (about 1.3 h at login-node speed)
   before the roughly 3 h text + 4 h TTS + 4 h training.
4. ~~Regenerate the report~~ — done; rerun only if rows change (CPU, a minute):
   `~/.venvs/csasr-abl/bin/python3 scripts/ablation_report.py --results-dir results/ablation --exp csasr-work/abl`
5. **Slides.** Figures are in `results/ablation/fig_*.png`. The `fig_grid_mer.png` label
   collision is fixed (6 Oct) in `scripts/ablation_report.py`; the three new-corpus runs will
   have added the LLM / prompt rows (`fig_llm_axis.png`, `fig_text_latin.png`).

### Optional, if GPU time allows (each strengthens a caveat above)

- **Finer early checkpoints.** The optimum is at the first saved step. A run to 200 steps saving at 10, 25, 50, 75, 100, 150 costs ~20 min of training plus selection. `train_whisper.py` already takes `--save-at`; `ablate.sh` does not expose it yet (needs a knob and a separate run tag so it does not collide with the finished runs).
- **Seeds.** Tags are supported: `RUNS="g26__whisper-medium__qlora__seed1 g26__whisper-medium__qlora__seed2" sbatch scripts/ablate.sh`.
- **Re-filter the existing corpus** for strict equality with the new ones (means rerunning the grid; probably not worth it for 1.1%).

## Where everything is

| What | Where |
|---|---|
| Runner (3-day walltime, self-resubmitting, lock + `.done` per unit) | `scripts/ablate.sh` |
| Results: one row per run, test hypotheses, report, CSV, figures | `results/ablation/` (`rows/`, `hyps/`, `report.md`, `ablation.csv`, `fig_*.png`, `text_stats.json`) |
| Run directories (selected checkpoint, dev curves, `run_meta.json`) | `csasr-work/abl/<corpus>__<asr>__<ft>[__tag]/` |
| New corpora: text, LLM response caches, 16 kHz audio, 22 h manifest | `data/ablation/{g31,s30,g31m}/` (`train.jsonl`, `audio/`, `sentences.jsonl`, `cache/`) |
| Existing corpus's 22 h subset | `data/ablation/g26/ids.json` |
| GGUF weights (Gemma-4-31B, Sarvam-30B) | `csasr-work/gguf/` (~38 GB; the 60 GB Sarvam-105B weights were deleted 6 Oct) |
| llama.cpp server, built for A100 and RTX 6000 Ada | `csasr-work/llama.cpp/build/bin/llama-server` |
| Job logs | `slurm-ablate-<jobid>.out`; LLM server and TTS logs in `csasr-work/logs/` |
| Environments | `~/.venvs/csasr-abl` (training, eval, PEFT), `~/.venvs/csasr-tts` (Parler-TTS) |
| Smoke-test outputs (not results) | `results/ablation-smoke/`, `csasr-work/abl-smoke/`, `data/ablation-smoke/` |

Run names are `<corpus>__<asr>__<ft>[__tag]`: corpus `g26 | g31 | s30 | s105 | g31m`
(`m` = matched prompt), asr `whisper-small | whisper-medium | whisper-large-v2 | mms-1b`,
ft `full | lora | qlora`, tag `lr<value>` or `seed<n>`.

`results/` and `data/` are git-ignored. Nothing in them is committed; if the results
should be versioned, add them explicitly.

## Job history

| Job | What | Outcome |
|---|---|---|
| 12508 | smoke test | found two faults (llama.cpp built for the wrong GPU; whisper-medium OOM at batch 32), both fixed |
| 12509, 12510, 12513 | first grid and text jobs | cancelled by me to pick up fixes (early checkpoints, sentence filter, hours-based text stopping rule) |
| 12514 | grid | 29.5 h; all 12 grid + MMS runs done; the 3 new-corpus runs OOM |
| 12519 | text + TTS | 22.4 h; g31, s30, g31m text and audio done; s105 failed its smoke gate; the 3 new-corpus runs OOM |
| 12578 | the 3 new-corpus runs + report, after the loader fix | 6.4 h, exit 0, 7 Oct 22:35. s30 had already been trained; this job resumed it and ran selection and decoding for s30 and g31m |

Measured costs on one A100-80GB: large-v2 run ≈ 4.2 h (2.0 h training + ~2.2 h selecting
among 8 checkpoints), medium ≈ 2.9 h, small ≈ 1.3 h, MMS ≈ 0.9 h; text ≈ 2.5–3 h per
LLM; TTS ≈ 4 h per 25 h corpus.

## Changes made to the method along the way

- **Checkpoints are selected after training**, on dev, with the recording-level decoder used for test (`src/csasr/eval/select.py`). The in-training metric was unstable and cost 18 minutes per eval.
- **Extra checkpoints at steps 50 and 100**, added after the smoke run showed gains within 20 steps.
- **Every run starts from the public checkpoint**; M7 and M8 had been chained.
- **Every corpus is cut to 22 h.**
- **Sentence filter for all new corpora** (`looks_spoken` in `src/csasr/llm/gen_sentences.py`): rejects task commentary, markup, romanised Hindi; at most four sentences per response.
- **Text generation stops on usable hours** (24 h projected), in doubling rounds up to 15,000 bigram calls, not on bigram count.
- **Matched sentence prompt** (`src/csasr/llm/prompts.py`): four Hindi-matrix sentences, few-shot on real MUCS transcripts. The paper-verbatim prompt is untouched and remains the anchor.
