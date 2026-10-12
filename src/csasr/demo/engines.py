"""Load a demo model and transcribe audio with the settings the ablation was scored with.

Whisper models run on faster-whisper (CTranslate2) with the decoder settings of
`eval/decode.py`: language detected once per recording by voting over windows,
temperature fallback, VAD-chunked batches, beam 5. Fine-tuned checkpoints are
converted on first use by `eval/ct2.py` (LoRA/QLoRA adapters are merged first) and
cached in `ct2_models/`; expect a few minutes for that first load, then seconds.

MMS-1B is greedy CTC over overlapping windows, shared with `eval/decode_ctc.py`.

Device policy: the login node's two GPUs belong to other users' jobs and are
visible without an allocation, so `auto` picks CUDA only inside a SLURM job.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .registry import REPO, DemoModel

__all__ = ["Result", "Engine", "load_audio", "pick_device", "load_engine"]

SR = 16_000
CT2_CACHE = REPO / "ct2_models"


@dataclass
class Result:
    text: str
    seconds: float                 # wall-clock decode time (not counting model load)
    language: str | None = None    # Whisper's detected language
    language_prob: float | None = None


def load_audio(path: str | Path) -> np.ndarray:
    """Any common audio format -> 16 kHz mono float32."""
    import librosa

    audio, _ = librosa.load(str(path), sr=SR, mono=True)
    return audio.astype(np.float32)


def pick_device(requested: str = "auto") -> str:
    """'auto' | 'cpu' | 'cuda' -> 'cpu' or 'cuda'."""
    in_job = bool(os.environ.get("SLURM_JOB_ID"))
    if requested == "cpu" or (requested == "auto" and not in_job):
        return "cpu"          # decided without importing torch, so the caller can hide GPUs
    import torch

    have = torch.cuda.is_available()
    if requested == "cuda":
        if not have:
            raise SystemExit("--device cuda requested but no GPU is visible")
        if not in_job:
            print("[demo] warning: not inside a SLURM allocation; this GPU may be "
                  "running other users' jobs", file=sys.stderr)
        return "cuda"
    return "cuda" if have else "cpu"


class Engine:
    """One loaded model. `transcribe` takes 16 kHz mono float32."""

    model: DemoModel
    device: str

    def transcribe(self, audio: np.ndarray) -> Result:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        pass


class WhisperEngine(Engine):
    def __init__(self, model: DemoModel, device: str, *, threads: int, beams: int):
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        from ..eval.ct2 import resolve_ct2

        self.model, self.device, self.beams = model, device, beams
        path = resolve_ct2(model.path, cache_dir=CT2_CACHE, quantization="float16",
                           token=os.environ.get("HF_TOKEN"))
        self._wm = WhisperModel(
            path, device=device, compute_type="float16" if device == "cuda" else "int8",
            cpu_threads=0 if device == "cuda" else threads)
        self._engine = BatchedInferencePipeline(model=self._wm)
        self._batch = 16 if device == "cuda" else 4

    def transcribe(self, audio: np.ndarray) -> Result:
        # Settings and thresholds are imported from the evaluation decoder so the
        # demo and the reported numbers cannot drift apart.
        from ..eval.decode import (COMPRESSION_RATIO_THRESHOLD, LOGPROB_THRESHOLD,
                                   NO_SPEECH_THRESHOLD, TEMPERATURES)

        t0 = time.monotonic()
        segments, info = self._engine.transcribe(
            audio, language=None, task="transcribe", beam_size=self.beams,
            temperature=list(TEMPERATURES),
            compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
            log_prob_threshold=LOGPROB_THRESHOLD, no_speech_threshold=NO_SPEECH_THRESHOLD,
            vad_filter=True, language_detection_segments=8, language_detection_threshold=0.5,
            batch_size=self._batch)
        text = " ".join(s.text.strip() for s in segments).strip()
        return Result(text, time.monotonic() - t0, info.language,
                      round(float(info.language_probability or 0.0), 3))

    def close(self) -> None:
        del self._engine, self._wm
        gc.collect()


class CtcEngine(Engine):
    def __init__(self, model: DemoModel, device: str, *, threads: int, beams: int):
        import torch

        self.model, self.device = model, device
        if device == "cpu":
            torch.set_num_threads(threads)
        if model.zero_shot:
            from transformers import AutoProcessor, Wav2Vec2ForCTC

            net = Wav2Vec2ForCTC.from_pretrained(
                model.path, target_lang=model.target_lang, ignore_mismatched_sizes=True)
            self._processor = AutoProcessor.from_pretrained(
                model.path, target_lang=model.target_lang)
        else:
            from ..train.train_ctc import load_ctc_model

            net, self._processor = load_ctc_model(model.path)
        self._dtype = torch.float16 if device == "cuda" else torch.float32
        self._net = net.to("cuda" if device == "cuda" else "cpu", self._dtype).eval()

    def transcribe(self, audio: np.ndarray) -> Result:
        from ..eval.decode_ctc import FRAME, ctc_transcribe

        t0 = time.monotonic()
        text = ctc_transcribe(
            audio, self._net, self._processor, "cuda" if self.device == "cuda" else "cpu",
            self._dtype, chunk=30 * SR // FRAME * FRAME, stride=5 * SR // FRAME * FRAME,
            batch_size=8)
        return Result(text.strip(), time.monotonic() - t0)

    def close(self) -> None:
        import torch

        del self._net
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()


def load_engine(model: DemoModel, device: str, *, threads: int = 8, beams: int = 5) -> Engine:
    cls = CtcEngine if model.kind == "ctc" else WhisperEngine
    return cls(model, device, threads=threads, beams=beams)
