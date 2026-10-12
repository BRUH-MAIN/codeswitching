"""Name the trained models, so a demo can ask for them by a short key.

A key such as `medium-qlora` stands for one ablation run: Whisper medium, QLoRA,
trained on the existing Gemma-4-26B corpus. The checkpoint used is the one the
ablation *selected* on dev (`hyp_test.step`, written by `eval/select.py`) -- not
the last one. The last checkpoints are over-trained and score worse than
zero-shot, so loading them would demo the wrong thing.

Run directories follow `<corpus>__<asr>__<ft>` (see ABLATION_PROGRESS.md); the
test numbers shown next to each model come from `results/ablation/ablation.jsonl`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["DemoModel", "PRESETS", "REPO", "all_models", "resolve", "model_table"]

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "csasr-work" / "abl"
ROWS = REPO / "results" / "ablation" / "ablation.jsonl"

#: short size name -> (ablation asr id, hub id of the public checkpoint, display name)
SIZES = {
    "small": ("whisper-small", "openai/whisper-small", "Whisper small"),
    "medium": ("whisper-medium", "openai/whisper-medium", "Whisper medium"),
    "large": ("whisper-large-v2", "openai/whisper-large-v2", "Whisper large-v2"),
    "mms": ("mms-1b", "facebook/mms-1b-all", "MMS-1B"),
}
FT_LABEL = {"full": "full fine-tune", "lora": "LoRA", "qlora": "QLoRA"}
#: text corpus -> how to describe the synthetic text the model was trained on
CORPUS_LABEL = {
    "g26": "Gemma-4 26B text",
    "g31": "Gemma-4 31B text",
    "g31m": "Gemma-4 31B text, matched prompt",
    "s30": "Sarvam-30B text",
}
#: the large-v2 runs on the other three text corpora
CORPUS_KEYS = {"large-gemma31": "g31", "large-matched": "g31m", "large-sarvam30": "s30"}

#: named sets, in the order they should be shown
PRESETS = {
    "quick": ["zeroshot-medium", "medium-qlora"],
    "methods": ["zeroshot-medium", "medium-full", "medium-lora", "medium-qlora"],
    "asr": ["zeroshot-large", "large-full", "zeroshot-medium", "medium-full",
            "zeroshot-small", "small-full", "zeroshot-mms", "mms-full"],
    "story": ["zeroshot-large", "large-full", "large-gemma31", "large-matched",
              "large-sarvam30"],
}


@dataclass(frozen=True)
class DemoModel:
    key: str
    label: str
    kind: str                 # "whisper" | "ctc"
    path: str                 # hub id (zero-shot) or checkpoint directory
    run: str                  # ablation run id; `zeroshot__<asr>` for the baselines
    zero_shot: bool
    step: int | None = None   # selected checkpoint (fine-tuned only)
    mer: float | None = None  # test MER from the ablation report
    cba_he: float | None = None
    target_lang: str | None = None   # MMS zero-shot only: which language adapter

    @property
    def available(self) -> bool:
        return self.zero_shot or Path(self.path).is_dir()

    @property
    def short(self) -> str:
        """Compact label for table columns."""
        return self.label.replace("Whisper ", "").replace(" · ", " ")


def _rows() -> dict[str, dict]:
    if not ROWS.exists():
        return {}
    return {r["run"]: r for r in map(json.loads, ROWS.read_text(encoding="utf-8").splitlines())
            if r.get("run")}


def _selected_step(run: str, rows: dict[str, dict]) -> int | None:
    f = EXP / run / "hyp_test.step"
    if f.exists():
        try:
            return int(f.read_text().strip())
        except ValueError:
            pass
    step = rows.get(run, {}).get("selected_step")
    return int(step) if step is not None else None


def all_models() -> dict[str, DemoModel]:
    """Every model the ablation can demonstrate, keyed by short name (insertion order)."""
    rows = _rows()
    out: dict[str, DemoModel] = {}

    def add(key, label, asr, run, hub, ft, corpus):
        kind = "ctc" if asr == "mms-1b" else "whisper"
        row = rows.get(run, {})
        if ft == "zeroshot":
            out[key] = DemoModel(
                key, f"{label} · zero-shot", kind, hub, run, True, mer=row.get("mer"),
                cba_he=row.get("cba_he"), target_lang="hin" if kind == "ctc" else None)
            return
        step = _selected_step(run, rows)
        path = str(EXP / run / f"checkpoint-{step}") if step is not None else str(EXP / run)
        text = "" if corpus == "g26" else f" · {CORPUS_LABEL[corpus]}"
        out[key] = DemoModel(
            key, f"{label} · {FT_LABEL[ft]}{text}", kind, path, run, False, step=step,
            mer=row.get("mer"), cba_he=row.get("cba_he"))

    for size, (asr, hub, label) in SIZES.items():
        add(f"zeroshot-{size}", label, asr, f"zeroshot__{asr}", hub, "zeroshot", None)
    for size, (asr, hub, label) in SIZES.items():
        for ft in (("full", "lora") if size == "mms" else ("full", "lora", "qlora")):
            add(f"{size}-{ft}", label, asr, f"g26__{asr}__{ft}", hub, ft, "g26")
    for key, corpus in CORPUS_KEYS.items():
        asr, hub, label = SIZES["large"]
        add(key, label, asr, f"{corpus}__{asr}__full", hub, "full", corpus)
    return out


def resolve(names: list[str], *, require_available: bool = True) -> list[DemoModel]:
    """Expand presets, validate, and de-duplicate while keeping order.

    `all` means every model whose checkpoint is on disk; `best` is the available
    fine-tuned model with the lowest test MER.
    """
    models = all_models()
    keys: list[str] = []
    for n in names:
        if n in PRESETS:
            keys += PRESETS[n]
        elif n == "all":
            keys += [k for k, m in models.items() if m.available]
        elif n == "best":
            ft = [m for m in models.values() if not m.zero_shot and m.available
                  and m.mer is not None]
            if not ft:
                raise SystemExit("no fine-tuned model with a recorded MER is available")
            keys.append(min(ft, key=lambda m: m.mer).key)
        elif n in models:
            keys.append(n)
        else:
            raise SystemExit(
                f"unknown model {n!r}.\n  models:  {', '.join(models)}\n"
                f"  presets: {', '.join(PRESETS)}, all, best")
    picked = [models[k] for k in dict.fromkeys(keys)]
    if require_available:
        missing = [m for m in picked if not m.available]
        if missing:
            raise SystemExit(
                "checkpoint not found for: " + ", ".join(f"{m.key} ({m.path})" for m in missing)
                + "\n  run `python -m csasr.demo models` to see what is on disk")
    return picked


def model_table() -> list[dict]:
    return [{"key": m.key, "label": m.label, "kind": m.kind, "available": m.available,
             "step": m.step, "mer": m.mer, "cba_he": m.cba_he, "run": m.run}
            for m in all_models().values()]
