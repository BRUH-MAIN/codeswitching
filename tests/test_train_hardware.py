"""The A100 path: hardware planning and config plumbing.

`_plan_hardware` decides precision, TF32 and gradient checkpointing from the
detected device. A wrong choice here does not fail loudly -- it shows up as a NaN
loss (fp16 where bf16 was needed) or an OOM at the first eval, 250 steps in. So
the decision table is asserted rather than trusted.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from csasr.train.train_whisper import (
    BYTES_PER_PARAM,
    CHECKPOINT_VRAM_FRACTION,
    _apply_config_defaults,
    _build_parser,
    _plan_hardware,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

SMALL = 244_000_000
LARGE_V2 = 1_540_000_000


class FakeCuda:
    """Minimal stand-in for torch.cuda: a named card with a compute capability."""

    def __init__(self, name, major, minor, gb, bf16=None):
        self._props = SimpleNamespace(
            name=name, major=major, minor=minor, total_memory=int(gb * 1e9)
        )
        self._bf16 = (major, minor) >= (8, 0) if bf16 is None else bf16

    def is_available(self):
        return True

    def get_device_properties(self, _i):
        return self._props

    def is_bf16_supported(self):
        return self._bf16


class FakeTorch:
    def __init__(self, cuda):
        self.cuda = cuda
        # _plan_hardware writes allow_tf32 through these; capture the writes.
        self.backends = SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
            cudnn=SimpleNamespace(allow_tf32=False),
        )


def args(**kw):
    base = dict(precision="auto", tf32="auto", gradient_checkpointing="auto")
    base.update(kw)
    return SimpleNamespace(**base)


T4 = FakeCuda("Tesla T4", 7, 5, 16)
A100_40 = FakeCuda("NVIDIA A100-SXM4-40GB", 8, 0, 40)
A100_80 = FakeCuda("NVIDIA A100-SXM4-80GB", 8, 0, 80)


def test_t4_gets_fp16_no_tf32():
    """Turing has neither bf16 nor TF32. Choosing bf16 here would raise at best
    and silently produce garbage at worst."""
    hw = _plan_hardware(args(), FakeTorch(T4), SMALL)
    assert hw["fp16"] is True
    assert hw["bf16"] is False
    assert hw["tf32"] is False


def test_a100_gets_bf16_and_tf32():
    hw = _plan_hardware(args(), FakeTorch(A100_40), LARGE_V2)
    assert hw["bf16"] is True
    assert hw["fp16"] is False
    assert hw["tf32"] is True


def test_tf32_is_actually_written_to_torch_backends():
    """Returning tf32=True is not enough -- the global flags must be set."""
    t = FakeTorch(A100_80)
    _plan_hardware(args(), t, LARGE_V2)
    assert t.backends.cuda.matmul.allow_tf32 is True
    assert t.backends.cudnn.allow_tf32 is True


def test_bf16_on_turing_is_refused_loudly():
    with pytest.raises(SystemExit, match="no bf16"):
        _plan_hardware(args(precision="bf16"), FakeTorch(T4), SMALL)


@pytest.mark.parametrize(
    "cuda,params,want_ckpt",
    [
        (A100_40, LARGE_V2, True),    # 24.6/40 = 0.62 -> on
        (A100_80, LARGE_V2, False),   # 24.6/80 = 0.31 -> off, ~30-40% faster
        (A100_40, SMALL, False),      # 3.9/40  = 0.10 -> off
    ],
)
def test_gradient_checkpointing_follows_vram(cuda, params, want_ckpt):
    hw = _plan_hardware(args(), FakeTorch(cuda), params)
    assert hw["gradient_checkpointing"] is want_ckpt
    # Non-reentrant kwargs only make sense when checkpointing is actually on.
    assert ("gradient_checkpointing_kwargs" in hw) is want_ckpt


def test_checkpointing_can_be_forced_either_way():
    assert _plan_hardware(args(gradient_checkpointing="on"),
                          FakeTorch(A100_80), LARGE_V2)["gradient_checkpointing"]
    assert not _plan_hardware(args(gradient_checkpointing="off"),
                              FakeTorch(A100_40), LARGE_V2)["gradient_checkpointing"]


def test_state_estimate_matches_the_documented_numbers():
    """The README and configs quote ~24.7 GB for large-v2 and ~3.9 GB for small.
    If BYTES_PER_PARAM ever changes, those docs are wrong -- catch it here."""
    assert LARGE_V2 * BYTES_PER_PARAM / 1e9 == pytest.approx(24.6, abs=0.3)
    assert SMALL * BYTES_PER_PARAM / 1e9 == pytest.approx(3.9, abs=0.2)
    # large-v2 must land on the "checkpoint it" side of a 40 GB card.
    assert LARGE_V2 * BYTES_PER_PARAM / 1e9 > CHECKPOINT_VRAM_FRACTION * 40


# ---- config plumbing ------------------------------------------------------


def _parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--extra-hf", nargs="*", default=[])
    return ap


def test_config_supplies_defaults_but_cli_still_wins(tmp_path):
    """The precedence that matters: CLI > yaml > code default. If yaml won, a
    one-off override on the command line would be silently ignored."""
    p = tmp_path / "c.yaml"
    p.write_text("lr: 2.0e-5\nbatch_size: 16\n", encoding="utf-8")

    ap = _parser()
    argv = ["--config", str(p), "--batch-size", "8"]
    _apply_config_defaults(ap, argv)
    got = ap.parse_args(argv)

    assert got.lr == 2e-5      # from yaml
    assert got.batch_size == 8  # CLI overrode yaml


def test_config_rejects_unknown_keys(tmp_path):
    """A typo'd key used to be silently ignored -- the configs were decorative.
    Now it must fail, or the run does something other than the file says."""
    p = tmp_path / "c.yaml"
    p.write_text("lr: 3.0\nbatch_sizee: 16\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="batch_sizee"):
        _apply_config_defaults(_parser(), ["--config", str(p)])


@pytest.mark.parametrize("name", ["m6", "m7", "m8"])
def test_shipped_large_configs_parse_against_the_real_parser(name):
    """The whole point of --config is that these files are no longer decorative.
    Parse each one through the actual parser and check the values that matter."""
    path = CONFIGS / f"train_{name}_large.yaml"
    assert path.exists(), path

    ap = _build_parser()
    argv = ["--config", str(path)]
    _apply_config_defaults(ap, argv)   # raises SystemExit on an unknown key
    got = ap.parse_args(argv)

    assert got.model == "openai/whisper-large-v2"   # D2 retired on this path
    assert got.lr == 2e-5                          # paper's recipe (D3)
    assert got.batch_size * got.grad_accum == got.effective_batch == 64
    assert got.eval_batch_size < got.batch_size    # generate() OOMs otherwise
    assert got.language == "hi"                    # <|hi|> for all of M6/M7/M8
    assert got.out is not None                     # `out:` must actually apply
    assert got.precision == "auto" and got.gradient_checkpointing == "auto"


def test_m6_large_is_the_subset_run_and_m7_is_not():
    """M6 vs M7 differ by exactly one thing. If subset_ids went missing, M6 would
    silently become a second M7 and the headline contrast would vanish."""
    def load(name):
        ap = _build_parser()
        argv = ["--config", str(CONFIGS / f"train_{name}_large.yaml")]
        _apply_config_defaults(ap, argv)
        return ap.parse_args(argv)

    assert load("m6").subset_ids is not None
    assert load("m7").subset_ids is None
    assert load("m8").extra_hf == [
        "RohanRamesh/mucs-he-cs:cv_hi",
        "RohanRamesh/mucs-he-cs:cv_en",
    ]


def test_config_handles_lists_and_absent_config(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("extra_hf:\n  - a:b\n  - c:d\n", encoding="utf-8")
    ap = _parser()
    _apply_config_defaults(ap, ["--config", str(p)])
    assert ap.parse_args(["--config", str(p)]).extra_hf == ["a:b", "c:d"]

    ap2 = _parser()
    _apply_config_defaults(ap2, [])  # no --config: must be a no-op, not a crash
    assert ap2.parse_args([]).lr == 1.0
