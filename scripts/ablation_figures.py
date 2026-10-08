#!/usr/bin/env python3
"""One labelled figure per ablation axis and metric.

Reads results/ablation/{ablation.jsonl,text_stats.json} and the per-run dev curves in
csasr-work/abl/<run>/select/dev_curve.json; writes PNGs to results/ablation/figures/.

Axes:   A  ASR model            (zero-shot vs full fine-tune)
        B  fine-tuning method   (full / LoRA / QLoRA)
        C  text-generation LLM and sentence prompt
        D  training length      (dev curves: why the paper's runs got worse)
        S  summary of every run

Metrics: MER (error rate, lower is better), CBA-HE / CBA-EH (switch-point accuracy, higher is
better), % Latin script in the output (compare with the reference), peak VRAM, trainable %.

    ~/.venvs/csasr-abl/bin/python3 scripts/ablation_figures.py
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# ---- design tokens (validated categorical palette, light surface) -----------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e1"
ZERO = "#8c8b86"                                   # neutral: the zero-shot baseline
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
METHOD_COLOR = {"full": BLUE, "lora": ORANGE, "qlora": AQUA}
METHOD_LABEL = {"full": "Full fine-tune", "lora": "LoRA", "qlora": "QLoRA (4-bit)"}
ASR_ORDER = ["whisper-small", "whisper-medium", "whisper-large-v2", "mms-1b"]
ASR_LABEL = {"whisper-small": "Whisper small", "whisper-medium": "Whisper medium",
             "whisper-large-v2": "Whisper large-v2", "mms-1b": "MMS-1B"}
ASR_TICK = {k: v.replace("Whisper ", "Whisper\n") for k, v in ASR_LABEL.items()}
# fixed order, tied to the corpus (not to its rank)
CORPUS_ORDER = ["g26", "g31", "g31m", "s30"]
CORPUS_COLOR = {"g26": BLUE, "g31": ORANGE, "g31m": AQUA, "s30": VIOLET}
CORPUS_LABEL = {"g26": "Gemma-4 26B\n(paper prompt)",
                "g31": "Gemma-4 31B\n(paper prompt)",
                "g31m": "Gemma-4 31B\n(matched prompt)",
                "s30": "Sarvam-30B\n(paper prompt)"}
CORPUS_SHORT = {"g26": "Gemma-4 26B", "g31": "Gemma-4 31B",
                "g31m": "Gemma-4 31B, matched prompt", "s30": "Sarvam-30B"}
REF_LATIN = 24.3        # % Latin in the MUCS test references
PAPER_FIRST_EVAL = 250  # the replication runs evaluated first at this step

NOTE = ("MUCS test, 30 recordings. Error bars: 95% bootstrap CI over recordings. "
        "One seed per cell.")
NOTE_NO_CI = "MUCS test, 30 recordings. One seed per cell."


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 1,
        "axes.axisbelow": True, "legend.frameon": False, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlesize": 13, "figure.dpi": 150,
    })


# ---- data --------------------------------------------------------------------------
def load(results: Path, exp: Path):
    rows = [json.loads(l) for l in (results / "ablation.jsonl").read_text().splitlines() if l]
    stats = json.loads((results / "text_stats.json").read_text())
    return rows, stats


def pick(rows, **kw):
    for r in rows:
        if all(r.get(k, "") == v for k, v in kw.items()):
            return r
    return None


def zs(rows, asr):
    return pick(rows, asr=asr, ft="zeroshot")


def val(r, key, lo=None, hi=None):
    """(value, lower, upper) or None; CI bounds fall back to the value when absent."""
    if r is None or r.get(key) is None:
        return None
    v = r[key]
    return v, r.get(lo, v) if lo else v, r.get(hi, v) if hi else v


MER = ("mer", "mer_lo", "mer_hi")
CBA_HE = ("cba_he", "cba_he_lo", "cba_he_hi")
CBA_EH = ("cba_eh", None, None)


# ---- drawing helpers ---------------------------------------------------------------
def finish(fig, title, subtitle, out: Path, note=NOTE, bottom=0.0):
    w, h = fig.get_size_inches()
    lines = textwrap.wrap(subtitle, int(w * 12))
    head = 0.6 + 0.2 * len(lines)                   # inches reserved for title + subtitle
    fig.tight_layout(rect=(0, bottom, 1, 1 - head / h))
    fig.text(0.01, 1 - 0.12 / h, title, ha="left", va="top", fontsize=14, fontweight="bold",
             color=INK)
    fig.text(0.01, 1 - 0.42 / h, "\n".join(lines), ha="left", va="top", fontsize=10.5,
             color=INK2, linespacing=1.35)
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())      # inches
    fig.text(bb.x0 / w + 0.005, (bb.y0 - 0.14) / h, "\n".join(textwrap.wrap(note, int(w * 14))),
             ha="left", va="top", fontsize=8.5, color=INK2, transform=fig.transFigure)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"[figures] {out}")


def grouped_bars(ax, groups, series, data, fmt="{:.1f}", width=0.8, ylim_pad=1.18):
    """groups: x labels; series: [(label, color)]; data[g][s] = (v, lo, hi) | None."""
    n = len(series)
    w = width / n
    top = 0
    for s, (label, color) in enumerate(series):
        xs, vs, errs, labs = [], [], [[], []], []
        for g in range(len(groups)):
            d = data[g][s]
            if d is None:
                continue
            v, lo, hi = d
            xs.append(g - width / 2 + w * (s + 0.5))
            vs.append(v)
            errs[0].append(max(v - lo, 0))
            errs[1].append(max(hi - v, 0))
            labs.append((xs[-1], hi, v))
        ax.bar(xs, vs, w * 0.92, color=color, edgecolor=SURFACE, linewidth=1.5, label=label,
               zorder=3)
        if any(e > 0 for e in errs[0] + errs[1]):
            ax.errorbar(xs, vs, yerr=errs, fmt="none", ecolor=INK2, elinewidth=1,
                        capsize=2.5, zorder=4)
        for x, hi, v in labs:
            ax.annotate(fmt.format(v), (x, hi), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, color=INK, zorder=5,
                        bbox=dict(fc=SURFACE, ec="none", pad=0.6))
            top = max(top, hi)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.set_ylim(0, top * ylim_pad)
    ax.grid(False, axis="x")
    ax.tick_params(axis="x", length=0)


def legend_below(ax, ncol, y=-0.14):
    ax.legend(ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, y), fontsize=10)


def ref_line(ax, y, text, color=INK2, side="left"):
    ax.axhline(y, color=color, lw=1.2, ls=(0, (4, 3)), zorder=2)
    x, ha, dx = (0.0, "left", 4) if side == "left" else (1.0, "right", -2)
    ax.annotate(text, (x, y), xycoords=("axes fraction", "data"), xytext=(dx, 4),
                textcoords="offset points", ha=ha, va="bottom", fontsize=9, color=INK2,
                bbox=dict(fc=SURFACE, ec="none", pad=1), zorder=6)


# ---- axis A: ASR model -------------------------------------------------------------
def fig_a_mer(rows, out):
    asrs = [a for a in ASR_ORDER if pick(rows, corpus="g26", asr=a, ft="full", tag="")]
    data = [[val(zs(rows, a), *MER), val(pick(rows, corpus="g26", asr=a, ft="full", tag=""), *MER)]
            for a in asrs]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    grouped_bars(ax, [ASR_TICK[a] for a in asrs],
                 [("Zero-shot (no training)", ZERO), ("Full fine-tune", BLUE)], data)
    for g, a in enumerate(asrs):
        d = pick(rows, corpus="g26", asr=a, ft="full", tag="")
        ax.annotate(f"{d['d_mer']:+.1f}", (g, 3), ha="center", fontsize=9.5, color=INK,
                    fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=GRID))
    ax.set_ylabel("MER (%) — lower is better")
    legend_below(ax, 2)
    finish(fig, "A1 · ASR model: error rate",
           "Every model improves after fine-tuning on 22 h of synthetic speech. Boxed numbers "
           "are the change in MER points.", out)


def fig_a_cba(rows, out):
    asrs = [a for a in ASR_ORDER if pick(rows, corpus="g26", asr=a, ft="full", tag="")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, (key, name) in zip(axes, [(CBA_HE, "Hindi → English switches (CBA-HE)"),
                                      (CBA_EH, "English → Hindi switches (CBA-EH)")]):
        data = [[val(zs(rows, a), *key),
                 val(pick(rows, corpus="g26", asr=a, ft="full", tag=""), *key)] for a in asrs]
        grouped_bars(ax, [ASR_TICK[a] for a in asrs],
                     [("Zero-shot (no training)", ZERO), ("Full fine-tune", BLUE)], data)
        ax.set_title(name, fontsize=11.5)
    axes[0].set_ylabel("Switch-point accuracy (%) — higher is better")
    legend_below(axes[0], 2, y=-0.14)
    finish(fig, "A2 · ASR model: switch-point accuracy",
           "MMS-1B barely recovers switch points (under 6%); the three Whisper models gain 7–49 points.", out)


# ---- axis B: fine-tuning method ----------------------------------------------------
def _grid_data(rows, key, asrs, with_zero=True):
    series = ([("Zero-shot", ZERO)] if with_zero else []) + [
        (METHOD_LABEL[f], METHOD_COLOR[f]) for f in ("full", "lora", "qlora")]
    data = []
    for a in asrs:
        row = [val(zs(rows, a), *key)] if with_zero else []
        row += [val(pick(rows, corpus="g26", asr=a, ft=f, tag=""), *key)
                for f in ("full", "lora", "qlora")]
        data.append(row)
    return series, data


def fig_b_metric(rows, out, key, ylabel, title, subtitle, ref=None, fmt="{:.1f}"):
    asrs = [a for a in ASR_ORDER if any(pick(rows, corpus="g26", asr=a, ft=f, tag="")
                                        for f in ("full", "lora", "qlora"))]
    series, data = _grid_data(rows, key, asrs)
    fig, ax = plt.subplots(figsize=(11, 4.9))
    grouped_bars(ax, [ASR_TICK[a] for a in asrs], series, data, fmt=fmt)
    ax.set_ylabel(ylabel)
    if ref:
        ref_line(ax, *ref)
    legend_below(ax, 4)
    note = (NOTE if key[1] else NOTE_NO_CI) + "  MMS-1B has no QLoRA run (no 4-bit CTC path)."
    finish(fig, title, subtitle, out, note=note)


def fig_b_delta(rows, out):
    items = []
    for a in ASR_ORDER:
        for f in ("full", "lora", "qlora"):
            r = pick(rows, corpus="g26", asr=a, ft=f, tag="")
            if r and "d_mer" in r:
                items.append((a, f, r))
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ys, y = [], 0
    labels = []
    last = None
    for a, f, r in items:
        if last is not None and a != last:
            y += 0.6
        last = a
        ys.append(y)
        ax.errorbar(r["d_mer"], y, xerr=[[r["d_mer"] - r["d_mer_lo"]], [r["d_mer_hi"] - r["d_mer"]]],
                    fmt="o", color=METHOD_COLOR[f], ecolor=METHOD_COLOR[f], elinewidth=2,
                    capsize=3, ms=8, mec=SURFACE, mew=2, zorder=3)
        ax.annotate(f"{r['d_mer']:+.1f}", (r["d_mer_hi"], y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
        labels.append(f"{ASR_LABEL[a]} · {METHOD_LABEL[f]}")
        y += 1
    ax.axvline(0, color=INK2, lw=1.2, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Change in MER vs the same model zero-shot (points) — left of 0 is better")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    for f in ("full", "lora", "qlora"):
        ax.scatter([], [], color=METHOD_COLOR[f], label=METHOD_LABEL[f], s=50)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.1))
    ax.set_xlim(None, 4)
    finish(fig, "B4 · Fine-tuning method: paired improvement over zero-shot",
           "An interval that excludes 0 is a significant gain; all do except MMS-1B LoRA.",
           out,
           note="Paired bootstrap against the same model's zero-shot output, 95% CI.")


def fig_b_memory(rows, out):
    asrs = ["whisper-medium", "whisper-large-v2"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    series = [(METHOD_LABEL[f], METHOD_COLOR[f]) for f in ("full", "lora", "qlora")]
    for ax, key, ylabel, title, fmt in [
            (axes[0], "peak_vram_gb", "Peak GPU memory (GB) — lower is better",
             "Memory to train", "{:.1f}"),
            (axes[1], "trainable_pct", "Trainable parameters (%, log scale)",
             "Parameters updated", "{:.2f}")]:
        data = [[val(pick(rows, corpus="g26", asr=a, ft=f, tag=""), key) for f in
                 ("full", "lora", "qlora")] for a in asrs]
        grouped_bars(ax, [ASR_TICK[a] for a in asrs], series, [[d for d in g] for g in data],
                     fmt=fmt)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11.5)
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.5, 300)
    legend_below(axes[0], 3, y=-0.17)
    finish(fig, "B5 · Fine-tuning method: cost",
           "QLoRA matches full fine-tuning at a third to under half of the memory, updating about "
           "1% of the parameters.", out,
           note="A100-80GB, effective batch 64, bf16, gradient checkpointing on. Small and MMS "
                "runs used different settings and are not comparable, so they are left out.")


# ---- axis C: LLM and prompt --------------------------------------------------------
def _llm_rows(rows):
    out = []
    for c in CORPUS_ORDER:
        r = pick(rows, corpus=c, asr="whisper-large-v2", ft="full", tag="")
        if r:
            out.append((c, r))
    return out


def fig_c_bars(rows, out, keys, ylabel_title, title, subtitle, ref=None):
    lr = _llm_rows(rows)
    z = zs(rows, "whisper-large-v2")
    names = ["Zero-shot\n(no training)"] + [CORPUS_LABEL[c] for c, _ in lr]
    colors = [ZERO] + [CORPUS_COLOR[c] for c, _ in lr]
    labels = ["Zero-shot"] + [CORPUS_SHORT[c] for c, _ in lr]
    npan = len(keys)
    fig, axes = plt.subplots(1, npan, figsize=(6.4 * npan + 1.5, 5.2), sharey=(npan > 1),
                             squeeze=False)
    for ax, (key, pname) in zip(axes[0], keys):
        for i, (r, col) in enumerate(zip([z] + [r for _, r in lr], colors)):
            d = val(r, *key)
            if d is None:
                continue
            v, lo, hi = d
            ax.bar(i, v, 0.7, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                   label=labels[i])
            if hi > lo:
                ax.errorbar(i, v, yerr=[[max(v - lo, 0)], [max(hi - v, 0)]], fmt="none",
                            ecolor=INK2, elinewidth=1, capsize=3, zorder=4)
            ax.annotate(f"{v:.1f}", (i, hi), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9.5, color=INK)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.tick_params(axis="x", length=0)
        ax.grid(False, axis="x")
        ax.set_xlim(-0.6, len(names) - 0.4)
        if npan > 1:
            ax.set_title(pname, fontsize=11.5)
        if ref:
            ref_line(ax, *ref)
    top = max(ax.get_ylim()[1] for ax in axes[0])
    for ax in axes[0]:
        ax.set_ylim(0, top * 1.08)
    axes[0][0].set_ylabel(ylabel_title)
    finish(fig, title, subtitle, out,
           note=(NOTE if any(k[1] for k, _ in keys) else NOTE_NO_CI)
           + "  Whisper large-v2, full fine-tune, 22 h of audio per corpus.")


# ---- text statistics ---------------------------------------------------------------
def fig_c_text(stats, out):
    order = [("Gemma-4-26B-A4B", "g26"), ("Gemma-4-31B", "g31"),
             ("Gemma-4-31B (matched prompt)", "g31m"), ("Sarvam-30B", "s30")]
    by = {t["label"]: t for t in stats}
    target = by.get("MUCS test (target)")
    items = [(by[l], c) for l, c in order if l in by]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    panels = [("latin_pct_hi_matrix", "English-script share inside Hindi sentences (%)",
               "Script mix of the training text", target and target["latin_pct_hi_matrix"],
               "real MUCS test: {:.1f}%"),
              ("test_switch_coverage_pct", "Test switch-points that appear in the text (%)",
               "Coverage of the test switch-points", None, "")]
    for ax, (key, ylabel, title, ref, reftext) in zip(axes, panels):
        for i, (t, c) in enumerate(items):
            ax.bar(i, t[key], 0.7, color=CORPUS_COLOR[c], edgecolor=SURFACE, linewidth=1.5,
                   zorder=3)
            ax.annotate(f"{t[key]:.1f}", (i, t[key]), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9.5)
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels([CORPUS_LABEL[c] for _, c in items], fontsize=8.5)
        ax.tick_params(axis="x", length=0)
        ax.grid(False, axis="x")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11.5)
        if ref:
            ref_line(ax, ref, reftext.format(ref))
        ax.set_ylim(0, max(t[key] for t, _ in items) * 1.2)
    finish(fig, "C4 · Text-generation LLM: what the synthetic text looks like",
           "Paper-prompt Gemma corpora under-use English inside Hindi sentences (9% vs 20% in "
           "real speech); the matched prompt and Sarvam-30B do not.", out,
           note="Statistics of the generated text before TTS (sentence-level, after filtering).")


# ---- axis D: training length -------------------------------------------------------
def dev_curve(exp, run):
    p = exp / run / "select" / "dev_curve.json"
    return json.loads(p.read_text()) if p.exists() else None


def points(d):
    """Curve points, dropping a final off-by-one checkpoint (1201 after 1200)."""
    out = []
    for c in d["curve"]:
        if out and c["step"] - out[-1]["step"] <= 1:
            continue
        out.append(c)
    return out


def _ordinal(curves):
    steps = sorted({c["step"] for d in curves for c in points(d)})
    return {s: i for i, s in enumerate(steps)}, steps


def _shade_paper(ax, pos, steps):
    """Shade the checkpoints at or after the paper's first evaluation step."""
    lower = max((i for s, i in pos.items() if s < PAPER_FIRST_EVAL), default=0)
    ax.axvspan(lower + (PAPER_FIRST_EVAL - steps[lower]) / (steps[lower + 1] - steps[lower]),
               len(steps) - 0.5, color=GRID, alpha=0.7, zorder=1, lw=0)


def fig_d_asr(rows, exp, out):
    asrs = [a for a in ASR_ORDER
            if any(dev_curve(exp, f"g26__{a}__{f}") for f in ("full", "lora", "qlora"))]
    fig, axes = plt.subplots(1, len(asrs), figsize=(4.4 * len(asrs), 4.6), sharey=False)
    for ax, a in zip(np_iter(axes), asrs):
        cs = {f: dev_curve(exp, f"g26__{a}__{f}") for f in ("full", "lora", "qlora")}
        cs = {f: d for f, d in cs.items() if d}
        pos, steps = _ordinal(cs.values())
        _shade_paper(ax, pos, steps)
        for f, d in cs.items():
            xs = [pos[c["step"]] for c in points(d)]
            ys = [c["mer"] for c in points(d)]
            ax.plot(xs, ys, color=METHOD_COLOR[f], lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5,
                    label=METHOD_LABEL[f], zorder=3)
            sel = next(c for c in d["curve"] if c["step"] == d["selected_step"])
            ax.scatter([pos[sel["step"]]], [sel["mer"]], s=120, facecolor="none", edgecolor=INK,
                       linewidth=1.3, zorder=4)
        z0 = next(iter(cs.values()))["curve"][0]["mer"]
        ax.axhline(z0, color=ZERO, lw=1.3, ls=(0, (4, 3)), zorder=2)
        ax.annotate("zero-shot", (len(steps) - 0.6, z0), xytext=(0, 4), textcoords="offset points",
                    ha="right", fontsize=8.5, color=INK2)
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels(steps, fontsize=8, rotation=45)
        ax.set_title(ASR_LABEL[a], fontsize=11.5)
        ax.set_xlabel("training step (checkpoints, not to scale)")
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
    np_iter(axes)[0].set_ylabel("Dev MER (%) — lower is better")
    h, l = np_iter(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.0), fontsize=10)
    finish(fig, "D1 · Training length: error rate on dev audio, by model and method",
           "Gains arrive in the first 50–100 steps, then the error rate climbs back; the two "
           "larger Whisper models end above their zero-shot level. Circle: selected checkpoint. "
           "Grey band: where the paper's runs were evaluated (step ≥ 250).", out,
           note="Dev set: 4 h of real MUCS audio, recording-level decoder. Dashed line: "
                "zero-shot. Learning-rate warm-up covers the first steps.", bottom=0.1)


def np_iter(axes):
    return list(axes.flat) if hasattr(axes, "flat") else [axes]


def fig_d_llm(rows, exp, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    curves = {c: dev_curve(exp, f"{c}__whisper-large-v2__full") for c in CORPUS_ORDER}
    curves = {c: d for c, d in curves.items() if d}
    pos, steps = _ordinal(curves.values())
    for ax, key, ylabel, title in [
            (axes[0], "mer", "Dev MER (%) — lower is better", "Error rate"),
            (axes[1], "latin_pct", "English-script share of the output (%)",
             "Script mix of the output")]:
        _shade_paper(ax, pos, steps)
        for c, d in curves.items():
            ax.plot([pos[p["step"]] for p in points(d)], [p[key] for p in points(d)],
                    color=CORPUS_COLOR[c], lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5,
                    label=CORPUS_SHORT[c], zorder=3)
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels(steps, fontsize=8.5)
        ax.set_xlabel("training step (checkpoints, not to scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11.5)
        ax.grid(False, axis="x")
    ref_line(axes[1], REF_LATIN, f"real MUCS test: {REF_LATIN}%")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.0), fontsize=10)
    finish(fig, "D2 · Training length by text corpus (Whisper large-v2, full fine-tune)",
           "The matched-prompt and Sarvam corpora recover after a mid-run bump and end near 42% "
           "dev MER, below zero-shot (53%); the paper-prompt corpora end near 58–60%, above it. "
           "Grey band: step ≥ 250.",
           out, note="Dev set: 4 h of real MUCS audio. Step 0 is zero-shot.", bottom=0.1)


# ---- summary -----------------------------------------------------------------------
def fig_summary(rows, out):
    items = [r for r in rows if r.get("mer") is not None and not r.get("tag")]
    items.sort(key=lambda r: r["mer"], reverse=True)
    fig, ax = plt.subplots(figsize=(9.5, 0.36 * len(items) + 2))
    for i, r in enumerate(items):
        col = ZERO if r["ft"] == "zeroshot" else METHOD_COLOR[r["ft"]]
        hatch = "////" if r["ft"] != "zeroshot" and r["corpus"] != "g26" else None
        name = (f"{ASR_LABEL[r['asr']]} · zero-shot" if r["ft"] == "zeroshot" else
                f"{ASR_LABEL[r['asr']]} · {METHOD_LABEL[r['ft']]}" +
                ("" if r["corpus"] == "g26" else f" · {CORPUS_SHORT[r['corpus']]}"))
        ax.barh(i, r["mer"], 0.66, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                hatch=hatch)
        ax.errorbar(r["mer"], i, xerr=[[r["mer"] - r["mer_lo"]], [r["mer_hi"] - r["mer"]]],
                    fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5, zorder=4)
        ax.annotate(f"{r['mer']:.1f}", (r["mer_hi"], i), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK)
        ax.text(-0.4, i, name, ha="right", va="center", fontsize=8.5, color=INK)
    ax.set_yticks([])
    ax.set_ylim(len(items) - 0.4, -0.7)
    ax.set_xlim(0, max(r["mer_hi"] for r in items) * 1.1)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_xlabel("MER (%) — lower is better")
    handles = [Patch(fc=ZERO, label="Zero-shot")] + [
        Patch(fc=METHOD_COLOR[f], label=METHOD_LABEL[f]) for f in ("full", "lora", "qlora")] + [
        Patch(fc=BLUE, hatch="////", ec=SURFACE, label="Hatched: other text corpus (C)")]
    ax.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.3, -0.08),
              fontsize=9)
    finish(fig, "S · Every run, ranked by error rate",
           "Best: Sarvam-30B and the matched-prompt corpus (~32), then Whisper medium + QLoRA "
           "(40.7). Worst: Whisper small zero-shot (67.2).", out)


# ---- main --------------------------------------------------------------------------
def main():
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=here / "results" / "ablation")
    ap.add_argument("--exp", type=Path, default=here / "csasr-work" / "abl")
    ap.add_argument("--out", type=Path, default=None, help="default: <results-dir>/figures")
    a = ap.parse_args()
    out = a.out or a.results_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    style()
    rows, stats = load(a.results_dir, a.exp)

    fig_a_mer(rows, out / "A1_asr_model_mer.png")
    fig_a_cba(rows, out / "A2_asr_model_switch_accuracy.png")
    fig_b_metric(rows, out / "B1_ft_method_mer.png", MER, "MER (%) — lower is better",
                 "B1 · Fine-tuning method: error rate",
                 "Full, LoRA and QLoRA are within each other's intervals at medium and large-v2; "
                 "full is clearly better only at small.")
    fig_b_metric(rows, out / "B2_ft_method_switch_accuracy.png", CBA_HE,
                 "Switch accuracy, Hindi → English (%)",
                 "B2 · Fine-tuning method: switch-point accuracy",
                 "CBA-HE, higher is better. Whisper medium + QLoRA reaches 68.5%, above the paper's M7 (52.3%).")
    fig_b_metric(rows, out / "B3_ft_method_english_share.png", ("latin_pct", None, None),
                 "English-script share of the output (%)",
                 "B3 · Fine-tuning method: how much English the model writes",
                 "Fine-tuning raises every model toward the real-speech share (24.3%). A share "
                 "near the reference is necessary, not sufficient: MMS-1B overshoots yet gets few "
                 "switches right.", ref=(REF_LATIN, f"real MUCS test: {REF_LATIN}%"))
    fig_b_delta(rows, out / "B4_ft_method_improvement_vs_zeroshot.png")
    fig_b_memory(rows, out / "B5_ft_method_cost.png")
    fig_c_bars(rows, out / "C1_llm_prompt_mer.png", [(MER, "")],
               "MER (%) — lower is better", "C1 · Text LLM and prompt: error rate",
               "Switching from Gemma 26B to 31B changes little; fixing the script mix "
               "(matched prompt, Sarvam-30B) cuts MER by 13 points.")
    fig_c_bars(rows, out / "C2_llm_prompt_switch_accuracy.png",
               [(CBA_HE, "Hindi → English (CBA-HE)"), (CBA_EH, "English → Hindi (CBA-EH)")],
               "Switch-point accuracy (%) — higher is better",
               "C2 · Text LLM and prompt: switch-point accuracy",
               "The two corpora with a realistic script mix reach ~79% on Hindi → English "
               "switches; the others stay at 46–51%.")
    fig_c_bars(rows, out / "C3_llm_prompt_english_share.png", [(("latin_pct", None, None), "")],
               "English-script share of the output (%)",
               "C3 · Text LLM and prompt: how much English the model writes",
               "Models trained on the matched and Sarvam text write English at, or above, the "
               "real rate.", ref=(REF_LATIN, f"real MUCS test: {REF_LATIN}%"))
    fig_c_text(stats, out / "C4_llm_text_statistics.png")
    fig_d_asr(rows, a.exp, out / "D1_training_length_by_model.png")
    fig_d_llm(rows, a.exp, out / "D2_training_length_by_corpus.png")
    fig_summary(rows, out / "S_all_runs_ranked.png")


if __name__ == "__main__":
    main()
