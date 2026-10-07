"""Tables and figures for the ablation study, from whatever runs have finished.

    python scripts/ablation_report.py --results-dir results/ablation --exp csasr-work/abl

Reads the one-row-per-run files `csasr.eval.select` writes (`rows/*.json`) and
the test hypotheses beside them (`hyps/*.jsonl`). CPU only. Safe to run on a
partial study: missing cells are left blank, never guessed.

Every MER is reported with a 95% bootstrap interval over the 30 test
recordings, and every difference against the matching zero-shot model is a
PAIRED bootstrap (the same resampled recordings on both sides). With 30
recordings a 1-2 point MER gap is frequently inside the noise; the `sig` column
says whether the interval of the difference excludes zero.

Outputs, all under --results-dir:
    report.md  ablation.csv  ablation.jsonl
    fig_grid_mer.png  fig_grid_cba.png  fig_efficiency.png  fig_dev_curves.png
    fig_llm_axis.png  fig_text_latin.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from csasr.eval.cba import cba_grouped
from csasr.eval.grouping import concat_refs, recording_id
from csasr.eval.mer import to_units
from csasr.manifest import read_jsonl
from csasr.normalize import normalize

# ---- presentation -----------------------------------------------------------
# Validated with the dataviz palette checker (light surface, all pairs): the
# three method hues clear the CVD and normal-vision floors. Aqua is under 3:1
# against the surface, so every bar carries a value label and the same numbers
# are in report.md.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e1"
METHOD_COLOR = {"full": "#2a78d6", "lora": "#eb6834", "qlora": "#1baf7a"}
METHOD_LABEL = {"full": "Full fine-tune", "lora": "LoRA", "qlora": "QLoRA (4-bit)",
                "zeroshot": "Zero-shot"}
REF_GRAY = "#8a8984"
ASR_ORDER = ["whisper-small", "whisper-medium", "whisper-large-v2", "mms-1b"]
ASR_LABEL = {"whisper-small": "Whisper small", "whisper-medium": "Whisper medium",
             "whisper-large-v2": "Whisper large-v2", "mms-1b": "MMS-1B"}
FT_ORDER = ["full", "lora", "qlora"]
CORPUS_LABEL = {"g26": "Gemma-4\n26B-A4B", "g31": "Gemma-4\n31B", "s30": "Sarvam\n30B",
                "s105": "Sarvam\n105B"}


def corpus_label(c: str, *, one_line: bool = False) -> str:
    if c.endswith("m") and c[:-1] in CORPUS_LABEL:
        s = CORPUS_LABEL[c[:-1]] + "\nmatched prompt"
    else:
        s = CORPUS_LABEL.get(c, c)
    return s.replace("\n", " ") if one_line else s


# ---- statistics --------------------------------------------------------------
def per_recording(refs_utt: list[dict], hyps: dict[str, str]) -> dict[str, np.ndarray]:
    """Error and reference counts per recording, so any resample is a sum."""
    import jiwer

    refs = concat_refs(refs_utt)
    by_reco: dict[str, list[tuple[str, str]]] = {}
    for r in refs_utt:
        by_reco.setdefault(recording_id(r["utt_id"]), []).append((r["utt_id"], r["text"]))
    ids = sorted(refs)
    out = {k: np.zeros(len(ids)) for k in ("err", "n", "he_m", "he_t", "eh_m", "eh_t")}
    for i, reco in enumerate(ids):
        ref = " ".join(to_units(normalize(refs[reco], "scoring")))
        hyp = " ".join(to_units(normalize(hyps.get(reco, ""), "scoring")))
        if ref.strip():
            m = jiwer.process_words(ref, hyp)
            out["err"][i] = m.substitutions + m.deletions + m.insertions
            out["n"][i] = m.hits + m.substitutions + m.deletions
        c = cba_grouped(by_reco[reco], {reco: hyps.get(reco, "")}, recording_id,
                        preset="scoring", mode="lenient")
        out["he_m"][i], out["he_t"][i] = c.he_matched, c.he_total
        out["eh_m"][i], out["eh_t"][i] = c.eh_matched, c.eh_total
    return out


def bootstrap(counts: dict, base: dict | None, n_boot: int, seed: int = 0) -> dict:
    """95% CIs for MER and CBA-HE; paired difference vs `base` if given."""
    rng = np.random.default_rng(seed)
    n = len(counts["err"])
    idx = rng.integers(0, n, size=(n_boot, n))

    def ratio(c, num, den):
        return 100.0 * c[num][idx].sum(1) / np.maximum(c[den][idx].sum(1), 1)

    mer = ratio(counts, "err", "n")
    he = ratio(counts, "he_m", "he_t")
    lo, hi = np.percentile(mer, [2.5, 97.5])
    out = {"mer_lo": round(lo, 2), "mer_hi": round(hi, 2),
           "cba_he_lo": round(np.percentile(he, 2.5), 2),
           "cba_he_hi": round(np.percentile(he, 97.5), 2)}
    if base is not None:
        d = mer - ratio(base, "err", "n")
        dlo, dhi = np.percentile(d, [2.5, 97.5])
        point = (100.0 * counts["err"].sum() / counts["n"].sum()
                 - 100.0 * base["err"].sum() / base["n"].sum())
        out.update(d_mer=round(point, 2), d_mer_lo=round(dlo, 2), d_mer_hi=round(dhi, 2),
                   sig=bool(dlo > 0 or dhi < 0))
    return out


# ---- loading -----------------------------------------------------------------
def load(results_dir: Path, n_boot: int) -> list[dict]:
    rows = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((results_dir / "rows").glob("*.json"))]
    refs_path = results_dir / "refs_test.jsonl"
    if not rows or not refs_path.exists():
        return rows
    refs = list(read_jsonl(refs_path))
    counts = {}
    for r in rows:
        hp = results_dir / "hyps" / f"{r['run']}.jsonl"
        if hp.exists():
            counts[r["run"]] = per_recording(
                refs, {h["utt_id"]: h.get("hyp", "") for h in read_jsonl(hp)})
    for r in rows:
        if r["run"] not in counts:
            continue
        base = None if r.get("ft") == "zeroshot" else counts.get(f"zeroshot__{r.get('asr')}")
        r.update(bootstrap(counts[r["run"]], base, n_boot))
    return rows


def pick(rows, **kw):
    for r in rows:
        if all(r.get(k, "") == v for k, v in kw.items()):
            return r
    return None


# ---- tables ------------------------------------------------------------------
def _f(v, nd=1, dash="–"):
    return dash if v is None or v == "" else f"{v:.{nd}f}"


def _mer(r):
    return "–" if r is None else (
        f"{r['mer']:.1f} [{r['mer_lo']:.1f}, {r['mer_hi']:.1f}]" if "mer_lo" in r
        else f"{r['mer']:.1f}")


def _delta(r):
    if r is None or "d_mer" not in r:
        return "–"
    return f"{r['d_mer']:+.1f} [{r['d_mer_lo']:+.1f}, {r['d_mer_hi']:+.1f}]" \
           + (" *" if r["sig"] else "")


def _row(label, r):
    if r is None:
        return f"| {label} | – | – | – | – | – | – | – | – | – |"
    t = r.get("train_runtime_s")
    vram, hours = _f(r.get("peak_vram_gb")), _f(t / 3600 if t else None, 2)
    if r.get("resumed"):
        # A resumed run only timed (and only peaked) over its final leg.
        vram = hours = "n/a †"
    return (f"| {label} | {_mer(r)} | {_delta(r)} | {_f(r.get('cba_he'))} | "
            f"{_f(r.get('cba_eh'))} | {_f(r.get('latin_pct'))} | "
            f"{_f(r.get('trainable_pct'), 2)} | {vram} | "
            f"{hours} | {r.get('selected_step', '–')} |")


HEAD = ("| System | MER ↓ [95% CI] | Δ vs zero-shot | CBA-HE ↑ | CBA-EH ↑ | % Latin | "
        "trainable % | peak VRAM GB | train h | step |\n"
        "|---|---|---|---|---|---|---|---|---|---|")


def write_report(rows, text_stats, out: Path) -> None:
    L = ["# Ablation study — Hindi-English code-switched ASR", "",
         "Test set: MUCS, 30 recordings. MER is the hybrid definition (characters on "
         "Devanagari, words on Latin); CBA is the lenient rule that reproduces the paper's "
         "zero-shot row. Reference % Latin is 24.3. `Δ` is a paired bootstrap against the "
         "same model zero-shot; `*` marks an interval that excludes zero. Checkpoints were "
         "selected on dev with the test decoder.", ""]

    L += ["## 1. ASR model × fine-tuning method", "",
          "Training text: existing Gemma-4-26B-A4B corpus, cut to 22 h.", "", HEAD]
    for asr in ASR_ORDER:
        L.append(_row(f"{ASR_LABEL[asr]} · zero-shot", pick(rows, asr=asr, ft="zeroshot")))
        for ft in FT_ORDER:
            r = pick(rows, corpus="g26", asr=asr, ft=ft, tag="")
            if r or asr != "mms-1b" or ft != "qlora":
                L.append(_row(f"{ASR_LABEL[asr]} · {METHOD_LABEL[ft]}", r))
    extra = [r for r in rows if r.get("tag")]
    if extra:
        L += ["", "Checks (learning rate, seeds):", "", HEAD]
        L += [_row(f"{ASR_LABEL.get(r['asr'], r['asr'])} · {METHOD_LABEL.get(r['ft'], r['ft'])}"
                   f" · {r['tag']}", r) for r in extra]

    L += ["", "## 2. Text-generation LLM and prompt", "",
          "Whisper large-v2, full fine-tune, 22 h of audio per corpus, paper-verbatim "
          "prompts unless marked.", "", HEAD,
          _row("Zero-shot", pick(rows, asr="whisper-large-v2", ft="zeroshot"))]
    corpora = ["g26"] + sorted({r["corpus"] for r in rows
                                if r.get("corpus") not in (None, "none", "g26")})
    for c in corpora:
        L.append(_row(corpus_label(c, one_line=True),
                      pick(rows, corpus=c, asr="whisper-large-v2", ft="full", tag="")))

    if any(r.get("resumed") for r in rows):
        L += ["", "† Training was resumed from a saved checkpoint, so the logged time and peak "
              "VRAM cover only the final leg and are not reported. Same data volume and "
              "settings as the Gemma-4-31B run above."]

    if text_stats:
        L += ["", "## 3. The synthetic text itself", "",
              "| Corpus | sentences | % Latin | % Hindi-matrix | % Latin in Hindi-matrix | "
              "switches / sentence | mean English run | test switch coverage % | "
              "valid bigrams | LLM calls |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for t in text_stats:
            L.append(f"| {t['label']} | {t['sentences']:,} | {t['latin_pct']} | "
                     f"{t['hi_matrix_pct']} | {t['latin_pct_hi_matrix']} | "
                     f"{t['switches_per_sentence']} | {t['mean_en_run']} | "
                     f"{t['test_switch_coverage_pct']} | {t.get('valid_bigrams', '–')} | "
                     f"{t.get('bigram_calls', '–')} |")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---- figures -----------------------------------------------------------------
def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 1,
        "axes.axisbelow": True, "legend.frameon": False, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlesize": 12, "figure.dpi": 150,
    })
    return plt


def _bar_label(ax, x, v, err_hi=None):
    ax.annotate(f"{v:.1f}", (x, err_hi if err_hi is not None else v), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=9, color=INK,
                zorder=4, bbox=dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none"))


def fig_grid(plt, rows, key, lo, hi, title, ylabel, out):
    asrs = [a for a in ASR_ORDER if any(r.get("asr") == a and r.get("corpus") == "g26"
                                        for r in rows)]
    if not asrs:
        return
    fig, ax = plt.subplots(figsize=(1.6 + 2.3 * len(asrs), 4.2))
    w = 0.2
    for i, asr in enumerate(asrs):
        for j, ft in enumerate(FT_ORDER):
            r = pick(rows, corpus="g26", asr=asr, ft=ft, tag="")
            if not r:
                continue
            x = i + (j - 1) * (w + 0.02)
            ax.bar(x, r[key], w, color=METHOD_COLOR[ft], label=METHOD_LABEL[ft] if i == 0 else None)
            top = r[key]
            if lo in r:
                ax.plot([x, x], [r[lo], r[hi]], color=INK2, lw=1)
                top = r[hi]
            _bar_label(ax, x, r[key], top)
        z = pick(rows, asr=asr, ft="zeroshot")
        if z:
            ax.plot([i - 0.38, i + 0.38], [z[key]] * 2, color=REF_GRAY, lw=2,
                    label="Zero-shot" if i == 0 else None)
            ax.annotate(f"{z[key]:.1f}", (i + 0.39, z[key]), xytext=(2, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.set_xticks(range(len(asrs)), [ASR_LABEL[a] for a in asrs])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    h, lab = ax.get_legend_handles_labels()
    seen = dict(zip(lab, h))
    ax.legend(seen.values(), seen.keys(), ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.1), fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_efficiency(plt, rows, out):
    pts = [r for r in rows if r.get("corpus") == "g26" and r.get("ft") in FT_ORDER
           and not r.get("tag") and r.get("params_trainable") and r.get("asr") in ASR_LABEL]
    if not pts:
        return
    marker = {"whisper-small": "o", "whisper-medium": "s", "whisper-large-v2": "D", "mms-1b": "^"}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for r in pts:
        ax.scatter(r["params_trainable"] / 1e6, r["mer"], s=70, marker=marker[r["asr"]],
                   color=METHOD_COLOR[r["ft"]], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(ASR_LABEL[r["asr"]].replace("Whisper ", ""),
                    (r["params_trainable"] / 1e6, r["mer"]), xytext=(7, 4),
                    textcoords="offset points", fontsize=9, color=INK2)
    for ft in FT_ORDER:
        ax.scatter([], [], s=70, color=METHOD_COLOR[ft], label=METHOD_LABEL[ft])
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (millions, log scale)")
    ax.set_ylabel("Test MER (%), lower is better")
    ax.set_title("Accuracy against how much of the model is updated")
    ax.grid(True, axis="both")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_dev_curves(plt, rows, exp: Path, out):
    asrs = [a for a in ASR_ORDER if any(
        (exp / f"g26__{a}__{ft}" / "select" / "dev_curve.json").exists() for ft in FT_ORDER)]
    if not asrs:
        return
    fig, axes = plt.subplots(1, len(asrs), figsize=(4.1 * len(asrs), 3.8), squeeze=False)
    for ax, asr in zip(axes[0], asrs):
        for ft in FT_ORDER:
            p = exp / f"g26__{asr}__{ft}" / "select" / "dev_curve.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            xs = [c["step"] for c in d["curve"]]
            ys = [c["mer"] for c in d["curve"]]
            ax.plot(xs, ys, color=METHOD_COLOR[ft], lw=2, marker="o", ms=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=METHOD_LABEL[ft])
            sel = next(c for c in d["curve"] if c["step"] == d["selected_step"])
            ax.scatter([sel["step"]], [sel["mer"]], s=110, facecolor="none",
                       edgecolor=INK, linewidth=1.2, zorder=4)
        ax.set_title(ASR_LABEL[asr])
        ax.set_xlabel("Training step (0 = pretrained)")
    axes[0][0].set_ylabel("Dev MER (%), recording-level")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Dev MER by checkpoint (ring = selected)", x=0.02, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_llm(plt, rows, out):
    corpora = [c for c in ["g26", "g31", "s30", "s105"] + sorted(
        {r["corpus"] for r in rows if str(r.get("corpus", "")).endswith("m")})
        if pick(rows, corpus=c, asr="whisper-large-v2", ft="full", tag="")]
    if not corpora:
        return
    z = pick(rows, asr="whisper-large-v2", ft="zeroshot")
    panels = [("mer", "mer_lo", "mer_hi", "Test MER (%) ↓"),
              ("cba_he", "cba_he_lo", "cba_he_hi", "CBA-HE (%) ↑"),
              ("latin_pct", None, None, "Latin-script words in output (%)")]
    fig, axes = plt.subplots(1, 3, figsize=(max(9.5, 1.5 * len(corpora) * 3), 4.0))
    for ax, (key, lo, hi, title) in zip(axes, panels):
        for i, c in enumerate(corpora):
            r = pick(rows, corpus=c, asr="whisper-large-v2", ft="full", tag="")
            ax.bar(i, r[key], 0.4, color=METHOD_COLOR["full"])
            top = r[key]
            if lo and lo in r:
                ax.plot([i, i], [r[lo], r[hi]], color=INK2, lw=1)
                top = r[hi]
            _bar_label(ax, i, r[key], top)
        ref = 24.3 if key == "latin_pct" else None
        # x in axes fraction: a data-coordinate anchor past the last bar falls
        # outside the x-limits and matplotlib silently drops the label.
        tr = ax.get_yaxis_transform()
        if z:
            ax.axhline(z[key], color=REF_GRAY, lw=2)
            ax.text(1.0, z[key], f"zero-shot {z[key]:.1f} ", transform=tr, ha="right",
                    va="bottom", fontsize=9, color=INK2)
        if ref:
            ax.axhline(ref, color=INK, lw=1)
            ax.text(1.0, ref, f"reference {ref} ", transform=tr, ha="right",
                    va="bottom", fontsize=9, color=INK2)
        ax.set_xticks(range(len(corpora)), [corpus_label(c) for c in corpora], fontsize=9)
        ax.set_title(title)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    fig.suptitle("Text-generation LLM and prompt (Whisper large-v2, full fine-tune)",
                 x=0.02, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_text_latin(plt, text_stats, out):
    ts = [t for t in text_stats if not t["corpus"].startswith("mucs")]
    target = next((t for t in text_stats if t["corpus"] == "mucs_test"), None)
    if not ts:
        return
    fig, ax = plt.subplots(figsize=(7.2, 0.9 + 0.55 * len(ts)))
    for i, t in enumerate(ts):
        ax.barh(i, t["latin_pct_hi_matrix"], 0.5, color=METHOD_COLOR["full"])
        ax.annotate(f"{t['latin_pct_hi_matrix']:.1f}", (t["latin_pct_hi_matrix"], i),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    if target:
        ax.axvline(target["latin_pct_hi_matrix"], color=INK, lw=1)
        ax.annotate(f"MUCS test {target['latin_pct_hi_matrix']:.1f}",
                    (target["latin_pct_hi_matrix"], len(ts) - 0.45), xytext=(4, 0),
                    textcoords="offset points", fontsize=9, color=INK2)
    ax.set_yticks(range(len(ts)), [t["label"] for t in ts])
    ax.invert_yaxis()
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_xlabel("Latin-script share of words in Hindi-matrix sentences (%)")
    ax.set_title("How much English the synthetic Hindi sentences keep in Latin script")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results/ablation"))
    ap.add_argument("--exp", type=Path, default=Path("csasr-work/abl"),
                    help="run directories, for the dev curves")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    rd = args.results_dir
    rows = load(rd, args.n_boot)
    if not rows:
        print(f"[report] no finished runs under {rd}/rows yet")
        return 0
    for r in rows:
        r.setdefault("tag", "")
    ts_path = rd / "text_stats.json"
    text_stats = json.loads(ts_path.read_text(encoding="utf-8")) if ts_path.exists() else []

    (rd / "ablation.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    cols = sorted({k for r in rows for k in r})
    with (rd / "ablation.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    write_report(rows, text_stats, rd / "report.md")
    print((rd / "report.md").read_text(encoding="utf-8"))

    if not args.no_figures:
        plt = _style()
        fig_grid(plt, rows, "mer", "mer_lo", "mer_hi",
                 "Test MER by ASR model and fine-tuning method",
                 "Test MER (%), lower is better", rd / "fig_grid_mer.png")
        fig_grid(plt, rows, "cba_he", "cba_he_lo", "cba_he_hi",
                 "Code-switch bigram accuracy (Hindi→English) by model and method",
                 "CBA-HE (%), higher is better", rd / "fig_grid_cba.png")
        fig_efficiency(plt, rows, rd / "fig_efficiency.png")
        fig_dev_curves(plt, rows, args.exp, rd / "fig_dev_curves.png")
        fig_llm(plt, rows, rd / "fig_llm_axis.png")
        if text_stats:
            fig_text_latin(plt, text_stats, rd / "fig_text_latin.png")
        print(f"[report] figures -> {rd}/fig_*.png")
    print(f"[report] {len(rows)} run(s) -> {rd}/report.md, ablation.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
