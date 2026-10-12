"""Show transcripts so that code-switching is visible: English words stand out.

Terminal output marks English-script (Latin) words in bold cyan; the HTML report
marks them with a tinted, bold chip, so the signal never depends on colour alone.
Statistics are script-based and need no reference:

  english %   share of Hindi+English words that are written in Latin script
  switches    adjacent word pairs where the script changes

A model that has not learned code-switching writes English terms in Devanagari
("टीम" for "team"), which shows up here as a low english % and few switches.
"""

from __future__ import annotations

import base64
import html
import io
import os
import sys
from dataclasses import dataclass

from ..lid import Lang, tag_words, word_lang

__all__ = ["Stats", "analyse", "ansi", "html_text", "score_mer", "build_html", "use_color"]

SR = 16_000


@dataclass
class Stats:
    words: int
    english_pct: float
    switches: int


def analyse(text: str) -> Stats:
    langs = [lang for _, lang in tag_words(text) if lang in (Lang.HI, Lang.EN)]
    n_en = sum(1 for lang in langs if lang is Lang.EN)
    return Stats(
        words=len(langs),
        english_pct=100.0 * n_en / len(langs) if langs else 0.0,
        switches=sum(1 for a, b in zip(langs, langs[1:]) if a is not b),
    )


def score_mer(refs: list[str], hyps: list[str]) -> float:
    """Hybrid MER (characters on Devanagari, words on Latin), as in the report."""
    from ..eval.mer import mer

    return mer(refs, hyps, mode="hybrid")


def _is_english(token: str) -> bool:
    return word_lang(token) in (Lang.EN, Lang.MIXED)


def use_color(stream=None) -> bool:
    stream = stream or sys.stdout
    return stream.isatty() and "NO_COLOR" not in os.environ


def ansi(text: str, color: bool = True) -> str:
    if not color:
        return text
    return " ".join(f"\033[1;36m{t}\033[0m" if _is_english(t) else t for t in text.split())


def html_text(text: str) -> str:
    return " ".join(f'<span class="en">{html.escape(t)}</span>' if _is_english(t)
                    else html.escape(t) for t in text.split())


def _audio_tag(audio) -> str:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="FLAC")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<audio controls preload="none" src="data:audio/flac;base64,{b64}"></audio>'


CSS = """
:root{--bg:#fcfcfb;--card:#fff;--ink:#0b0b0b;--ink2:#52514e;--line:#e7e6e1;
--chip:#dbe8fb;--chipink:#0b3a78;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--card:#232321;--ink:#fff;
--ink2:#c3c2b7;--line:#3a3a37;--chip:#1f3a63;--chipink:#cfe1fb;--accent:#3987e5}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans Devanagari",sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px 56px}
h1{font-size:1.5rem;margin:0 0 4px}h2{font-size:1.1rem;margin:32px 0 10px}
.sub{color:var(--ink2);margin:0 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;
margin:14px 0}
.meta{color:var(--ink2);font-size:.85rem;margin-bottom:8px}
audio{width:100%;margin:6px 0 10px}
.row{display:grid;grid-template-columns:190px 1fr;gap:6px 14px;padding:9px 0;
border-top:1px solid var(--line)}
.row:first-of-type{border-top:0}
.name{font-weight:600;font-size:.9rem}.name small{display:block;color:var(--ink2);
font-weight:400}
.txt{font-size:1.02rem;overflow-wrap:anywhere}
.stats{color:var(--ink2);font-size:.82rem;margin-top:3px}
.en{background:var(--chip);color:var(--chipink);font-weight:600;border-radius:4px;padding:0 3px}
.ref .name{color:var(--accent)}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}th{color:var(--ink2);font-weight:600}
.best{font-weight:700}
.legend{font-size:.88rem;color:var(--ink2)}
.note{font-size:.82rem;color:var(--ink2);margin-top:28px}
@media (max-width:640px){.row{grid-template-columns:1fr}}
"""


def build_html(title: str, subtitle: str, clips, models, results, *, notes: str = "",
               score: bool = True) -> str:
    """One self-contained page.

    clips: objects with clip_id, ref (or None), source, optional audio
    models: DemoModel list, in display order
    results: {model.key: {clip_id: Result-like with .text and .seconds}}
    score: compute MER per excerpt; off when hypotheses are not aligned to the reference
    """
    have_ref = bool(clips) and all(c.ref for c in clips)
    score = score and have_ref
    stats = {(m.key, c.clip_id): analyse(results[m.key][c.clip_id].text)
             for m in models for c in clips if c.clip_id in results.get(m.key, {})}

    # ---- summary table
    head = ["System", "Test MER, whole set"]
    if score:
        head.append("MER, these excerpts")
    head += ["English words %", "Switches", "Decode s"]
    summary = []
    if have_ref:
        ref_stats = [analyse(c.ref) for c in clips]
        summary.append(("Reference (human transcript)", "–", "–" if score else None,
                        sum(s.english_pct * s.words for s in ref_stats)
                        / max(1, sum(s.words for s in ref_stats)),
                        sum(s.switches for s in ref_stats), None))
    mers = {}
    for m in models:
        done = [c for c in clips if c.clip_id in results.get(m.key, {})]
        if not done:
            continue
        hyps = [results[m.key][c.clip_id].text for c in done]
        mers[m.key] = score_mer([c.ref for c in done], hyps) if score else None
        st = [stats[(m.key, c.clip_id)] for c in done]
        words = max(1, sum(s.words for s in st))
        summary.append((m.label, f"{m.mer:.1f}" if m.mer is not None else "–",
                        f"{mers[m.key]:.1f}" if score else None,
                        sum(s.english_pct * s.words for s in st) / words,
                        sum(s.switches for s in st),
                        sum(results[m.key][c.clip_id].seconds for c in done)))
    best = min((v, k) for k, v in mers.items() if v is not None)[1] if score and mers else None

    rows = []
    for label, full, ex, en, sw, secs in summary:
        key = next((m.key for m in models if m.label == label), None)
        cls = ' class="best"' if key and key == best else ""
        cells = [label, full] + ([ex] if score else []) + [
            f"{en:.1f}", str(sw), "–" if secs is None else f"{secs:.1f}"]
        rows.append(f"<tr{cls}>" + "".join(
            f"<td>{html.escape(str(c))}</td>" for c in cells) + "</tr>")
    table = ("<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in head)
             + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")

    # ---- per-clip cards
    cards = []
    for n, c in enumerate(clips, 1):
        audio = getattr(c, "audio", None)
        body = [f'<div class="meta">Clip {n} · {html.escape(c.source)}'
                + (f" · {len(audio) / SR:.0f} s" if audio is not None else "") + "</div>"]
        if audio is not None:
            body.append(_audio_tag(audio))
        if c.ref:
            s = analyse(c.ref)
            body.append(f'<div class="row ref"><div class="name">Reference</div><div>'
                        f'<div class="txt">{html_text(c.ref)}</div>'
                        f'<div class="stats">{s.english_pct:.0f}% English words · '
                        f'{s.switches} switches</div></div></div>')
        for m in models:
            r = results.get(m.key, {}).get(c.clip_id)
            if r is None:
                continue
            s = stats[(m.key, c.clip_id)]
            bits = [f"{s.english_pct:.0f}% English words", f"{s.switches} switches"]
            if score:
                bits.insert(0, f"MER {score_mer([c.ref], [r.text]):.1f}")
            if r.seconds:
                bits.append(f"{r.seconds:.1f} s")
            sub = f"<small>step {m.step}</small>" if m.step is not None else ""
            body.append(f'<div class="row"><div class="name">{html.escape(m.label)}{sub}</div>'
                        f'<div><div class="txt">{html_text(r.text) or "(no speech detected)"}'
                        f'</div><div class="stats">{" · ".join(bits)}</div></div></div>')
        cards.append('<div class="card">' + "".join(body) + "</div>")

    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>"
        f"<h1>{html.escape(title)}</h1><p class=sub>{html.escape(subtitle)}</p>"
        '<p class=legend>English-script words are shown as <span class="en">chips</span>. '
        "A model that has not learned code-switching writes them in Devanagari instead.</p>"
        f"<h2>Summary</h2>{table}<h2>Clips</h2>{''.join(cards)}"
        f"<p class=note>{html.escape(notes)}</p></main></body></html>")
