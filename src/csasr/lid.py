"""Word-level script language identification for Hindi-English code-switched text.

This is the primitive that Gate 0 (Table 1 counts), the bigram filter, and the
CBA metric all rest on. It is deliberately deterministic and dependency-free:
language is decided by *script*, not by a model or a lexicon.

Hindi is written in Devanagari, English in Latin. A code-switch point is any
adjacent word pair whose languages differ.
"""

from __future__ import annotations

import unicodedata
from enum import Enum
from typing import Iterator, NamedTuple

__all__ = [
    "Lang",
    "CSBigram",
    "is_devanagari",
    "is_latin",
    "word_lang",
    "split_script_boundaries",
    "tokenize",
    "tag_words",
    "cs_bigrams",
    "all_bigrams",
    "count_words",
]


class Lang(str, Enum):
    HI = "hi"
    EN = "en"
    #: contains both scripts inside one whitespace-delimited token
    MIXED = "mixed"
    #: digits, punctuation, or any script that is neither Devanagari nor Latin
    OTHER = "other"


# Devanagari main block. Devanagari Extended (U+A8E0-U+A8FF) and the Vedic
# Extensions (U+1CD0-U+1CFF) are included for completeness; MUCS uses the main
# block almost exclusively.
_DEVANAGARI_RANGES: tuple[tuple[int, int], ...] = (
    (0x0900, 0x097F),
    (0x1CD0, 0x1CFF),
    (0xA8E0, 0xA8FF),
)


def _in_ranges(cp: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(lo <= cp <= hi for lo, hi in ranges)


def is_devanagari(ch: str) -> bool:
    """True if `ch` is a Devanagari *letter or combining mark*.

    Block membership alone is not enough: the danda `।` (U+0964) and the
    Devanagari digits (U+0966-U+096F) live inside the Devanagari block but are
    punctuation and digits, not script. Counting them as Hindi would make a bare
    `।` a Hindi word. We therefore require category L* or M*, which mirrors the
    Latin side's `isalpha()` test.
    """
    if not _in_ranges(ord(ch), _DEVANAGARI_RANGES):
        return False
    return unicodedata.category(ch)[0] in ("L", "M")


def is_latin(ch: str) -> bool:
    """True if `ch` is a Latin letter (ASCII or accented)."""
    if not ch.isalpha():
        return False
    # unicodedata.name is the robust way to distinguish Latin from Devanagari
    # letters without hardcoding accented ranges.
    try:
        return unicodedata.name(ch).startswith("LATIN")
    except ValueError:  # unnamed codepoint
        return False


def word_lang(word: str) -> Lang:
    """Classify a single whitespace-delimited token by script.

    A token containing any Devanagari and no Latin is Hindi; the converse is
    English. Tokens carrying both scripts are MIXED (e.g. a Devanagari word with
    a stray Latin character). Tokens with neither — bare digits, punctuation —
    are OTHER.
    """
    has_dev = False
    has_lat = False
    for ch in word:
        if is_devanagari(ch):
            has_dev = True
        elif is_latin(ch):
            has_lat = True
        if has_dev and has_lat:
            return Lang.MIXED
    if has_dev:
        return Lang.HI
    if has_lat:
        return Lang.EN
    return Lang.OTHER


def split_script_boundaries(token: str) -> list[str]:
    """Split a token wherever Devanagari meets Latin.

    The MUCS transcripts contain dropped-space typos - `दायाँclick`,
    `करेंspoken`, `detailsको` - which hide a real code-switch point inside a
    single whitespace token. Splitting them recovers the switch point and brings
    our Hindi word count to within 4 of the paper's (Gate 0).

    Characters that are neither script (digits, punctuation) attach to the run
    in progress, so `mp3` stays one token and `gnu/लिनक्स` becomes `gnu/`,
    `लिनक्स`.
    """
    parts: list[str] = []
    cur: list[str] = []
    cur_kind: str | None = None

    for ch in token:
        kind = "d" if is_devanagari(ch) else ("l" if is_latin(ch) else None)
        if kind is None or cur_kind is None or kind == cur_kind:
            cur.append(ch)
            cur_kind = cur_kind or kind
        else:
            parts.append("".join(cur))
            cur = [ch]
            cur_kind = kind

    if cur:
        parts.append("".join(cur))
    return [p for p in parts if p]


def tokenize(text: str, *, split_mixed: bool = True) -> list[str]:
    """Whitespace split, then split any token straddling both scripts.

    Normalization is `normalize.py`'s job, not ours. `split_mixed=False` gives
    the naive whitespace tokenization, kept for Gate 0's comparison grid.
    """
    toks = text.split()
    if not split_mixed:
        return toks
    out: list[str] = []
    for t in toks:
        if word_lang(t) is Lang.MIXED:
            out.extend(split_script_boundaries(t))
        else:
            out.append(t)
    return out


def tag_words(text: str, **kw) -> list[tuple[str, Lang]]:
    """Tokenize and tag each token with its language."""
    return [(w, word_lang(w)) for w in tokenize(text, **kw)]


def count_words(text: str, **kw) -> dict[Lang, int]:
    """Per-language token counts for one utterance.

    Table 1 reports Words_H + Words_E == TotalWords exactly, implying the
    authors' tokenizer emitted no OTHER tokens (they appear to have expanded
    digits into words). Ours keeps digits as OTHER; Gate 0 records the residual.
    """
    counts = {lang: 0 for lang in Lang}
    for _, lang in tag_words(text, **kw):
        counts[lang] += 1
    return counts


class CSBigram(NamedTuple):
    first: str
    second: str
    #: "HE" = Hindi word followed by English word; "EH" = the converse
    kind: str

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.first} {self.second}"


def all_bigrams(text: str, **kw) -> list[tuple[str, str]]:
    """Every adjacent token pair, in order. Used for the hypothesis side of CBA."""
    toks = tokenize(text, **kw)
    return list(zip(toks, toks[1:]))


def cs_bigrams(text: str, **kw) -> Iterator[CSBigram]:
    """Yield adjacent word pairs that straddle a Hindi<->English switch point.

    Only HI->EN and EN->HI transitions count. OTHER tokens (digits, symbols)
    never open or close a switch point, so `हर्ट्ज़ 5 hertz` yields nothing across
    the digit. This matches the paper's own bigram rule, which requires each
    bigram to contain "both Hindi and English characters".
    """
    tagged = tag_words(text, **kw)
    for (w1, l1), (w2, l2) in zip(tagged, tagged[1:]):
        if l1 is Lang.HI and l2 is Lang.EN:
            yield CSBigram(w1, w2, "HE")
        elif l1 is Lang.EN and l2 is Lang.HI:
            yield CSBigram(w1, w2, "EH")
