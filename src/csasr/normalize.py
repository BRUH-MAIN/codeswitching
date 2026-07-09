"""Text normalization for Hindi-English code-switched transcripts.

WHY THIS IS NOT `transformers.models.whisper.english_normalizer.BasicTextNormalizer`
-----------------------------------------------------------------------------------
Whisper's `BasicTextNormalizer` replaces every character whose Unicode category
starts with M, S, or P with a space. Devanagari vowel signs, the virama, and the
nukta are *combining marks* (categories Mn/Mc), so that normalizer destroys Hindi:

    दस्तावेज़  ->  ['दस', 'त', 'व', 'ज']      # one word becomes four

That would inflate Hindi word counts several-fold, break Gate 0, and silently
corrupt MER and CBA. We therefore strip only punctuation (P*) and symbols (S*)
and always preserve marks. The Devanagari danda `।` is category Po, so it is
still removed as punctuation, which is what we want.

`whisper_basic` is provided *only* so Gate 0 can demonstrate the discrepancy.
Never use it for scoring.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable

__all__ = ["normalize", "PRESETS", "collapse_ws"]

_WS = re.compile(r"\s+")


def collapse_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _strip_categories(text: str, cats: str, *, form: str = "NFKC") -> str:
    """Replace every char whose Unicode category starts with a letter in `cats`."""
    return "".join(
        " " if unicodedata.category(ch)[0] in cats else ch
        for ch in unicodedata.normalize(form, text)
    )


def _raw(text: str) -> str:
    """Whitespace collapse only. Nothing else touched."""
    return collapse_ws(text)


def _punct(text: str) -> str:
    """Strip punctuation and symbols; preserve case and all combining marks."""
    return collapse_ws(_strip_categories(text, "PS"))


def _scoring(text: str) -> str:
    """Default for MER / CBA.

    Strips punctuation and symbols, lowercases (a no-op for Devanagari, which is
    caseless), preserves every combining mark.
    """
    return collapse_ws(_strip_categories(text, "PS")).lower()


def _whisper_basic(text: str) -> str:
    """DESTRUCTIVE for Devanagari. Present only for Gate 0's comparison grid."""
    return collapse_ws(_strip_categories(text, "MSP")).lower()


PRESETS: dict[str, Callable[[str], str]] = {
    "raw": _raw,
    "punct": _punct,
    "scoring": _scoring,
    "whisper_basic": _whisper_basic,
}


def normalize(text: str, preset: str = "scoring") -> str:
    try:
        fn = PRESETS[preset]
    except KeyError:
        raise ValueError(
            f"unknown preset {preset!r}; choose from {sorted(PRESETS)}"
        ) from None
    return fn(text)
