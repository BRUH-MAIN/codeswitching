"""Regroup per-utterance rows into whole recordings.

The paper decodes with WhisperX, which transcribes an entire recording: it
detects the language once, feeds 30-second chunks with real surrounding context,
and applies temperature fallback. Decoding isolated 6-second clips instead is a
different -- and much harder -- task, and it destroys the very thing CBA
measures. With no surrounding context Whisper renders English loanwords in
Devanagari, so the hypothesis contains no script boundary and *no switch bigram
can match*. Measured on real test audio: hypotheses went from 0.0% Latin
(per-segment) to 12.2% Latin (long-form), against a 21.6% Latin reference.

MUCS utterance ids are `<speaker>_<recording>_<index>`, and the trailing index
is chronological (verified against the `segments` file for all 30 test
recordings), so we can rebuild each recording without any extra metadata.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

__all__ = ["recording_id", "segment_index", "group_by_recording", "concat_refs"]


def recording_id(utt_id: str) -> str:
    """`103085_w5Jyq3XMbb3WwiKQ_0000` -> `w5Jyq3XMbb3WwiKQ`."""
    parts = utt_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"unexpected utt_id shape: {utt_id!r} (want <spk>_<reco>_<idx>)")
    return parts[-2]


def segment_index(utt_id: str) -> int:
    """Trailing zero-padded index. Chronological within a recording."""
    tail = utt_id.split("_")[-1]
    if not tail.isdigit():
        raise ValueError(f"utt_id {utt_id!r} does not end in a numeric index")
    return int(tail)


def group_by_recording(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Bucket rows by recording, each list sorted chronologically."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[recording_id(r["utt_id"])].append(r)
    for g in groups.values():
        g.sort(key=lambda r: segment_index(r["utt_id"]))
    return dict(groups)


def concat_refs(rows: Iterable[dict[str, Any]], field: str = "text") -> dict[str, str]:
    """One reference string per recording, segments joined in order."""
    return {
        reco: " ".join(r[field].strip() for r in g if r.get(field, "").strip())
        for reco, g in group_by_recording(rows).items()
    }
