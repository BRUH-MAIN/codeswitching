"""Speaker descriptions for `ai4bharat/indic-parler-tts`.

The paper (§3.2.2) synthesises with "one Indian male and one Indian female
voice". The model card's recommended Hindi speakers are Rohit (male) and Divya
(female), so those are the two.

The literal phrase "very clear audio" is documented on the model card as raising
output quality; it is not decoration.

The model auto-detects language from the prompt text, so no language tag is
needed in the description - which is exactly what we want for code-mixed input.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

__all__ = ["Speaker", "SPEAKERS", "assign_speaker", "description_for"]


@dataclass(frozen=True, slots=True)
class Speaker:
    name: str
    gender: str

    @property
    def description(self) -> str:
        return (
            f"{self.name} speaks at a moderate pace with a neutral tone. "
            f"The recording is very clear audio, close-sounding, with minimal "
            f"background noise."
        )


SPEAKERS: tuple[Speaker, ...] = (
    Speaker("Rohit", "male"),
    Speaker("Divya", "female"),
)


def assign_speaker(sent_id: str) -> Speaker:
    """Deterministic 50/50 split keyed on sentence id.

    Python's `hash()` is salted per process (PYTHONHASHSEED), so a run resumed
    after a Kaggle timeout would reassign voices and silently produce a corpus
    with inconsistent speakers. Use a stable digest instead.
    """
    digest = hashlib.sha1(sent_id.encode("utf-8")).digest()
    return SPEAKERS[digest[0] % len(SPEAKERS)]


def description_for(sent_id: str) -> tuple[Speaker, str]:
    sp = assign_speaker(sent_id)
    return sp, sp.description
