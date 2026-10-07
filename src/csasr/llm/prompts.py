"""Prompts, transcribed verbatim from the paper. Do not paraphrase.

Sections 3.2.1 (bigram generation) and 3.2.2 (sentence generation) quote the
exact system/user prompts. The PDF-to-text extraction mangled the Devanagari
few-shot examples into a transliteration ("pr-tEt document; bEnyAdF
formatting; isspoken").

All three examples are recoverable from the corpus itself: they are lifted
verbatim from the opening sentence of the MUCS Hindi-English test transcripts,

    लिबर ऑफिस impress में एक प्रस्तुति document बनाना और बुनियादी formatting
    के इस spoken tutorial में आपका स्वागत है

giving the mapping

    pr-tEt   -> प्रस्तुति   ("presentation")   [1 occurrence in test `text`]
    bEnyAdF  -> बुनियादी    ("basic")          [1 occurrence]
    is       -> इस          ("this")           [5 occurrences, as "इस spoken"]

Note `pr-tEt` is NOT प्रतीत ("appears"), which occurs zero times in the corpus.

The translation-check prompt (§3.2.1: "used the Llama to verify that paired
words were not direct translations") has no verbatim text in the paper. Ours is
marked below and batches 20 bigrams per call to keep the token budget down.
"""

from __future__ import annotations

__all__ = [
    "BIGRAM_SYSTEM",
    "BIGRAM_USER_TEMPLATE",
    "SENTENCE_SYSTEM",
    "SENTENCE_USER_TEMPLATE",
    "SENTENCE_USER_TEMPLATE_MATCHED",
    "TRANSLATION_CHECK_SYSTEM",
    "TRANSLATION_CHECK_USER_TEMPLATE",
    "bigram_messages",
    "sentence_messages",
    "translation_check_messages",
]

# --------------------------------------------------------------------------
# Stage 1: bigram generation  (paper §3.2.1, verbatim)
# --------------------------------------------------------------------------

BIGRAM_SYSTEM = (
    "You're a helpful mix-bigrams generator, fluent in both English and Hindi, "
    "and that only prints out the bigrams and no additional comments."
)

BIGRAM_USER_TEMPLATE = (
    "Generate {n} language mix bigrams in Hindi-English. Each bigram is a couple "
    "of different words that usually go together and often written this way: one "
    "in Hindi and the other one in English. Never put just a translation of one "
    "word next to each other. Hindi words should be in Devnagari and English "
    "words should be in Latin. Follow these examples: प्रस्तुति document; "
    "बुनियादी formatting; इस spoken. Use these phrases with mixed language for "
    "inspiration. Use different words for bigrams different from the ones in the "
    "text: {examples}"
)

# --------------------------------------------------------------------------
# Stage 2: sentence generation  (paper §3.2.2, verbatim)
# --------------------------------------------------------------------------

SENTENCE_SYSTEM = (
    "You're a helpful sentence generator, fluent in both English and Hindi, and "
    "that output only required sentences without any additional comments."
)

SENTENCE_USER_TEMPLATE = (
    "Generate four sentences that would use this bigram in Hindi-English in a "
    "natural way. Make 2 sentences with English as the main language and 2 "
    "sentences with Hindi as the main language. Word bigram to use: {bigram}"
)

# --------------------------------------------------------------------------
# Stage 2, ablation variant "matched".  OURS -- NOT from the paper.
#
# The verbatim prompt above asks for 2 English-matrix and 2 Hindi-matrix
# sentences and shows the model no in-domain sentence. Measured on the corpus it
# produced: the Hindi-matrix half is 9% Latin-script (exactly one English word,
# everything else pure Hindi), the English-matrix half 92%. The target domain is
# Hindi-matrix at 24% Latin, and a Whisper fine-tuned on that corpus reproduces
# the 9% -- it starts writing English terms in Devanagari and loses to zero-shot.
#
# This variant changes only what that diagnosis implicates: all four sentences
# are Hindi-matrix, and the model is shown real in-domain sentences so it
# carries over THEIR convention for which words stay in Latin script. The paper
# invites exactly this ("the prompts can be further refined").
# --------------------------------------------------------------------------

SENTENCE_USER_TEMPLATE_MATCHED = (
    "Generate four sentences that would use this bigram in Hindi-English in a "
    "natural way. Write all four sentences with Hindi as the main language, in "
    "the style of the example sentences below: spoken software-tutorial "
    "narration where technical and English-origin words are kept in English, "
    "written in Latin script, exactly as the examples do, and all other words "
    "are Hindi written in Devanagari. Example sentences: {examples} "
    "Word bigram to use: {bigram}"
)

# --------------------------------------------------------------------------
# Bigram filtering: translation check.  OURS - the paper describes but does not
# quote this prompt. Batched to amortize the system prompt.
# --------------------------------------------------------------------------

TRANSLATION_CHECK_SYSTEM = (
    "You judge whether a Hindi word and an English word are direct translations "
    "of each other. Answer with one line per numbered item, in the form "
    "'<number>: YES' if the two words mean the same thing (they are direct "
    "translations), or '<number>: NO' if they do not. Print nothing else."
)

TRANSLATION_CHECK_USER_TEMPLATE = (
    "For each numbered pair below, answer YES if the Hindi word and the English "
    "word are direct translations of one another, otherwise NO.\n\n{items}"
)


def bigram_messages(example_sentences: list[str], n: int = 10) -> list[dict]:
    """Few-shot bigram prompt seeded with real in-domain sentences."""
    examples = " ".join(s.strip() for s in example_sentences)
    return [
        {"role": "system", "content": BIGRAM_SYSTEM},
        {"role": "user", "content": BIGRAM_USER_TEMPLATE.format(n=n, examples=examples)},
    ]


def sentence_messages(
    bigram: str, *, variant: str = "paper", example_sentences: list[str] | None = None
) -> list[dict]:
    """`variant="paper"` is the verbatim prompt; `"matched"` is the ablation arm."""
    if variant == "paper":
        user = SENTENCE_USER_TEMPLATE.format(bigram=bigram)
    elif variant == "matched":
        if not example_sentences:
            raise ValueError("the 'matched' prompt needs in-domain example sentences")
        examples = " ".join(s.strip().rstrip("।.") + "." for s in example_sentences)
        user = SENTENCE_USER_TEMPLATE_MATCHED.format(bigram=bigram, examples=examples)
    else:
        raise ValueError(f"unknown prompt variant {variant!r}")
    return [
        {"role": "system", "content": SENTENCE_SYSTEM},
        {"role": "user", "content": user},
    ]


def translation_check_messages(bigrams: list[tuple[str, str]]) -> list[dict]:
    """`bigrams` is a list of (hindi_word, english_word)."""
    items = "\n".join(f"{i+1}. {hi} / {en}" for i, (hi, en) in enumerate(bigrams))
    return [
        {"role": "system", "content": TRANSLATION_CHECK_SYSTEM},
        {"role": "user", "content": TRANSLATION_CHECK_USER_TEMPLATE.format(items=items)},
    ]
