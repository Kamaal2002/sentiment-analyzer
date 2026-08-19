"""
Day 2 — Text preprocessing.

DistilBERT's own tokenizer (WordPiece) already handles lowercasing,
punctuation splitting, and subword tokenization internally. Classic NLP
steps like stemming, stopword removal, or manual regex tokenization are
NOT used here on purpose — they strip information (word order, casing,
punctuation like "!" or "?") that transformer models actually learn from,
and can measurably hurt fine-tuning performance.

What we DO clean by hand: dataset-specific noise that isn't part of
natural language at all — the IMDB dataset stores literal "<br />" HTML
line breaks and occasional stray whitespace, which are scraping artifacts,
not signal.

This module is imported by both the training script (Day 3) and the API
(Day 4), so cleaning + tokenization stays identical everywhere the model
is used.
"""

import re

from transformers import DistilBertTokenizerFast

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256  # covers ~90% of reviews without truncation (see Day 1 EDA)

_HTML_BREAK_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Strip HTML line-break artifacts and collapse whitespace.

    Intentionally minimal: no lowercasing, no punctuation/stopword removal.
    The DistilBERT tokenizer handles that; over-cleaning here would throw
    away signal the model relies on.
    """
    text = _HTML_BREAK_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def get_tokenizer() -> DistilBertTokenizerFast:
    return DistilBertTokenizerFast.from_pretrained(MODEL_NAME)


def tokenize(texts, tokenizer=None):
    """Tokenize a list of (already-cleaned) texts into model-ready tensors."""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
