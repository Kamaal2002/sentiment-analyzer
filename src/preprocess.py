import re

from transformers import DistilBertTokenizerFast

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256

_HTML_BREAK_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    # No lowercasing/stopword removal here — DistilBERT's tokenizer handles
    # that, and stripping it manually tends to hurt fine-tuning accuracy.
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
