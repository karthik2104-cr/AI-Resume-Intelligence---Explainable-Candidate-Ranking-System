"""Text preprocessing utilities for V2."""

from __future__ import annotations

import re
from typing import Optional

from src.utils.config import PreprocessingConfig, get_settings

URL_PATTERN = re.compile(r"http\S+\s*", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\S+")
HASHTAG_PATTERN = re.compile(r"#\S+")
NON_ASCII_PATTERN = re.compile(r"[^\x00-\x7f]")
WHITESPACE_PATTERN = re.compile(r"\s+")
PUNCTUATION_PATTERN = re.compile(
    r"[%s]" % re.escape("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
)


def clean_text(
    text: str,
    config: Optional[PreprocessingConfig] = None,
    *,
    lowercase: Optional[bool] = None,
) -> str:
    """
    Normalize resume or job description text.

    Compatible with the legacy notebook/app cleaning pipeline while being
    configurable via PreprocessingConfig.
    """
    if not text:
        return ""

    cfg = config or get_settings().preprocessing
    use_lower = cfg.lowercase if lowercase is None else lowercase

    cleaned = text
    if cfg.remove_urls:
        cleaned = URL_PATTERN.sub(" ", cleaned)
        cleaned = re.sub(r"\bRT\b|\bcc\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = HASHTAG_PATTERN.sub("", cleaned)
        cleaned = MENTION_PATTERN.sub("", cleaned)

    cleaned = PUNCTUATION_PATTERN.sub(" ", cleaned)

    if cfg.remove_non_ascii:
        cleaned = NON_ASCII_PATTERN.sub(" ", cleaned)

    if cfg.normalize_whitespace:
        cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()

    if use_lower:
        cleaned = cleaned.lower()

    return cleaned
