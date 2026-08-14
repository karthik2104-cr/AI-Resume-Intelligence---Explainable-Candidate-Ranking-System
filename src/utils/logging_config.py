"""Structured logging configuration and PII redaction filter.

Provides a helper to configure the root logger using application settings and
a filter that redacts emails and phone-like patterns from log messages so
that accidental PII does not appear in logs.
"""
from __future__ import annotations

import logging
import re
from typing import Pattern


_EMAIL_RE: Pattern = re.compile(r"[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}")
# Very permissive phone-like pattern: sequences of digits, spaces, +, -, parentheses
_PHONE_RE: Pattern = re.compile(r"(\+?\d[\d\s\-()]{6,}\d)")


def redact_pii(text: str) -> str:
    if not text:
        return text
    t = _EMAIL_RE.sub("<REDACTED_EMAIL>", text)
    t = _PHONE_RE.sub("<REDACTED_PHONE>", t)
    return t


class PiiFilter(logging.Filter):
    """Logging filter that redacts PII from record messages in-place."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            # Format the message safely and redact
            msg = record.getMessage()
            redacted = redact_pii(msg)
            # Set msg to redacted plain string and clear args to avoid reformatting
            record.msg = redacted
            record.args = ()
        except Exception:
            # Never fail logging due to filter
            pass
        return True


def configure_logging(level: str | int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)

    # Basic handler if none configured
    if not root.handlers:
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        ch.setFormatter(fmt)
        root.addHandler(ch)

    # Add PII filter globally
    pf = PiiFilter()
    root.addFilter(pf)
