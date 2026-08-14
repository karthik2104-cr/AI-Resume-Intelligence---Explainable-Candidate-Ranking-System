"""Logging utilities for V2."""

from __future__ import annotations

import logging
import sys
from typing import Optional

from src.utils.config import get_settings


def setup_logging(level: Optional[str] = None) -> None:
    """Configure root logger from application settings."""
    settings = get_settings()
    log_level = level or settings.observability.log_level
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=settings.observability.log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)
