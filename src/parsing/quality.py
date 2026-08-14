"""Parsing quality assessment for structured resumes."""

from __future__ import annotations

from src.models.resume import ParsedResume, ResumeSectionType
from src.utils.config import ParsingConfig, get_settings


def assess_parsing_quality(
    parsed: ParsedResume,
    config: ParsingConfig | None = None,
) -> str:
    """
    Assign parsing quality: high | medium | low.

    Based on detected sections, contact info, and extracted entities.
    Not a statistical confidence score.
    """
    cfg = config or get_settings().parsing
    score = 0

    if parsed.email or parsed.phone:
        score += 1
    if parsed.name:
        score += 1

    detected_types = {section.section_type for section in parsed.sections}
    if len(detected_types) >= cfg.min_sections_for_high_quality:
        score += 1

    if parsed.skills:
        score += 1
    if parsed.experience:
        score += 1
    if parsed.education:
        score += 1

    has_core = ResumeSectionType.EXPERIENCE in detected_types or ResumeSectionType.SKILLS in detected_types

    if score >= 5 and has_core:
        return "high"
    if score >= 2:
        return "medium"
    return "low"
