"""Requirement evidence classification for JD skill mentions."""

from __future__ import annotations

import re

from src.models.job import JobSectionType, SkillRequirementType
from src.parsing.jd_section_patterns import default_section_requirement_type
from src.utils.config import JobParsingConfig, get_settings

# Sections where skills are neutral unless explicit requirement language appears.
NEUTRAL_SKILL_SECTIONS = {
    JobSectionType.RESPONSIBILITIES,
    JobSectionType.ABOUT,
    JobSectionType.DESCRIPTION,
    JobSectionType.BENEFITS,
}

# Phrases that require word-boundary matching to avoid false positives.
# e.g. "must" should not match "mustard"; "required" should not match embed errors.
_WORD_BOUNDARY_REQUIRED_PHRASES = {
    "must",
    "required",
    "mandatory",
    "essential",
    "minimum",
}


def classify_requirement_type(
    sentence: str,
    section_type: JobSectionType,
    config: JobParsingConfig | None = None,
) -> SkillRequirementType:
    """
    Classify skill requirement strength from sentence evidence and section context.

    Rules (evaluated in order):
    1. Preferred phrases → PREFERRED  (checked first to prevent misclassification)
    2. Required phrases  → REQUIRED
    3. Neutral sections  → MENTIONED
    4. Section default   → per section type
    5. Fallback          → MENTIONED

    Neutral mentions (e.g. responsibilities bullets) default to MENTIONED,
    not REQUIRED, unless explicit requirement language is present.
    """
    cfg = config or get_settings().job_parsing
    lower = sentence.lower()

    # Check preferred BEFORE required to avoid e.g. "is required" overriding
    # "nice to have" in the same sentence.
    for phrase in sorted(cfg.preferred_phrases, key=len, reverse=True):
        if _phrase_matches(lower, phrase):
            return SkillRequirementType.PREFERRED

    for phrase in sorted(cfg.required_phrases, key=len, reverse=True):
        if _phrase_matches(lower, phrase):
            return SkillRequirementType.REQUIRED

    if section_type in NEUTRAL_SKILL_SECTIONS:
        return SkillRequirementType.MENTIONED

    section_default = default_section_requirement_type(section_type)
    if section_default == "preferred":
        return SkillRequirementType.PREFERRED
    if section_default == "required":
        return SkillRequirementType.REQUIRED

    if section_type == JobSectionType.QUALIFICATIONS:
        return SkillRequirementType.REQUIRED

    return SkillRequirementType.MENTIONED


def _phrase_matches(text: str, phrase: str) -> bool:
    """
    Match a phrase against text with word-boundary safety.

    Ambiguous single-word phrases (e.g. 'must', 'required', 'mandatory') use
    strict word boundaries so they don't accidentally match inside other words.
    Multi-word phrases use simple substring matching (already specific enough).
    """
    if phrase in _WORD_BOUNDARY_REQUIRED_PHRASES:
        return bool(re.search(rf"\b{re.escape(phrase)}\b", text))
    # Multi-word or specific phrases: substring match is safe and intentional.
    return phrase in text