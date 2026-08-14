"""Section heading patterns for resume parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.models.resume import ResumeSectionType
from src.utils.config import ParsingConfig, get_settings

MARKDOWN_HEADING = re.compile(r"^#{1,3}\s+(.+)$")
TRAILING_COLON = re.compile(r"^(.{2,50}):$")
BULLET_PREFIX = re.compile(r"^[\u2022\*\-\u2013\u2014\u2023\u25E6\u2043]\s+")


@dataclass(frozen=True)
class SectionHeadingMatch:
    section_type: ResumeSectionType
    title: str
    line_index: int


def build_heading_patterns(config: ParsingConfig | None = None) -> dict[ResumeSectionType, list[re.Pattern[str]]]:
    """Compile regex patterns for each section type from config keywords."""
    cfg = config or get_settings().parsing
    type_map = {
        "summary": ResumeSectionType.SUMMARY,
        "experience": ResumeSectionType.EXPERIENCE,
        "education": ResumeSectionType.EDUCATION,
        "skills": ResumeSectionType.SKILLS,
        "projects": ResumeSectionType.PROJECTS,
        "certifications": ResumeSectionType.CERTIFICATIONS,
        "achievements": ResumeSectionType.ACHIEVEMENTS,
        "publications": ResumeSectionType.PUBLICATIONS,
        "languages": ResumeSectionType.LANGUAGES,
    }

    patterns: dict[ResumeSectionType, list[re.Pattern[str]]] = {}
    for key, section_type in type_map.items():
        keywords = cfg.section_heading_keywords.get(key, [])
        compiled: list[re.Pattern[str]] = []
        for keyword in keywords:
            escaped = re.escape(keyword)
            compiled.append(re.compile(rf"^\s*{escaped}\s*:?\s*$", re.IGNORECASE))
            compiled.append(re.compile(rf"^\s*{escaped}\s*[\-\u2013\u2014]\s*.{{0,30}}$", re.IGNORECASE))
        patterns[section_type] = compiled
    return patterns


def normalize_heading_line(line: str) -> str:
    cleaned = line.strip()
    md_match = MARKDOWN_HEADING.match(cleaned)
    if md_match:
        cleaned = md_match.group(1).strip()
    colon_match = TRAILING_COLON.match(cleaned)
    if colon_match:
        cleaned = colon_match.group(1).strip()
    return cleaned


def is_likely_section_heading(line: str, max_length: int) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > max_length:
        return False
    if BULLET_PREFIX.match(stripped):
        return False
    if "@" in stripped or "http" in stripped.lower():
        return False
    if re.search(r"\d{3}[-.\s]?\d{3}[-.\s]?\d{4}", stripped):
        return False
    return True


def detect_section_heading(
    line: str,
    line_index: int,
    patterns: dict[ResumeSectionType, list[re.Pattern[str]]],
    max_header_line_length: int,
) -> SectionHeadingMatch | None:
    """Return a section match if the line looks like a section heading."""
    if not is_likely_section_heading(line, max_header_line_length):
        return None

    normalized = normalize_heading_line(line)
    for section_type, type_patterns in patterns.items():
        for pattern in type_patterns:
            if pattern.match(normalized) or pattern.match(line.strip()):
                return SectionHeadingMatch(
                    section_type=section_type,
                    title=normalized,
                    line_index=line_index,
                )
    return None
