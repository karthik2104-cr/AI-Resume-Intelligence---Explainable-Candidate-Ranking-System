"""Job description section heading detection."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.models.job import JobSectionType
from src.parsing.section_patterns import (
    MARKDOWN_HEADING,
    TRAILING_COLON,
    BULLET_PREFIX,
    is_likely_section_heading,
    normalize_heading_line,
)
from src.utils.config import JobParsingConfig, get_settings


@dataclass(frozen=True)
class JobSectionHeadingMatch:
    section_type: JobSectionType
    title: str
    line_index: int


def _type_map(config: JobParsingConfig) -> dict[str, JobSectionType]:
    mapping = {
        "about": JobSectionType.ABOUT,
        "description": JobSectionType.DESCRIPTION,
        "responsibilities": JobSectionType.RESPONSIBILITIES,
        "requirements": JobSectionType.REQUIREMENTS,
        "required_qualifications": JobSectionType.REQUIRED_QUALIFICATIONS,
        "preferred_qualifications": JobSectionType.PREFERRED_QUALIFICATIONS,
        "qualifications": JobSectionType.QUALIFICATIONS,
        "education": JobSectionType.EDUCATION,
        "experience": JobSectionType.EXPERIENCE,
        "benefits": JobSectionType.BENEFITS,
    }
    return mapping


def build_jd_heading_patterns(config: JobParsingConfig | None = None) -> dict[JobSectionType, list[re.Pattern[str]]]:
    cfg = config or get_settings().job_parsing
    type_map = _type_map(cfg)
    patterns: dict[JobSectionType, list[re.Pattern[str]]] = {}

    for key, section_type in type_map.items():
        keywords = cfg.section_aliases.get(key, [])
        compiled: list[re.Pattern[str]] = []
        for keyword in keywords:
            escaped = re.escape(keyword)
            compiled.append(re.compile(rf"^\s*{escaped}\s*:?\s*$", re.IGNORECASE))
            compiled.append(re.compile(rf"^\s*{escaped}\s*[\-\u2013\u2014]\s*.{{0,40}}$", re.IGNORECASE))
        patterns[section_type] = compiled
    return patterns


def detect_jd_section_heading(
    line: str,
    line_index: int,
    patterns: dict[JobSectionType, list[re.Pattern[str]]],
    max_header_line_length: int,
) -> JobSectionHeadingMatch | None:
    if not is_likely_section_heading(line, max_header_line_length):
        return None

    normalized = normalize_heading_line(line)
    for section_type, type_patterns in patterns.items():
        for pattern in type_patterns:
            if pattern.match(normalized) or pattern.match(line.strip()):
                return JobSectionHeadingMatch(
                    section_type=section_type,
                    title=normalized,
                    line_index=line_index,
                )
    return None


def default_section_requirement_type(section_type: JobSectionType) -> str | None:
    """Return 'required', 'preferred', or None for neutral sections."""
    if section_type in {
        JobSectionType.REQUIRED_QUALIFICATIONS,
        JobSectionType.REQUIREMENTS,
    }:
        return "required"
    if section_type in {
        JobSectionType.PREFERRED_QUALIFICATIONS,
    }:
        return "preferred"
    return None
