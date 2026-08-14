"""Extract structured entries from resume section content."""

from __future__ import annotations

import re

from src.extraction.skill_normalizer import normalize_skill
from src.models.resume import (
    CertificationEntry,
    EducationEntry,
    ExperienceEntry,
    ProjectEntry,
)

DATE_TOKEN = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
DATE_RANGE_PATTERN = re.compile(
    rf"(?P<start>{DATE_TOKEN}\.?\s*\d{{4}}|\d{{1,2}}/\d{{4}}|\d{{4}})"
    rf"\s*[-–—to]+\s*"
    rf"(?P<end>Present|Current|Now|{DATE_TOKEN}\.?\s*\d{{4}}|\d{{1,2}}/\d{{4}}|\d{{4}})",
    re.IGNORECASE,
)
YEAR_RANGE_PATTERN = re.compile(
    r"(?P<start>\d{4})\s*[-–—to]+\s*(?P<end>Present|Current|\d{4})",
    re.IGNORECASE,
)
DEGREE_PATTERN = re.compile(
    r"\b("
    r"B\.?\s?(?:Tech|E\.?|Sc\.?|A\.?)|M\.?\s?(?:Tech|E\.?|Sc\.?|A\.?)|"
    r"MBA|Ph\.?\s?D\.?|MCA|BCA|Bachelor|Master|Doctor"
    r")\b",
    re.IGNORECASE,
)
INTERNSHIP_PATTERN = re.compile(r"\bintern(ship)?\b", re.IGNORECASE)
BULLET_LINE = re.compile(r"^[\u2022\*\-\u2013\u2014\u2023\u25E6\u2043]\s+", re.MULTILINE)
SKILL_SPLIT_PATTERN = re.compile(r"[,|;•\n]|(?:\s{2,})")


def split_into_blocks(content: str) -> list[str]:
    """Split section content into logical blocks by blank lines."""
    blocks: list[str] = []
    current: list[str] = []
    for line in content.splitlines():
        if not line.strip():
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return [b for b in blocks if b]


def extract_date_range(text: str) -> tuple[str | None, str | None]:
    match = DATE_RANGE_PATTERN.search(text) or YEAR_RANGE_PATTERN.search(text)
    if not match:
        return None, None
    return match.group("start").strip(), match.group("end").strip()


def extract_skills(content: str) -> list[str]:
    """Parse skill tokens from a skills section."""
    lines = content.splitlines()
    tokens: list[str] = []
    for line in lines:
        cleaned = BULLET_LINE.sub("", line.strip())
        if not cleaned:
            continue
        if cleaned.lower().startswith(("skills", "technical skills", "programming languages")):
            if ":" in cleaned:
                cleaned = cleaned.split(":", 1)[1].strip()
            else:
                continue
        parts = SKILL_SPLIT_PATTERN.split(cleaned)
        for part in parts:
            skill = part.strip(" .-•*")
            if skill and len(skill) > 1 and len(skill) < 60:
                tokens.append(normalize_skill(skill))

    seen: set[str] = set()
    unique: list[str] = []
    for skill in tokens:
        key = skill.lower()
        if key not in seen:
            seen.add(key)
            unique.append(skill)
    return unique


def extract_experience_entries(content: str) -> list[ExperienceEntry]:
    entries: list[ExperienceEntry] = []
    for block in split_into_blocks(content):
        start_date, end_date = extract_date_range(block)
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        title: str | None = None
        organization: str | None = None
        description_lines: list[str] = []

        if lines:
            header = lines[0]
            if " at " in header.lower():
                parts = re.split(r"\s+at\s+", header, maxsplit=1, flags=re.IGNORECASE)
                title = parts[0].strip(" -–—|,")
                organization = parts[1].strip(" -–—|,") if len(parts) > 1 else None
            elif " | " in header:
                parts = [p.strip() for p in header.split("|")]
                title = parts[0] if parts else None
                organization = parts[1] if len(parts) > 1 else None
            else:
                title = header

        for line in lines[1:]:
            if DATE_RANGE_PATTERN.search(line) or YEAR_RANGE_PATTERN.search(line):
                if not start_date:
                    start_date, end_date = extract_date_range(line)
                continue
            description_lines.append(BULLET_LINE.sub("", line))

        entries.append(
            ExperienceEntry(
                title=title,
                organization=organization,
                start_date=start_date,
                end_date=end_date,
                description="\n".join(description_lines).strip(),
                is_internship=bool(INTERNSHIP_PATTERN.search(block)),
                raw_text=block,
            )
        )
    return entries


def extract_education_entries(content: str) -> list[EducationEntry]:
    entries: list[EducationEntry] = []
    for block in split_into_blocks(content):
        start_date, end_date = extract_date_range(block)
        degree_match = DEGREE_PATTERN.search(block)
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]

        institution: str | None = None
        degree = degree_match.group(0) if degree_match else None
        specialization: str | None = None

        for line in lines:
            if DEGREE_PATTERN.search(line):
                if not degree:
                    degree = DEGREE_PATTERN.search(line).group(0)
                remainder = DEGREE_PATTERN.sub("", line).strip(" ,-–—|")
                if remainder and not specialization:
                    specialization = remainder
            elif not institution and len(line) > 3:
                institution = line

        entries.append(
            EducationEntry(
                institution=institution,
                degree=degree,
                specialization=specialization,
                start_date=start_date,
                end_date=end_date,
                raw_text=block,
            )
        )
    return entries


def extract_project_entries(content: str) -> list[ProjectEntry]:
    entries: list[ProjectEntry] = []
    for block in split_into_blocks(content):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        name = lines[0] if lines else None
        if name and BULLET_LINE.match(name):
            name = BULLET_LINE.sub("", name)

        description_lines = [BULLET_LINE.sub("", ln) for ln in lines[1:]]
        tech: list[str] = []
        for line in lines:
            if "technolog" in line.lower() and ":" in line:
                tech_part = line.split(":", 1)[1]
                tech = [t.strip() for t in SKILL_SPLIT_PATTERN.split(tech_part) if t.strip()]

        entries.append(
            ProjectEntry(
                name=name,
                description="\n".join(description_lines).strip(),
                technologies=tech,
                raw_text=block,
            )
        )
    return entries


def extract_certification_entries(content: str) -> list[CertificationEntry]:
    entries: list[CertificationEntry] = []
    for block in split_into_blocks(content):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        name = BULLET_LINE.sub("", lines[0])
        issuer: str | None = None
        date: str | None = None

        start_date, end_date = extract_date_range(block)
        date = end_date or start_date

        if len(lines) > 1:
            issuer = lines[1]

        entries.append(
            CertificationEntry(
                name=name,
                issuer=issuer,
                date=date,
                raw_text=block,
            )
        )
    return entries


def estimate_years_experience(experience_entries: list[ExperienceEntry]) -> float | None:
    """Estimate total years of experience from date ranges (conservative overlap-ignored sum)."""
    total_months = 0
    for entry in experience_entries:
        if not entry.start_date:
            continue
        start_year = _extract_year(entry.start_date)
        end_year = _extract_year(entry.end_date) if entry.end_date else _current_year()
        if start_year and end_year and end_year >= start_year:
            months = (end_year - start_year) * 12
            if entry.is_internship:
                months = int(months * 0.5)
            total_months += max(months, 0)

    if total_months == 0:
        return None
    return round(total_months / 12.0, 1)


def _extract_year(date_str: str) -> int | None:
    if not date_str:
        return None
    lower = date_str.lower()
    if lower in {"present", "current", "now"}:
        return _current_year()
    match = re.search(r"(\d{4})", date_str)
    return int(match.group(1)) if match else None


def _current_year() -> int:
    from datetime import datetime

    return datetime.now().year
