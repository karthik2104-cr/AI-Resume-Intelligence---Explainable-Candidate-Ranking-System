"""JD parsing quality assessment."""

from __future__ import annotations

from src.models.job import JobParsingQuality, ParsedJobDescription


def assess_jd_parsing_quality(parsed: ParsedJobDescription) -> JobParsingQuality:
    """
    Transparent rule-based JD parsing quality — not an ML confidence score.
    """
    score = 0
    warnings: list[str] = list(parsed.parsing_warnings)

    if parsed.title or parsed.job_title:
        score += 1
    else:
        warnings.append("No job title detected.")

    if parsed.sections:
        score += 1
    else:
        warnings.append("No section headings detected.")

    if parsed.required_skills:
        score += 2
    else:
        warnings.append("No required skills detected.")

    if parsed.preferred_skills:
        score += 1

    if parsed.experience_requirements:
        score += 1
    else:
        warnings.append("No explicit experience requirement detected.")

    if parsed.education_requirements:
        score += 1

    if parsed.responsibilities:
        score += 1
    else:
        warnings.append("No responsibilities extracted.")

    if score >= 7:
        level = "high"
    elif score >= 4:
        level = "medium"
    else:
        level = "low"

    return JobParsingQuality(score=score, level=level, warnings=warnings)
