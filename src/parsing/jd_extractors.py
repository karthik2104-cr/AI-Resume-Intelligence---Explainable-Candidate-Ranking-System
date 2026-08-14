"""Job description field extractors."""

from __future__ import annotations

import re

from src.models.job import (
    EducationRequirement,
    ExperienceContext,
    ExperienceRequirement,
    JobTitle,
    ResponsibilityEntry,
    SkillCategory,
    SkillRequirement,
    SkillRequirementType,
)
from src.extraction.requirement_classifier import classify_requirement_type
from src.extraction.skill_normalizer import (
    dedupe_skill_matches,
    find_skills_in_text,
    get_skill_category,
)
from src.parsing.jd_section_patterns import default_section_requirement_type
from src.models.job import JobSectionType
from src.utils.config import JobParsingConfig, get_settings

TITLE_LABEL_PATTERN = re.compile(
    r"^(?:job title|position|role|opening for|hiring for)\s*[:\-]\s*(.+)$",
    re.IGNORECASE,
)
BULLET_LINE = re.compile(r"^[\u2022\*\-\u2013\u2014\u2023\u25E6\u2043]\s+")

EXPERIENCE_PLUS_PATTERN = re.compile(
    r"(?:at least|minimum|min\.?)\s*(\d+(?:\.\d+)?)\s*\+?\s*years?",
    re.IGNORECASE,
)
EXPERIENCE_YEARS_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*years?(?:\s+of)?",
    re.IGNORECASE,
)
EXPERIENCE_RANGE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*[-–—to]+\s*(\d+(?:\.\d+)?)\s*years?",
    re.IGNORECASE,
)
ENTRY_LEVEL_PATTERN = re.compile(
    r"\b(entry[\s-]?level|fresh graduate|freshers?|no experience required)\b",
    re.IGNORECASE,
)
INTERNSHIP_PATTERN = re.compile(r"\bintern(ship)?\b", re.IGNORECASE)

LOCATION_PATTERN = re.compile(
    r"\b(?:location|based in|office location)\s*[:\-]\s*([A-Za-z\s,]+)",
    re.IGNORECASE,
)
CITY_PATTERN = re.compile(
    r"\b(Bangalore|Bengaluru|Chennai|Mumbai|Delhi|Hyderabad|Pune|Remote|San Francisco|New York|London)\b",
    re.IGNORECASE,
)

DEGREE_INLINE_PATTERN = re.compile(
    r"(?P<level>B\.?\s?Tech|B\.?\s?E\.?|B\.?\s?Sc\.?|M\.?\s?Tech|M\.?\s?Sc\.?|MBA|Ph\.?\s?D\.?|"
    r"Bachelor(?:'s)?|Master(?:'s)?|Doctorate)"
    r"(?:\s+(?:degree\s+)?(?:in|of)\s+(?P<field>[A-Za-z\s&]+))?",
    re.IGNORECASE,
)


def extract_job_title(header_text: str, full_text: str, config: JobParsingConfig | None = None) -> JobTitle | None:
    cfg = config or get_settings().job_parsing
    candidates: list[tuple[str, str]] = []

    for line in header_text.splitlines()[:8]:
        stripped = line.strip()
        if not stripped:
            continue
        label_match = TITLE_LABEL_PATTERN.match(stripped)
        if label_match:
            candidates.append((label_match.group(1).strip(), stripped))
            continue
        if _looks_like_title_line(stripped):
            candidates.append((stripped, stripped))

    if not candidates:
        for line in full_text.splitlines()[:5]:
            stripped = line.strip()
            label_match = TITLE_LABEL_PATTERN.match(stripped)
            if label_match:
                candidates.append((label_match.group(1).strip(), stripped))
                break

    if not candidates:
        return None

    raw_title, evidence = candidates[0]
    seniority = extract_seniority(raw_title, cfg)
    normalized = _strip_seniority_prefix(raw_title, seniority)
    return JobTitle(
        raw_title=raw_title,
        normalized_title=normalized,
        seniority_level=seniority,
        evidence=evidence,
    )


def _looks_like_title_line(line: str) -> bool:
    if len(line) > 90 or len(line.split()) > 12:
        return False
    if "@" in line or "http" in line.lower():
        return False
    title_words = (
        "engineer", "developer", "scientist", "analyst", "manager",
        "architect", "designer", "consultant", "specialist", "intern",
    )
    lower = line.lower()
    return any(word in lower for word in title_words)


def extract_seniority(text: str, config: JobParsingConfig | None = None) -> str | None:
    cfg = config or get_settings().job_parsing
    lower = text.lower()
    for level in sorted(cfg.seniority_levels, key=len, reverse=True):
        if level in lower:
            return level.title() if level.islower() else level
    return None


def _strip_seniority_prefix(title: str, seniority: str | None) -> str:
    if not seniority:
        return title.strip()
    pattern = re.compile(rf"^{re.escape(seniority)}\s+", re.IGNORECASE)
    return pattern.sub("", title).strip()


def extract_employment_type(text: str, config: JobParsingConfig | None = None) -> str | None:
    cfg = config or get_settings().job_parsing
    lower = text.lower()
    for emp_type in sorted(cfg.employment_types, key=len, reverse=True):
        if emp_type in lower:
            return emp_type.title()
    return None


def extract_work_mode(text: str, config: JobParsingConfig | None = None) -> str | None:
    cfg = config or get_settings().job_parsing
    lower = text.lower()
    for mode in sorted(cfg.work_modes, key=len, reverse=True):
        if mode in lower:
            if "on-site" in mode or "on site" in mode:
                return "onsite"
            if "work from home" in mode or mode == "wfh":
                return "remote"
            return mode.lower()
    return None


def extract_location(text: str) -> str | None:
    loc_match = LOCATION_PATTERN.search(text)
    if loc_match:
        return loc_match.group(1).strip()
    city_match = CITY_PATTERN.search(text)
    if city_match:
        return city_match.group(1).strip()
    return None


def classify_sentence_requirement_type(
    sentence: str,
    section_type: JobSectionType,
    config: JobParsingConfig | None = None,
) -> SkillRequirementType:
    """Delegate to unified requirement evidence classifier."""
    return classify_requirement_type(sentence, section_type, config)


def extract_skills_from_text(
    text: str,
    section_type: JobSectionType,
    source_section: str,
    config: JobParsingConfig | None = None,
) -> list[SkillRequirement]:
    cfg = config or get_settings().job_parsing
    soft_set = {s.lower() for s in cfg.soft_skills}
    requirements: list[SkillRequirement] = []

    sentences = re.split(r"(?<=[.!?])\s+|\n", text)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        req_type = classify_sentence_requirement_type(sentence, section_type, cfg)
        matches = dedupe_skill_matches(find_skills_in_text(sentence))
        for match in matches:
            canonical = match.canonical
            cat_label = get_skill_category(canonical)
            category = SkillCategory.SOFT if canonical.lower() in soft_set else SkillCategory.TECHNICAL
            if cat_label and "machine learning" in cat_label.lower():
                category = SkillCategory.TECHNICAL
            requirements.append(
                SkillRequirement(
                    raw_skill=match.matched_text,
                    normalized_skill=canonical,
                    category=category,
                    requirement_type=req_type,
                    evidence=sentence,
                    source_section=source_section,
                )
            )
    return requirements


def extract_soft_skills(text: str, source_section: str, config: JobParsingConfig | None = None) -> list[SkillRequirement]:
    cfg = config or get_settings().job_parsing
    found: list[SkillRequirement] = []
    lower = text.lower()
    for skill in cfg.soft_skills:
        pattern = re.compile(rf"\b{re.escape(skill)}\b", re.IGNORECASE)
        if pattern.search(lower):
            found.append(
                SkillRequirement(
                    raw_skill=skill,
                    normalized_skill=skill.title(),
                    category=SkillCategory.SOFT,
                    requirement_type=SkillRequirementType.REQUIRED,
                    evidence=_find_evidence_line(text, skill),
                    source_section=source_section,
                )
            )
    return found


def _find_evidence_line(text: str, term: str) -> str:
    for line in text.splitlines():
        if term.lower() in line.lower():
            return line.strip()
    return text.strip()[:200]


def extract_responsibilities(content: str, source_section: str) -> list[ResponsibilityEntry]:
    entries: list[ResponsibilityEntry] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        cleaned = BULLET_LINE.sub("", stripped)
        if len(cleaned) > 10:
            entries.append(ResponsibilityEntry(text=cleaned, source_section=source_section))
    return entries


def extract_experience_requirements(text: str, source_section: str) -> list[ExperienceRequirement]:
    requirements: list[ExperienceRequirement] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        lower = sentence.lower()

        if ENTRY_LEVEL_PATTERN.search(lower):
            requirements.append(
                ExperienceRequirement(
                    raw_text=sentence,
                    min_years=0.0,
                    max_years=1.0,
                    is_required=True,
                    evidence=sentence,
                    experience_context=_experience_context(sentence),
                    is_entry_level=True,
                )
            )
            continue

        range_match = EXPERIENCE_RANGE_PATTERN.search(sentence)
        if range_match:
            requirements.append(
                ExperienceRequirement(
                    raw_text=sentence,
                    min_years=float(range_match.group(1)),
                    max_years=float(range_match.group(2)),
                    is_required=_is_required_sentence(sentence),
                    evidence=sentence,
                    experience_context=_experience_context(sentence),
                    accepts_internship=bool(INTERNSHIP_PATTERN.search(sentence)),
                )
            )
            continue

        plus_match = EXPERIENCE_PLUS_PATTERN.search(sentence) or EXPERIENCE_YEARS_PATTERN.search(sentence)
        if plus_match:
            min_years = float(plus_match.group(1))
            requirements.append(
                ExperienceRequirement(
                    raw_text=sentence,
                    min_years=min_years,
                    max_years=None,
                    is_required=_is_required_sentence(sentence),
                    evidence=sentence,
                    experience_context=_experience_context(sentence),
                    accepts_internship=bool(INTERNSHIP_PATTERN.search(sentence)),
                )
            )
    return requirements


def _experience_context(sentence: str) -> ExperienceContext:
    lower = sentence.lower()
    if any(kw in lower for kw in ("machine learning", "data science", "software development", "relevant")):
        if "professional" in lower or "work" in lower:
            return ExperienceContext.PROFESSIONAL
        return ExperienceContext.RELEVANT
    if "professional" in lower or "industry" in lower:
        return ExperienceContext.PROFESSIONAL
    return ExperienceContext.GENERAL


def _is_required_sentence(sentence: str) -> bool:
    lower = sentence.lower()
    if any(p in lower for p in ("preferred", "nice to have", "plus", "desirable")):
        return False
    return True


def extract_education_requirements(text: str, source_section: str, config: JobParsingConfig | None = None) -> list[EducationRequirement]:
    cfg = config or get_settings().job_parsing
    requirements: list[EducationRequirement] = []

    for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        lower = sentence.lower()
        degree_level = _match_degree_level(lower, cfg)
        inline = DEGREE_INLINE_PATTERN.search(sentence)

        if not degree_level and not inline:
            continue

        req_type = SkillRequirementType.PREFERRED if any(
            p in lower for p in cfg.preferred_phrases
        ) else SkillRequirementType.REQUIRED

        field = inline.group("field").strip() if inline and inline.group("field") else None
        level = degree_level or (inline.group("level") if inline else None)

        requirements.append(
            EducationRequirement(
                raw_text=sentence,
                degree_level=level,
                field=field,
                fields=[field] if field else [],
                requirement_type=req_type,
                evidence=sentence,
            )
        )
    return requirements


def _match_degree_level(text: str, config: JobParsingConfig) -> str | None:
    for level, aliases in config.degree_aliases.items():
        for alias in aliases:
            if alias.lower() in text:
                return level
    return None


def extract_domains(text: str, config: JobParsingConfig | None = None) -> list[str]:
    cfg = config or get_settings().job_parsing
    found: list[str] = []
    lower = text.lower()
    for domain in cfg.domains:
        if domain.lower() in lower:
            found.append(domain.title())
    return list(dict.fromkeys(found))


def merge_skill_requirements(
    requirements: list[SkillRequirement],
) -> tuple[list[SkillRequirement], list[SkillRequirement], list[SkillRequirement], list[SkillRequirement]]:
    """Split and dedupe skills; required > preferred > mentioned."""
    required: dict[str, SkillRequirement] = {}
    preferred: dict[str, SkillRequirement] = {}
    mentioned: dict[str, SkillRequirement] = {}
    soft: dict[str, SkillRequirement] = {}

    for req in requirements:
        key = (req.normalized_skill or req.raw_skill).lower()

        if req.category == SkillCategory.SOFT:
            soft[key] = req
            continue

        if req.requirement_type == SkillRequirementType.REQUIRED:
            preferred.pop(key, None)
            mentioned.pop(key, None)
            required[key] = req
        elif req.requirement_type == SkillRequirementType.PREFERRED:
            if key not in required:
                mentioned.pop(key, None)
                preferred[key] = req
        elif req.requirement_type == SkillRequirementType.MENTIONED:
            if key not in required and key not in preferred:
                mentioned[key] = req

    return (
        list(required.values()),
        list(preferred.values()),
        list(mentioned.values()),
        list(soft.values()),
    )
