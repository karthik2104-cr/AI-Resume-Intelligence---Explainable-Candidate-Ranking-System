"""Heuristic job description parser."""

from __future__ import annotations

import logging
from typing import Optional

from src.models.job import (
    JobDescription,
    JobSectionType,
    ParsedJobDescription,
    SkillCategory,
)
from src.parsing.jd_extractors import (
    extract_domains,
    extract_education_requirements,
    extract_employment_type,
    extract_experience_requirements,
    extract_job_title,
    extract_location,
    extract_responsibilities,
    extract_skills_from_text,
    extract_soft_skills,
    extract_work_mode,
    merge_skill_requirements,
)
from src.parsing.jd_quality import assess_jd_parsing_quality
from src.parsing.jd_section_splitter import section_content_by_type, split_jd_into_sections
from src.parsing.job_base import JobDescriptionParser
from src.utils.config import JobParsingConfig, get_settings

logger = logging.getLogger(__name__)

SKILL_REQUIREMENT_SECTIONS = {
    JobSectionType.REQUIREMENTS,
    JobSectionType.REQUIRED_QUALIFICATIONS,
    JobSectionType.PREFERRED_QUALIFICATIONS,
    JobSectionType.QUALIFICATIONS,
}

NEUTRAL_MENTION_SECTIONS = {
    JobSectionType.RESPONSIBILITIES,
    JobSectionType.DESCRIPTION,
    JobSectionType.ABOUT,
}


class HeuristicJobDescriptionParser(JobDescriptionParser):
    """Deterministic, evidence-preserving job description parser."""

    def __init__(self, config: Optional[JobParsingConfig] = None) -> None:
        self._config = config or get_settings().job_parsing

    def parse(self, job: JobDescription) -> ParsedJobDescription:
        text = job.raw_text.strip()
        warnings: list[str] = []

        if not text:
            warnings.append("Job description text is empty.")
            parsed = ParsedJobDescription(
                title=job.title,
                raw_text="",
                parsing_warnings=warnings,
            )
            parsed.parsing_quality = assess_jd_parsing_quality(parsed)
            return parsed

        header_text, sections = split_jd_into_sections(text, self._config)
        if not sections:
            warnings.append("No JD section headings detected; parsing from full text.")

        job_title = extract_job_title(header_text or text[:500], text, self._config)
        title = job.title or (job_title.normalized_title if job_title else None)
        seniority = job_title.seniority_level if job_title else None

        all_skill_reqs: list[SkillRequirement] = []
        responsibilities = []
        experience_reqs = []
        education_reqs = []

        if sections:
            for section in sections:
                source = section.title
                if section.section_type == JobSectionType.RESPONSIBILITIES:
                    responsibilities.extend(extract_responsibilities(section.content, source))
                    all_skill_reqs.extend(
                        extract_skills_from_text(
                            section.content, section.section_type, source, self._config
                        )
                    )
                if section.section_type in SKILL_REQUIREMENT_SECTIONS:
                    all_skill_reqs.extend(
                        extract_skills_from_text(section.content, section.section_type, source, self._config)
                    )
                    all_skill_reqs.extend(extract_soft_skills(section.content, source, self._config))
                if section.section_type in NEUTRAL_MENTION_SECTIONS - {JobSectionType.RESPONSIBILITIES}:
                    all_skill_reqs.extend(
                        extract_skills_from_text(section.content, section.section_type, source, self._config)
                    )
                if section.section_type in {
                    JobSectionType.EXPERIENCE,
                    JobSectionType.REQUIREMENTS,
                    JobSectionType.REQUIRED_QUALIFICATIONS,
                    JobSectionType.QUALIFICATIONS,
                }:
                    experience_reqs.extend(extract_experience_requirements(section.content, source))
                if section.section_type in {
                    JobSectionType.EDUCATION,
                    JobSectionType.REQUIREMENTS,
                    JobSectionType.QUALIFICATIONS,
                    JobSectionType.REQUIRED_QUALIFICATIONS,
                }:
                    education_reqs.extend(extract_education_requirements(section.content, source, self._config))
        else:
            all_skill_reqs.extend(extract_skills_from_text(text, JobSectionType.REQUIREMENTS, "full_text", self._config))
            all_skill_reqs.extend(extract_soft_skills(text, "full_text", self._config))
            responsibilities.extend(extract_responsibilities(text, "full_text"))
            experience_reqs.extend(extract_experience_requirements(text, "full_text"))
            education_reqs.extend(extract_education_requirements(text, "full_text", self._config))

        required, preferred, mentioned, soft = merge_skill_requirements(all_skill_reqs)
        tools = [
            s.normalized_skill or s.raw_skill
            for s in required + preferred
            if s.category != SkillCategory.SOFT
        ]
        domains = extract_domains(text, self._config)

        parsed = ParsedJobDescription(
            title=title,
            job_title=job_title,
            seniority_level=seniority,
            employment_type=extract_employment_type(text, self._config),
            location=extract_location(text),
            work_mode=extract_work_mode(text, self._config),
            responsibilities=responsibilities,
            required_skills=required,
            preferred_skills=preferred,
            mentioned_skills=mentioned,
            soft_skills=soft,
            education_requirements=education_reqs,
            experience_requirements=experience_reqs,
            tools_technologies=list(dict.fromkeys(tools)),
            domain_requirements=domains,
            sections=sections,
            raw_text=text,
            parsing_warnings=warnings,
        )
        parsed.parsing_quality = assess_jd_parsing_quality(parsed)

        logger.info(
            "Parsed JD: title=%s required_skills=%d preferred_skills=%d quality=%s",
            title,
            len(required),
            len(preferred),
            parsed.parsing_quality.level,
        )
        return parsed


def parse_job_description(
    text: str,
    title: str | None = None,
    config: Optional[JobParsingConfig] = None,
) -> ParsedJobDescription:
    """Parse raw job description text into a structured ParsedJobDescription."""
    parser = HeuristicJobDescriptionParser(config=config)
    return parser.parse_text(text, title=title)
