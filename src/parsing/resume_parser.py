"""Heuristic resume parser — Document to ParsedResume."""

from __future__ import annotations

import logging
from typing import Optional

from src.models.document import Document
from src.models.resume import ParsedResume, ResumeSectionType
from src.parsing.base import ResumeParser
from src.parsing.contact_extractor import extract_contact_info
from src.extraction.skill_normalizer import normalize_skills_list
from src.parsing.entry_extractors import (
    estimate_years_experience,
    extract_certification_entries,
    extract_education_entries,
    extract_experience_entries,
    extract_project_entries,
    extract_skills,
    extract_achievements,
    extract_publications,
    extract_awards,
    extract_languages,
    extract_soft_skills,
    extract_responsibilities,
    detect_employment_location_workmode_notice,
)
from src.parsing.quality import assess_parsing_quality
from src.parsing.section_splitter import section_content_by_type, split_into_sections
from src.utils.config import ParsingConfig, get_settings

logger = logging.getLogger(__name__)


class HeuristicResumeParser(ResumeParser):
    """
    Rule-based resume parser tolerant of varied layouts.

    Consumes a format-agnostic Document from the ingestion layer.
    """

    def __init__(self, config: Optional[ParsingConfig] = None) -> None:
        self._config = config or get_settings().parsing

    def parse(self, document: Document) -> ParsedResume:
        text = document.extracted_text.strip()
        warnings: list[str] = list(document.extraction_warnings)

        if not text:
            warnings.append("Document contains no text to parse.")
            return ParsedResume(
                raw_text="",
                parsing_quality="low",
                parsing_warnings=warnings,
            )

        header_text, sections = split_into_sections(text, self._config)
        contact = extract_contact_info(header_text)

        if not sections:
            warnings.append("No section headings detected; treating entire document as unstructured text.")

        summary = section_content_by_type(sections, ResumeSectionType.SUMMARY) or None
        skills_content = section_content_by_type(sections, ResumeSectionType.SKILLS)
        experience_content = section_content_by_type(sections, ResumeSectionType.EXPERIENCE)
        education_content = section_content_by_type(sections, ResumeSectionType.EDUCATION)
        projects_content = section_content_by_type(sections, ResumeSectionType.PROJECTS)
        certifications_content = section_content_by_type(sections, ResumeSectionType.CERTIFICATIONS)

        skills = extract_skills(skills_content) if skills_content else []
        skills_normalized = normalize_skills_list(skills)
        if not skills and not sections:
            warnings.append("No skills section detected.")

        experience = extract_experience_entries(experience_content) if experience_content else []
        education = extract_education_entries(education_content) if education_content else []
        projects = extract_project_entries(projects_content) if projects_content else []
        certifications = (
            extract_certification_entries(certifications_content) if certifications_content else []
        )

        years_experience = estimate_years_experience(experience)

        # Advanced extraction
        achievements_content = section_content_by_type(sections, ResumeSectionType.ACHIEVEMENTS)
        publications_content = section_content_by_type(sections, ResumeSectionType.PUBLICATIONS)
        languages_content = section_content_by_type(sections, ResumeSectionType.LANGUAGES)

        achievements = extract_achievements(achievements_content) if achievements_content else []
        publications = extract_publications(publications_content) if publications_content else []
        languages = extract_languages(languages_content) if languages_content else []

        # Soft skills are often listed in skills section; fall back to summary if missing
        soft_skills = []
        if skills_content:
            soft_skills = extract_soft_skills(skills_content)
        if not soft_skills and summary:
            soft_skills = extract_soft_skills(summary)

        # Responsibilities from experience blocks
        responsibilities = []
        if experience_content:
            responsibilities = extract_responsibilities(experience_content)

        # Domains: exact phrase matches from canonical domains list (no inference)
        settings = get_settings()
        domains: list[str] = []
        text_lower = text.lower()
        for d in settings.skills.domains:
            if d.lower() in text_lower:
                domains.append(d)

        # Project technologies: consolidate project entry tech lists
        project_technologies: list[str] = []
        for p in projects:
            for t in getattr(p, "technologies", []) or []:
                if t and t not in project_technologies:
                    project_technologies.append(t)

        # Employment/location/work_mode/notice heuristics
        employment_type, location, work_mode, notice_period = detect_employment_location_workmode_notice(
            header_text + "\n" + text
        )

        parsed = ParsedResume(
            name=contact.name,
            email=contact.email,
            phone=contact.phone,
            summary=summary,
            skills=skills,
            skills_normalized=skills_normalized,
            education=education,
            experience=experience,
            projects=projects,
            certifications=certifications,
            sections=sections,
            years_experience=years_experience,
            raw_text=text,
            parsing_warnings=warnings,
            achievements=achievements,
            publications=publications,
            languages=languages,
            soft_skills=soft_skills,
            domains=domains,
            responsibilities=responsibilities,
            employment_type=employment_type,
            location=location,
            work_mode=work_mode,
            notice_period=notice_period,
            project_technologies=project_technologies,
        )

        parsed.parsing_quality = assess_parsing_quality(parsed, self._config)

        logger.info(
            "Parsed resume: sections=%d skills=%d experience=%d quality=%s",
            len(sections),
            len(skills),
            len(experience),
            parsed.parsing_quality,
        )
        return parsed
