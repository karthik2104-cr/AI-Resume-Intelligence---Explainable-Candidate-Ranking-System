"""Structured job description models."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SkillRequirementType(str, Enum):
    REQUIRED = "required"
    PREFERRED = "preferred"
    MENTIONED = "mentioned"


class SkillCategory(str, Enum):
    TECHNICAL = "technical"
    SOFT = "soft"
    TOOL = "tool"
    DOMAIN = "domain"


class ExperienceContext(str, Enum):
    PROFESSIONAL = "professional"
    RELEVANT = "relevant"
    DOMAIN = "domain"
    GENERAL = "general"


class JobSectionType(str, Enum):
    ABOUT = "about"
    DESCRIPTION = "description"
    RESPONSIBILITIES = "responsibilities"
    REQUIREMENTS = "requirements"
    REQUIRED_QUALIFICATIONS = "required_qualifications"
    PREFERRED_QUALIFICATIONS = "preferred_qualifications"
    QUALIFICATIONS = "qualifications"
    EDUCATION = "education"
    EXPERIENCE = "experience"
    BENEFITS = "benefits"
    OTHER = "other"


class JobSection(BaseModel):
    section_type: JobSectionType
    title: str
    content: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None


class JobTitle(BaseModel):
    raw_title: str
    normalized_title: str
    seniority_level: Optional[str] = None
    evidence: str = ""


class SkillRequirement(BaseModel):
    raw_skill: str
    normalized_skill: Optional[str] = None
    category: SkillCategory = SkillCategory.TECHNICAL
    requirement_type: SkillRequirementType = SkillRequirementType.REQUIRED
    evidence: str = ""
    source_section: str = ""


class ExperienceRequirement(BaseModel):
    raw_text: str
    min_years: Optional[float] = None
    max_years: Optional[float] = None
    is_required: bool = True
    evidence: str = ""
    experience_context: ExperienceContext = ExperienceContext.PROFESSIONAL
    accepts_internship: bool = False
    is_entry_level: bool = False


class EducationRequirement(BaseModel):
    raw_text: str
    degree_level: Optional[str] = None
    field: Optional[str] = None
    fields: list[str] = Field(default_factory=list)
    requirement_type: SkillRequirementType = SkillRequirementType.REQUIRED
    evidence: str = ""


class ResponsibilityEntry(BaseModel):
    text: str
    source_section: str = ""


class JobRequirement(BaseModel):
    """Generic requirement with evidence for explainability."""

    text: str
    requirement_type: SkillRequirementType
    source_section: str = ""
    evidence: str = ""


class JobParsingQuality(BaseModel):
    score: int = 0
    level: str = "low"  # high | medium | low
    warnings: list[str] = Field(default_factory=list)


class JobDescription(BaseModel):
    """Raw job description input."""

    title: Optional[str] = None
    raw_text: str
    source: Optional[str] = None


class ParsedJobDescription(BaseModel):
    """Structured job description after parsing."""

    title: Optional[str] = None
    job_title: Optional[JobTitle] = None
    seniority_level: Optional[str] = None
    employment_type: Optional[str] = None
    location: Optional[str] = None
    work_mode: Optional[str] = None
    responsibilities: list[ResponsibilityEntry] = Field(default_factory=list)
    required_skills: list[SkillRequirement] = Field(default_factory=list)
    preferred_skills: list[SkillRequirement] = Field(default_factory=list)
    mentioned_skills: list[SkillRequirement] = Field(default_factory=list)
    soft_skills: list[SkillRequirement] = Field(default_factory=list)
    education_requirements: list[EducationRequirement] = Field(default_factory=list)
    experience_requirements: list[ExperienceRequirement] = Field(default_factory=list)
    tools_technologies: list[str] = Field(default_factory=list)
    domain_requirements: list[str] = Field(default_factory=list)
    sections: list[JobSection] = Field(default_factory=list)
    raw_text: str = ""
    parsing_warnings: list[str] = Field(default_factory=list)
    parsing_quality: JobParsingQuality = Field(default_factory=JobParsingQuality)

    @property
    def full_text_for_matching(self) -> str:
        parts = [self.raw_text, self.title or ""]
        parts.extend(r.text for r in self.responsibilities)
        parts.extend(s.raw_skill for s in self.required_skills)
        parts.extend(s.raw_skill for s in self.preferred_skills)
        return " ".join(p for p in parts if p).strip()

    @property
    def all_required_skill_names(self) -> list[str]:
        return [
            s.normalized_skill or s.raw_skill
            for s in self.required_skills
        ]

    @property
    def all_preferred_skill_names(self) -> list[str]:
        return [
            s.normalized_skill or s.raw_skill
            for s in self.preferred_skills
        ]

    @property
    def responsibility_texts(self) -> list[str]:
        """Backward-compatible access to responsibility strings."""
        return [r.text for r in self.responsibilities]
