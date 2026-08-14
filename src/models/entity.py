"""Unified entity representation for resumes and job descriptions."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from src.models.job import EducationRequirement, ExperienceRequirement
from src.models.resume import (
    CertificationEntry,
    EducationEntry,
    ExperienceEntry,
    ProjectEntry,
)


class EntityType(str, Enum):
    TECHNICAL_SKILL = "technical_skill"
    SOFT_SKILL = "soft_skill"
    DOMAIN = "domain"
    TECHNOLOGY = "technology"
    PROGRAMMING_LANGUAGE = "programming_language"
    FRAMEWORK = "framework"
    LIBRARY = "library"
    DATABASE = "database"
    CLOUD = "cloud"
    TOOL = "tool"


class EntitySource(str, Enum):
    RESUME = "resume"
    JOB = "job"


class Entity(BaseModel):
    """Normalized entity with evidence for explainability."""

    raw_text: str
    normalized_value: str
    entity_type: EntityType
    category: Optional[str] = None
    source: str = ""
    evidence: str = ""
    quality_note: Optional[str] = None


class EntityProfile(BaseModel):
    """Collection of normalized entities from a resume or job description."""

    source_type: EntitySource
    entities: list[Entity] = Field(default_factory=list)
    technical_skills: list[Entity] = Field(default_factory=list)
    soft_skills: list[Entity] = Field(default_factory=list)
    domains: list[Entity] = Field(default_factory=list)

    @property
    def normalized_skill_names(self) -> list[str]:
        """All normalized technical skill values (deduplicated)."""
        seen: set[str] = set()
        names: list[str] = []
        for entity in self.technical_skills:
            key = entity.normalized_value.lower()
            if key not in seen:
                seen.add(key)
                names.append(entity.normalized_value)
        return names


class CandidateProfile(BaseModel):
    """Lightweight normalized view of a parsed resume for matching."""

    technical_skills: list[Entity] = Field(default_factory=list)
    soft_skills: list[Entity] = Field(default_factory=list)
    domains: list[Entity] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
    certifications: list[CertificationEntry] = Field(default_factory=list)
    years_experience: Optional[float] = None

    @property
    def normalized_skill_names(self) -> list[str]:
        seen: set[str] = set()
        names: list[str] = []
        for entity in self.technical_skills:
            key = entity.normalized_value.lower()
            if key not in seen:
                seen.add(key)
                names.append(entity.normalized_value)
        return names


class JobProfile(BaseModel):
    """Lightweight normalized view of a parsed job description for matching."""

    required_technical_skills: list[Entity] = Field(default_factory=list)
    preferred_technical_skills: list[Entity] = Field(default_factory=list)
    mentioned_skills: list[Entity] = Field(default_factory=list)
    soft_skills: list[Entity] = Field(default_factory=list)
    domains: list[Entity] = Field(default_factory=list)
    experience_requirements: list[ExperienceRequirement] = Field(default_factory=list)
    education_requirements: list[EducationRequirement] = Field(default_factory=list)
    seniority_level: Optional[str] = None
