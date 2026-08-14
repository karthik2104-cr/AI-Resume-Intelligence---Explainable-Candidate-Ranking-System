"""Structured resume models."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ResumeSectionType(str, Enum):
    SUMMARY = "summary"
    EDUCATION = "education"
    EXPERIENCE = "experience"
    SKILLS = "skills"
    PROJECTS = "projects"
    CERTIFICATIONS = "certifications"
    ACHIEVEMENTS = "achievements"
    PUBLICATIONS = "publications"
    LANGUAGES = "languages"
    OTHER = "other"


class ResumeSection(BaseModel):
    section_type: ResumeSectionType
    title: str
    content: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None


class EducationEntry(BaseModel):
    institution: Optional[str] = None
    degree: Optional[str] = None
    specialization: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    raw_text: str = ""


class ExperienceEntry(BaseModel):
    organization: Optional[str] = None
    title: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: str = ""
    is_internship: bool = False
    raw_text: str = ""


class ProjectEntry(BaseModel):
    name: Optional[str] = None
    description: str = ""
    technologies: list[str] = Field(default_factory=list)
    raw_text: str = ""


class CertificationEntry(BaseModel):
    name: str
    issuer: Optional[str] = None
    date: Optional[str] = None
    raw_text: str = ""


class ParsedResume(BaseModel):
    """Structured resume after parsing and extraction."""

    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    summary: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    skills_normalized: list[str] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
    certifications: list[CertificationEntry] = Field(default_factory=list)
    sections: list[ResumeSection] = Field(default_factory=list)
    years_experience: Optional[float] = None
    raw_text: str = ""
    parsing_quality: str = "unknown"  # high | medium | low | unknown
    parsing_warnings: list[str] = Field(default_factory=list)

    # Phase 10 — advanced extraction fields (optional, backward compatible)
    achievements: list[str] = Field(default_factory=list)
    publications: list[str] = Field(default_factory=list)
    awards: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    employment_type: Optional[str] = None
    location: Optional[str] = None
    work_mode: Optional[str] = None
    notice_period: Optional[str] = None
    project_technologies: list[str] = Field(default_factory=list)

    @property
    def full_text_for_matching(self) -> str:
        """Concatenated text used for baseline/semantic matching."""
        parts = [self.raw_text]
        if self.summary:
            parts.append(self.summary)
        if self.skills:
            parts.append(" ".join(self.skills))
        return " ".join(p for p in parts if p).strip()
