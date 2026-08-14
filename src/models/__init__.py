"""Domain models for V2 resume intelligence system."""

from src.models.document import Document, DocumentPage, DocumentSourceType
from src.models.job import (
    EducationRequirement,
    ExperienceRequirement,
    JobDescription,
    JobParsingQuality,
    JobSection,
    JobSectionType,
    JobTitle,
    ParsedJobDescription,
    ResponsibilityEntry,
    SkillCategory,
    SkillRequirement,
    SkillRequirementType,
)
from src.models.matching import (
    ComponentScores,
    MatchExplanation,
    MatchRequest,
    MatchResult,
    SkillCoverage,
)
from src.models.ranking import CandidateRankEntry, RankingRequest, RankingResult
from src.models.resume import (
    CertificationEntry,
    EducationEntry,
    ExperienceEntry,
    ParsedResume,
    ProjectEntry,
    ResumeSection,
    ResumeSectionType,
)

__all__ = [
    "CertificationEntry",
    "ComponentScores",
    "Document",
    "DocumentPage",
    "DocumentSourceType",
    "EducationEntry",
    "EducationRequirement",
    "ExperienceEntry",
    "ExperienceRequirement",
    "JobDescription",
    "MatchExplanation",
    "MatchRequest",
    "MatchResult",
    "ParsedJobDescription",
    "ParsedResume",
    "ProjectEntry",
    "CandidateRankEntry",
    "RankingRequest",
    "RankingResult",
    "ResumeSection",
    "ResumeSectionType",
    "SkillCoverage",
    "SkillRequirement",
    "SkillRequirementType",
]
