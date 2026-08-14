"""Deterministic skill gap analysis between candidate and job profiles."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.extraction.entity_extractor import EntityExtractor
from src.models.entity import CandidateProfile, Entity, EntityProfile, JobProfile
from src.models.job import ParsedJobDescription, SkillRequirement
from src.models.resume import ParsedResume


class SkillGapEntry(BaseModel):
    skill: str
    candidate_evidence: str | None = None
    job_evidence: str | None = None


class SkillGapResult(BaseModel):
    matched_required: list[SkillGapEntry] = Field(default_factory=list)
    missing_required: list[SkillGapEntry] = Field(default_factory=list)
    matched_preferred: list[SkillGapEntry] = Field(default_factory=list)
    missing_preferred: list[SkillGapEntry] = Field(default_factory=list)
    mentioned_skills: list[SkillGapEntry] = Field(default_factory=list)
    additional_candidate_skills: list[SkillGapEntry] = Field(default_factory=list)
    required_total: int = 0
    required_matched: int = 0
    required_missing: int = 0
    preferred_total: int = 0
    preferred_matched: int = 0
    preferred_missing: int = 0
    required_skill_coverage: float = 0.0
    preferred_skill_coverage: float = 0.0


def _normalized_key(skill: str) -> str:
    return skill.strip().lower()


def _job_skill_map(requirements: list[SkillRequirement]) -> dict[str, SkillRequirement]:
    mapping: dict[str, SkillRequirement] = {}
    for req in requirements:
        key = _normalized_key(req.normalized_skill or req.raw_skill)
        mapping[key] = req
    return mapping


def _candidate_skill_map(entities: list[Entity]) -> dict[str, str]:
    """Map normalized skill -> evidence from candidate technical skills."""
    mapping: dict[str, str] = {}
    for entity in entities:
        mapping[_normalized_key(entity.normalized_value)] = entity.evidence
    return mapping


def _resolve_candidate_entities(
    resume: ParsedResume,
    candidate_profile: EntityProfile | CandidateProfile | None,
    extractor: EntityExtractor,
) -> list[Entity]:
    if candidate_profile is not None:
        return candidate_profile.technical_skills
    return extractor.extract_from_resume(resume).technical_skills


def compute_skill_gap(
    resume: ParsedResume,
    job: ParsedJobDescription,
    candidate_profile: EntityProfile | CandidateProfile | None = None,
    job_profile: EntityProfile | JobProfile | None = None,
) -> SkillGapResult:
    """
    Compute set-based skill coverage between candidate and job.

    This is NOT a final candidate match score — only skill coverage analysis.
    """
    extractor = EntityExtractor()
    candidate_entities = _resolve_candidate_entities(resume, candidate_profile, extractor)
    candidate_skills = set(_candidate_skill_map(candidate_entities).keys())
    candidate_evidence = _candidate_skill_map(candidate_entities)

    required_map = _job_skill_map(job.required_skills)
    preferred_map = _job_skill_map(job.preferred_skills)
    mentioned_map = _job_skill_map(job.mentioned_skills)

    required_keys = set(required_map.keys())
    preferred_keys = set(preferred_map.keys())
    mentioned_keys = set(mentioned_map.keys())
    job_all_keys = required_keys | preferred_keys | mentioned_keys

    matched_required: list[SkillGapEntry] = []
    missing_required: list[SkillGapEntry] = []
    for key in sorted(required_keys):
        req = required_map[key]
        skill_name = req.normalized_skill or req.raw_skill
        if key in candidate_skills:
            matched_required.append(
                SkillGapEntry(
                    skill=skill_name,
                    candidate_evidence=candidate_evidence.get(key),
                    job_evidence=req.evidence,
                )
            )
        else:
            missing_required.append(
                SkillGapEntry(skill=skill_name, job_evidence=req.evidence)
            )

    matched_preferred: list[SkillGapEntry] = []
    missing_preferred: list[SkillGapEntry] = []
    for key in sorted(preferred_keys):
        req = preferred_map[key]
        skill_name = req.normalized_skill or req.raw_skill
        if key in candidate_skills:
            matched_preferred.append(
                SkillGapEntry(
                    skill=skill_name,
                    candidate_evidence=candidate_evidence.get(key),
                    job_evidence=req.evidence,
                )
            )
        else:
            missing_preferred.append(
                SkillGapEntry(skill=skill_name, job_evidence=req.evidence)
            )

    mentioned: list[SkillGapEntry] = []
    for key in sorted(mentioned_keys):
        req = mentioned_map[key]
        skill_name = req.normalized_skill or req.raw_skill
        mentioned.append(
            SkillGapEntry(
                skill=skill_name,
                candidate_evidence=candidate_evidence.get(key) if key in candidate_skills else None,
                job_evidence=req.evidence,
            )
        )

    additional: list[SkillGapEntry] = []
    for key in sorted(candidate_skills - job_all_keys):
        entity = next(
            (e for e in candidate_entities if _normalized_key(e.normalized_value) == key),
            None,
        )
        if entity:
            additional.append(
                SkillGapEntry(
                    skill=entity.normalized_value,
                    candidate_evidence=entity.evidence,
                )
            )

    req_total = len(required_keys)
    req_matched = len(matched_required)
    pref_total = len(preferred_keys)
    pref_matched = len(matched_preferred)

    return SkillGapResult(
        matched_required=matched_required,
        missing_required=missing_required,
        matched_preferred=matched_preferred,
        missing_preferred=missing_preferred,
        mentioned_skills=mentioned,
        additional_candidate_skills=additional,
        required_total=req_total,
        required_matched=req_matched,
        required_missing=len(missing_required),
        preferred_total=pref_total,
        preferred_matched=pref_matched,
        preferred_missing=len(missing_preferred),
        required_skill_coverage=(req_matched / req_total * 100) if req_total else 0.0,
        preferred_skill_coverage=(pref_matched / pref_total * 100) if pref_total else 0.0,
    )
