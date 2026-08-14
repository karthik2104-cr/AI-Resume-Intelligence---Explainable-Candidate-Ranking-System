"""Unified entity extraction from parsed resume and job objects."""

from __future__ import annotations

from src.extraction.skill_normalizer import (
    dedupe_skill_matches,
    entity_type_for_category,
    extract_sentence_evidence,
    find_skills_in_text,
    get_skill_category,
    normalize_skill,
    normalize_skills_list,
)
from src.models.entity import (
    CandidateProfile,
    Entity,
    EntityProfile,
    EntitySource,
    EntityType,
    JobProfile,
)
from src.models.job import ParsedJobDescription, SkillRequirementType
from src.models.resume import ParsedResume
from src.utils.config import get_settings


class EntityExtractor:
    """Extract normalized entities from structured parse outputs — not raw files."""

    def extract_from_resume(self, resume: ParsedResume) -> EntityProfile:
        entities: list[Entity] = []
        soft_skills_cfg = {s.lower() for s in get_settings().skills.soft_skills}
        domains_cfg = get_settings().skills.domains

        # Normalize explicit skills list
        normalized_from_list = normalize_skills_list(resume.skills)
        for raw, norm in zip(resume.skills, normalized_from_list):
            entities.append(
                self._skill_entity(
                    raw_text=raw,
                    normalized=norm,
                    source="resume.skills",
                    evidence=f"Listed in skills section: {raw}",
                    soft_set=soft_skills_cfg,
                )
            )

        # Extract from experience, projects, summary using vocabulary
        text_sources = [
            ("resume.experience", "\n".join(e.raw_text for e in resume.experience)),
            ("resume.projects", "\n".join(p.raw_text for p in resume.projects)),
            ("resume.summary", resume.summary or ""),
        ]
        for source, text in text_sources:
            if not text.strip():
                continue
            for match in dedupe_skill_matches(find_skills_in_text(text)):
                evidence = extract_sentence_evidence(text, match.start, match.end)
                entities.append(
                    self._skill_entity(
                        raw_text=match.matched_text,
                        normalized=match.canonical,
                        source=source,
                        evidence=evidence,
                        soft_set=soft_skills_cfg,
                    )
                )

        domains = self._extract_domains(resume.raw_text, "resume", domains_cfg)
        entities.extend(domains)
        soft_entities = self._extract_soft_skills(resume.raw_text, "resume", soft_skills_cfg)

        return self._build_profile(EntitySource.RESUME, entities, soft_entities, domains)

    def extract_from_job(self, job: ParsedJobDescription) -> EntityProfile:
        entities: list[Entity] = []
        soft_skills_cfg = {s.lower() for s in get_settings().skills.soft_skills}
        domains_cfg = get_settings().skills.domains

        for bucket, source_label in (
            (job.required_skills, "job.required"),
            (job.preferred_skills, "job.preferred"),
            (job.mentioned_skills, "job.mentioned"),
        ):
            for req in bucket:
                norm = req.normalized_skill or normalize_skill(req.raw_skill)
                entities.append(
                    self._skill_entity(
                        raw_text=req.raw_skill,
                        normalized=norm,
                        source=source_label,
                        evidence=req.evidence,
                        soft_set=soft_skills_cfg,
                    )
                )

        domains = self._extract_domains(job.raw_text, "job", domains_cfg)
        for domain in domains:
            entities.append(domain)

        soft_entities = self._extract_soft_skills(job.raw_text, "job", soft_skills_cfg)
        for soft in job.soft_skills:
            norm = soft.normalized_skill or soft.raw_skill
            soft_entities.append(
                Entity(
                    raw_text=soft.raw_skill,  # raw_skill is the surface text for SkillRequirement
                    normalized_value=norm,
                    entity_type=EntityType.SOFT_SKILL,
                    category="Soft Skills",
                    source="job.soft_skills",
                    evidence=soft.evidence,
                )
            )

        return self._build_profile(EntitySource.JOB, entities, soft_entities, domains)

    def build_candidate_profile(self, resume: ParsedResume) -> CandidateProfile:
        """Build a lightweight normalized candidate profile from a parsed resume."""
        entity_profile = self.extract_from_resume(resume)
        return CandidateProfile(
            technical_skills=entity_profile.technical_skills,
            soft_skills=entity_profile.soft_skills,
            domains=entity_profile.domains,
            education=resume.education,
            experience=resume.experience,
            projects=resume.projects,
            certifications=resume.certifications,
            years_experience=resume.years_experience,
        )

    def build_job_profile(self, job: ParsedJobDescription) -> JobProfile:
        """Build a lightweight normalized job profile from a parsed JD."""
        entity_profile = self.extract_from_job(job)
        required: list[Entity] = []
        preferred: list[Entity] = []
        mentioned: list[Entity] = []

        for entity in entity_profile.technical_skills:
            if entity.source.startswith("job.required"):
                required.append(entity)
            elif entity.source.startswith("job.preferred"):
                preferred.append(entity)
            elif entity.source.startswith("job.mentioned"):
                mentioned.append(entity)

        return JobProfile(
            required_technical_skills=required,
            preferred_technical_skills=preferred,
            mentioned_skills=mentioned,
            soft_skills=entity_profile.soft_skills,
            domains=entity_profile.domains,
            experience_requirements=job.experience_requirements,
            education_requirements=job.education_requirements,
            seniority_level=job.seniority_level,
        )

    def _skill_entity(
        self,
        raw_text: str,
        normalized: str,
        source: str,
        evidence: str,
        soft_set: set[str],
    ) -> Entity:
        if normalized.lower() in soft_set or raw_text.lower() in soft_set:
            return Entity(
                raw_text=raw_text,
                normalized_value=normalized.title(),
                entity_type=EntityType.SOFT_SKILL,
                category="Soft Skills",
                source=source,
                evidence=evidence,
            )
        category = get_skill_category(normalized)
        type_slug = entity_type_for_category(category)
        return Entity(
            raw_text=raw_text,
            normalized_value=normalized,
            entity_type=EntityType(type_slug),
            category=category,
            source=source,
            evidence=evidence,
        )

    def _extract_domains(self, text: str, prefix: str, domains_cfg: list[str]) -> list[Entity]:
        import re

        found: list[Entity] = []
        lower = text.lower()
        for domain in domains_cfg:
            pattern = re.compile(rf"\b{re.escape(domain.lower())}\b", re.IGNORECASE)
            if pattern.search(lower):
                for line in text.splitlines():
                    if domain.lower() in line.lower():
                        found.append(
                            Entity(
                                raw_text=domain,
                                normalized_value=domain.title(),
                                entity_type=EntityType.DOMAIN,
                                category="Domain",
                                source=f"{prefix}.domain",
                                evidence=line.strip(),
                            )
                        )
                        break
        return found

    def _extract_soft_skills(self, text: str, prefix: str, soft_set: set[str]) -> list[Entity]:
        import re

        found: list[Entity] = []
        lower = text.lower()
        for skill in soft_set:
            if re.search(rf"\b{re.escape(skill)}\b", lower):
                for line in text.splitlines():
                    if skill in line.lower():
                        found.append(
                            Entity(
                                raw_text=skill,
                                normalized_value=skill.title(),
                                entity_type=EntityType.SOFT_SKILL,
                                category="Soft Skills",
                                source=f"{prefix}.soft_skill",
                                evidence=line.strip(),
                            )
                        )
                        break
        return found

    def _build_profile(
        self,
        source_type: EntitySource,
        entities: list[Entity],
        soft_entities: list[Entity],
        domains: list[Entity],
    ) -> EntityProfile:
        technical: dict[str, Entity] = {}
        soft: dict[str, Entity] = {}
        domain_map: dict[str, Entity] = {}

        for entity in entities:
            if entity.entity_type == EntityType.SOFT_SKILL:
                soft[entity.normalized_value.lower()] = entity
            elif entity.entity_type == EntityType.DOMAIN:
                domain_map[entity.normalized_value.lower()] = entity
            else:
                technical[entity.normalized_value.lower()] = entity

        for entity in soft_entities:
            soft[entity.normalized_value.lower()] = entity
        for entity in domains:
            domain_map[entity.normalized_value.lower()] = entity

        all_entities = list(technical.values()) + list(soft.values()) + list(domain_map.values())
        return EntityProfile(
            source_type=source_type,
            entities=all_entities,
            technical_skills=list(technical.values()),
            soft_skills=list(soft.values()),
            domains=list(domain_map.values()),
        )
