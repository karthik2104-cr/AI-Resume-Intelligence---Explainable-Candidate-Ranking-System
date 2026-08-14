"""Integration: entity profiles and skill gap without matching engine."""

from src.extraction.entity_extractor import EntityExtractor
from src.matching.skill_gap import compute_skill_gap
from src.models.document import Document, DocumentSourceType
from src.parsing.job_parser import parse_job_description
from src.parsing.resume_parser import HeuristicResumeParser
from tests.fixtures.jd_samples import ML_ENGINEER_JD
from tests.fixtures.resume_samples import SAMPLE_RESUME


def test_resume_and_job_profiles_compatible():
    resume = HeuristicResumeParser().parse(
        Document(source_type=DocumentSourceType.TXT, raw_text=SAMPLE_RESUME)
    )
    job = parse_job_description(ML_ENGINEER_JD)

    extractor = EntityExtractor()
    candidate_profile = extractor.build_candidate_profile(resume)
    job_profile = extractor.build_job_profile(job)

    assert candidate_profile.technical_skills
    assert job_profile.required_technical_skills or job_profile.preferred_technical_skills

    candidate_norm = {e.normalized_value.lower() for e in candidate_profile.technical_skills}
    job_required = {s.normalized_skill.lower() for s in job.required_skills if s.normalized_skill}
    assert candidate_norm & job_required  # shared normalized concepts exist

    gap = compute_skill_gap(resume, job, candidate_profile, job_profile)
    assert gap.required_total >= 1
    assert 0 <= gap.required_skill_coverage <= 100
