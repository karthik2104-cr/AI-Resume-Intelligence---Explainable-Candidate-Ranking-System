"""Integration: ParsedResume + ParsedJobDescription compatibility."""

from src.parsing.job_parser import parse_job_description
from src.parsing.resume_parser import HeuristicResumeParser
from src.models.document import Document, DocumentSourceType
from tests.fixtures.jd_samples import ML_ENGINEER_JD
from tests.fixtures.resume_samples import SAMPLE_RESUME


def test_resume_and_jd_share_normalized_skills():
    resume = HeuristicResumeParser().parse(
        Document(source_type=DocumentSourceType.TXT, raw_text=SAMPLE_RESUME)
    )
    job = parse_job_description(ML_ENGINEER_JD)

    resume_skills = {s.lower() for s in resume.skills}
    required = {s.normalized_skill.lower() for s in job.required_skills if s.normalized_skill}

    overlap = resume_skills & required
    assert "python" in overlap or "Python".lower() in overlap

    assert resume.parsing_quality in {"high", "medium", "low"}
    assert job.parsing_quality.level in {"high", "medium", "low"}
    assert job.required_skills
    assert all(s.evidence for s in job.required_skills)
    assert all(s.normalized_skill for s in job.required_skills)
