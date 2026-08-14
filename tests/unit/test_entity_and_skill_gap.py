"""Phase 6 tests for entity extraction and skill gap analysis."""

from src.extraction.entity_extractor import EntityExtractor
from src.extraction.requirement_classifier import classify_requirement_type
from src.extraction.skill_normalizer import normalize_skill
from src.matching.skill_gap import compute_skill_gap
from src.models.document import Document, DocumentSourceType
from src.models.job import JobSectionType, SkillRequirementType
from src.parsing.job_parser import parse_job_description
from src.parsing.resume_parser import HeuristicResumeParser
from tests.fixtures.jd_samples import ML_ENGINEER_JD, REQUIRED_PREFERRED_JD
from tests.fixtures.resume_samples import SAMPLE_RESUME


class TestNormalizationConsistency:
    def test_resume_sklearn_matches_jd_scikit_learn(self):
        resume = HeuristicResumeParser().parse(
            Document(
                source_type=DocumentSourceType.TXT,
                raw_text="SKILLS\nsklearn, Python",
            )
        )
        job = parse_job_description("Requirements\nMust have scikit-learn experience.")
        gap = compute_skill_gap(resume, job)
        assert any(e.skill == "Scikit-learn" for e in gap.matched_required)

    def test_js_matches_javascript(self):
        assert normalize_skill("js") == normalize_skill("JavaScript")

    def test_java_not_javascript(self):
        resume = HeuristicResumeParser().parse(
            Document(source_type=DocumentSourceType.TXT, raw_text="SKILLS\nJava")
        )
        job = parse_job_description("Requirements\nMust have JavaScript.")
        gap = compute_skill_gap(resume, job)
        assert gap.required_matched == 0
        assert gap.missing_required


class TestRequirementSemantics:
    def test_required_signal(self):
        assert (
            classify_requirement_type("Must have AWS experience.", JobSectionType.RESPONSIBILITIES)
            == SkillRequirementType.REQUIRED
        )

    def test_preferred_signal(self):
        assert (
            classify_requirement_type("AWS experience is preferred.", JobSectionType.REQUIREMENTS)
            == SkillRequirementType.PREFERRED
        )

    def test_mentioned_in_responsibilities(self):
        assert (
            classify_requirement_type("You will work with AWS and Docker.", JobSectionType.RESPONSIBILITIES)
            == SkillRequirementType.MENTIONED
        )

    def test_responsibility_skills_not_required(self):
        jd = """
Senior Developer

Responsibilities
- Deploy models to AWS
- Build pipelines with Docker

Required Qualifications
Must have Python and SQL.
""".strip()
        parsed = parse_job_description(jd)
        required = {s.normalized_skill for s in parsed.required_skills}
        mentioned = {s.normalized_skill for s in parsed.mentioned_skills}
        assert "Python" in required
        assert "AWS" not in required
        assert "AWS" in mentioned


class TestSkillGap:
    def _resume_with_skills(self, *skills: str) -> object:
        text = "SKILLS\n" + ", ".join(skills)
        return HeuristicResumeParser().parse(
            Document(source_type=DocumentSourceType.TXT, raw_text=text)
        )

    def test_complete_required_match(self):
        resume = self._resume_with_skills("Python", "SQL", "PyTorch")
        job = parse_job_description(
            "Requirements\nMust have Python, SQL, and PyTorch."
        )
        gap = compute_skill_gap(resume, job)
        assert gap.required_matched == 3
        assert gap.required_missing == 0
        assert gap.required_skill_coverage == 100.0

    def test_partial_match(self):
        resume = self._resume_with_skills("Python", "SQL", "Docker")
        job = parse_job_description(
            "Requirements\nMust have Python, SQL, PyTorch."
        )
        gap = compute_skill_gap(resume, job)
        assert gap.required_matched == 2
        assert gap.required_missing == 1
        assert any(e.skill == "PyTorch" for e in gap.missing_required)
        assert abs(gap.required_skill_coverage - 66.666) < 0.1

    def test_zero_match(self):
        resume = self._resume_with_skills("Java")
        job = parse_job_description("Requirements\nMust have Python.")
        gap = compute_skill_gap(resume, job)
        assert gap.required_matched == 0
        assert gap.required_skill_coverage == 0.0

    def test_preferred_only_match(self):
        resume = self._resume_with_skills("Python", "AWS")
        job = parse_job_description(
            "Requirements\nPython required.\nPreferred: AWS experience is a plus."
        )
        gap = compute_skill_gap(resume, job)
        assert gap.preferred_matched == 1
        assert any(e.skill == "AWS" for e in gap.matched_preferred)

    def test_additional_candidate_skills(self):
        resume = self._resume_with_skills("Python", "Docker")
        job = parse_job_description("Requirements\nMust have Python.")
        gap = compute_skill_gap(resume, job)
        assert any(e.skill == "Docker" for e in gap.additional_candidate_skills)

    def test_evidence_preserved(self):
        resume = HeuristicResumeParser().parse(
            Document(source_type=DocumentSourceType.TXT, raw_text=SAMPLE_RESUME)
        )
        job = parse_job_description(ML_ENGINEER_JD)
        gap = compute_skill_gap(resume, job)
        if gap.matched_required:
            assert gap.matched_required[0].candidate_evidence
            assert gap.matched_required[0].job_evidence
        if gap.missing_required:
            assert gap.missing_required[0].job_evidence


class TestEntityExtractor:
    def test_resume_entity_profile(self):
        resume = HeuristicResumeParser().parse(
            Document(source_type=DocumentSourceType.TXT, raw_text=SAMPLE_RESUME)
        )
        profile = EntityExtractor().extract_from_resume(resume)
        assert profile.source_type.value == "resume"
        assert profile.normalized_skill_names
        assert "Python" in profile.normalized_skill_names

    def test_job_entity_profile(self):
        job = parse_job_description(REQUIRED_PREFERRED_JD)
        profile = EntityExtractor().extract_from_job(job)
        assert profile.source_type.value == "job"
        assert profile.technical_skills

    def test_candidate_profile_structure(self):
        resume = HeuristicResumeParser().parse(
            Document(source_type=DocumentSourceType.TXT, raw_text=SAMPLE_RESUME)
        )
        profile = EntityExtractor().build_candidate_profile(resume)
        assert profile.technical_skills
        assert profile.education or profile.experience

    def test_job_profile_structure(self):
        job = parse_job_description(REQUIRED_PREFERRED_JD)
        profile = EntityExtractor().build_job_profile(job)
        assert profile.required_technical_skills or profile.preferred_technical_skills


class TestEdgeCases:
    def test_c_not_cpp(self):
        resume = HeuristicResumeParser().parse(
            Document(source_type=DocumentSourceType.TXT, raw_text="SKILLS\nC++")
        )
        job = parse_job_description("Requirements\nMust have C++.")
        gap = compute_skill_gap(resume, job)
        assert gap.required_matched == 1
        assert gap.matched_required[0].skill == "C++"

    def test_empty_required_skills(self):
        resume = HeuristicResumeParser().parse(
            Document(source_type=DocumentSourceType.TXT, raw_text="SKILLS\nPython")
        )
        job = parse_job_description(
            "Responsibilities\nYou will work with Docker.\nPreferred: AWS is a plus."
        )
        gap = compute_skill_gap(resume, job)
        assert gap.required_total == 0
        assert gap.required_skill_coverage == 0.0
        assert gap.preferred_total >= 0

    def test_duplicate_skills_deduped(self):
        resume = HeuristicResumeParser().parse(
            Document(
                source_type=DocumentSourceType.TXT,
                raw_text="SKILLS\nPython, python, Python3",
            )
        )
        profile = EntityExtractor().build_candidate_profile(resume)
        python_count = sum(
            1 for s in profile.normalized_skill_names if s.lower() == "python"
        )
        assert python_count == 1

    def test_mentioned_skills_in_gap_result(self):
        jd = """
Responsibilities
- Deploy models to AWS

Required Qualifications
Must have Python.
""".strip()
        job = parse_job_description(jd)
        resume = HeuristicResumeParser().parse(
            Document(source_type=DocumentSourceType.TXT, raw_text="SKILLS\nPython, AWS")
        )
        gap = compute_skill_gap(resume, job)
        mentioned = {e.skill for e in gap.mentioned_skills}
        assert "AWS" in mentioned
        assert gap.required_matched == 1

    def test_coverage_calculation(self):
        resume = HeuristicResumeParser().parse(
            Document(source_type=DocumentSourceType.TXT, raw_text="SKILLS\nPython, SQL")
        )
        job = parse_job_description(
            "Requirements\nMust have Python, SQL, PyTorch."
        )
        gap = compute_skill_gap(resume, job)
        assert gap.required_total == 3
        assert gap.required_matched == 2
        assert abs(gap.required_skill_coverage - 66.666) < 0.1
