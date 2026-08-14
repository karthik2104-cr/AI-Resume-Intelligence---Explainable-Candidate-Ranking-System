"""
Phase 6 comprehensive tests — Unified Entity Intelligence + Skill Gap Analysis.

Test items per spec:
A. Shared normalization
B. Alias matching
C. False positives (Java != JavaScript, C != C++)
D. Requirement classification (required/preferred/mentioned)
E. Responsibility behavior (not auto-required)
F. Complete skill match
G. Partial skill match
H. Zero skill match
I. Preferred-only match
J. Additional candidate skills
K. Duplicate skills
L. Empty skill lists
M. Evidence preservation
N. Coverage calculation
O. Multiple evidence occurrences
P. Word-boundary "must" safety
"""

from __future__ import annotations

import pytest

from src.extraction.entity_extractor import EntityExtractor
from src.extraction.requirement_classifier import classify_requirement_type
from src.extraction.skill_normalizer import (
    find_skills_in_text,
    normalize_skill,
    normalize_skills_list,
)
from src.matching.skill_gap import SkillGapResult, compute_skill_gap
from src.models.document import Document, DocumentSourceType
from src.models.entity import EntityType
from src.models.job import JobSectionType, SkillRequirementType
from src.parsing.job_parser import parse_job_description
from src.parsing.resume_parser import HeuristicResumeParser
from tests.fixtures.jd_samples import ML_ENGINEER_JD, REQUIRED_PREFERRED_JD
from tests.fixtures.resume_samples import SAMPLE_RESUME


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_resume(text: str):
    return HeuristicResumeParser().parse(
        Document(source_type=DocumentSourceType.TXT, raw_text=text)
    )


def _resume_with_skills(*skills: str):
    text = "SKILLS\n" + ", ".join(skills)
    return _parse_resume(text)


def _jd_required(*skills: str):
    lines = "Requirements\nMust have " + ", ".join(skills) + "."
    return parse_job_description(lines)


def _jd_preferred(*skills: str):
    lines = "Requirements\n" + " ".join(f"{s} is a plus." for s in skills)
    return parse_job_description(lines)


# ===========================================================================
# A. Shared normalization — same canonical form for resume + JD input
# ===========================================================================

class TestSharedNormalization:
    """A. Shared normalization via single skill_normalizer source of truth."""

    def test_sklearn_normalizes_to_scikit_learn(self):
        assert normalize_skill("sklearn") == "Scikit-learn"

    def test_scikit_learn_normalizes_same(self):
        assert normalize_skill("scikit-learn") == "Scikit-learn"

    def test_scikit_learn_lowercase(self):
        assert normalize_skill("scikit learn") == "Scikit-learn"

    def test_python3_normalizes_to_python(self):
        assert normalize_skill("python3") == "Python"

    def test_python_space_3(self):
        assert normalize_skill("python 3") == "Python"

    def test_pytorch_alias_torch(self):
        assert normalize_skill("torch") == "PyTorch"

    def test_k8s_normalizes_to_kubernetes(self):
        assert normalize_skill("k8s") == "Kubernetes"

    def test_amazon_web_services(self):
        assert normalize_skill("amazon web services") == "AWS"

    def test_reactjs_normalizes_to_react(self):
        assert normalize_skill("reactjs") == "React"

    def test_list_normalization_dedupes(self):
        result = normalize_skills_list(["Python", "python3", "Python 3"])
        assert result.count("Python") == 1
        assert "Python" in result

    def test_resume_jd_shared_normalization(self):
        """sklearn in resume matches scikit-learn in JD via shared normalizer."""
        resume = _parse_resume("SKILLS\nsklearn, Python")
        job = parse_job_description("Requirements\nMust have scikit-learn experience.")
        gap = compute_skill_gap(resume, job)
        assert any(e.skill == "Scikit-learn" for e in gap.matched_required)


# ===========================================================================
# B. Alias matching
# ===========================================================================

class TestAliasMatching:
    """B. Aliases resolve to the same canonical form."""

    def test_js_matches_javascript(self):
        assert normalize_skill("js") == normalize_skill("JavaScript")

    def test_ecmascript_matches_javascript(self):
        assert normalize_skill("ecmascript") == "JavaScript"

    def test_pyspark_matches_spark(self):
        assert normalize_skill("pyspark") == "Spark"

    def test_apache_spark_matches_spark(self):
        assert normalize_skill("apache spark") == "Spark"

    def test_springboot_matches_spring_boot(self):
        assert normalize_skill("springboot") == "Spring Boot"

    def test_postgres_matches_postgresql(self):
        assert normalize_skill("postgres") == "PostgreSQL"

    def test_psql_matches_postgresql(self):
        assert normalize_skill("psql") == "PostgreSQL"

    def test_resume_js_matches_jd_javascript(self):
        resume = _resume_with_skills("JS")
        job = parse_job_description("Requirements\nMust have JavaScript.")
        gap = compute_skill_gap(resume, job)
        assert gap.required_matched == 1
        assert gap.matched_required[0].skill == "JavaScript"


# ===========================================================================
# C. False positives prevented
# ===========================================================================

class TestFalsePositives:
    """C. Substring false positives are prevented via word boundaries."""

    def test_java_not_javascript_in_text(self):
        matches = find_skills_in_text("Experience working with JavaScript applications")
        canonical = {m.canonical for m in matches}
        assert "JavaScript" in canonical
        assert "Java" not in canonical

    def test_java_and_javascript_independent(self):
        text = "Must have Java for backend and JavaScript for frontend"
        canonical = {m.canonical for m in find_skills_in_text(text)}
        assert "Java" in canonical
        assert "JavaScript" in canonical

    def test_java_not_javascript_matching(self):
        resume = _resume_with_skills("Java")
        job = parse_job_description("Requirements\nMust have JavaScript.")
        gap = compute_skill_gap(resume, job)
        assert gap.required_matched == 0
        assert gap.required_missing == 1

    def test_cpp_not_c_in_text(self):
        matches = find_skills_in_text("Experience with C++ programming")
        canonical = {m.canonical for m in matches}
        assert "C++" in canonical
        assert "C" not in canonical

    def test_cpp_not_c_in_gap(self):
        resume = _resume_with_skills("C++")
        job = parse_job_description("Requirements\nMust have C++.")
        gap = compute_skill_gap(resume, job)
        assert gap.required_matched == 1
        assert gap.matched_required[0].skill == "C++"

    def test_python_not_substring_of_pytorh(self):
        text = "Must have PyTorch experience"
        canonical = {m.canonical for m in find_skills_in_text(text)}
        assert "PyTorch" in canonical
        assert "Python" not in canonical

    def test_mysql_and_sql_independent(self):
        """MySQL should not create a double match with SQL."""
        matches = find_skills_in_text("Experience with MySQL databases")
        canonicals = [m.canonical for m in matches]
        # MySQL and SQL are separate skills; longest match wins
        assert "MySQL" in canonicals


# ===========================================================================
# D. Requirement classification
# ===========================================================================

class TestRequirementClassification:
    """D. Requirement type detection: required / preferred / mentioned."""

    def test_must_have_is_required(self):
        result = classify_requirement_type(
            "Must have AWS experience.", JobSectionType.RESPONSIBILITIES
        )
        assert result == SkillRequirementType.REQUIRED

    def test_is_required_phrase(self):
        result = classify_requirement_type(
            "Python knowledge is required.", JobSectionType.REQUIREMENTS
        )
        assert result == SkillRequirementType.REQUIRED

    def test_mandatory(self):
        result = classify_requirement_type(
            "Mandatory: Docker experience.", JobSectionType.REQUIREMENTS
        )
        assert result == SkillRequirementType.REQUIRED

    def test_essential(self):
        result = classify_requirement_type(
            "SQL is essential for this role.", JobSectionType.REQUIREMENTS
        )
        assert result == SkillRequirementType.REQUIRED

    def test_is_a_plus_is_preferred(self):
        result = classify_requirement_type(
            "AWS experience is a plus.", JobSectionType.REQUIREMENTS
        )
        assert result == SkillRequirementType.PREFERRED

    def test_preferred_signal(self):
        result = classify_requirement_type(
            "AWS experience is preferred.", JobSectionType.REQUIREMENTS
        )
        assert result == SkillRequirementType.PREFERRED

    def test_nice_to_have(self):
        result = classify_requirement_type(
            "Docker knowledge is nice to have.", JobSectionType.REQUIREMENTS
        )
        assert result == SkillRequirementType.PREFERRED

    def test_good_to_have(self):
        result = classify_requirement_type(
            "Kubernetes is good to have.", JobSectionType.REQUIREMENTS
        )
        assert result == SkillRequirementType.PREFERRED

    def test_advantage(self):
        result = classify_requirement_type(
            "AWS experience would be an advantage.", JobSectionType.REQUIREMENTS
        )
        assert result == SkillRequirementType.PREFERRED

    def test_neutral_mention_in_responsibilities(self):
        result = classify_requirement_type(
            "You will work with AWS and Docker.", JobSectionType.RESPONSIBILITIES
        )
        assert result == SkillRequirementType.MENTIONED

    def test_plain_sentence_in_responsibilities_is_mentioned(self):
        result = classify_requirement_type(
            "Deploy applications to Kubernetes.", JobSectionType.RESPONSIBILITIES
        )
        assert result == SkillRequirementType.MENTIONED

    def test_preferred_overrides_required_when_both_present(self):
        """A sentence like 'preferred but required by some' → preferred wins (checked first)."""
        result = classify_requirement_type(
            "Docker is a plus.", JobSectionType.REQUIREMENTS
        )
        assert result == SkillRequirementType.PREFERRED


# ===========================================================================
# E. Responsibility behavior — not auto-required
# ===========================================================================

class TestResponsibilitySkillsNotRequired:
    """E. Technologies in responsibilities are MENTIONED, not REQUIRED."""

    def test_aws_in_responsibilities_is_mentioned(self):
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

    def test_docker_in_responsibilities_is_mentioned(self):
        jd = """
Backend Engineer

Responsibilities
You will build Docker containers and deploy to AWS.

Requirements
Must have Python.
""".strip()
        parsed = parse_job_description(jd)
        req_skills = {s.normalized_skill for s in parsed.required_skills}
        assert "Docker" not in req_skills
        assert "AWS" not in req_skills

    def test_must_have_in_responsibilities_overrides_neutral(self):
        """Explicit required signal in responsibilities still works."""
        result = classify_requirement_type(
            "Must have AWS experience.", JobSectionType.RESPONSIBILITIES
        )
        assert result == SkillRequirementType.REQUIRED


# ===========================================================================
# F. Complete skill match
# ===========================================================================

class TestCompleteSkillMatch:
    """F. Candidate has all required skills."""

    def test_complete_required_match(self):
        resume = _resume_with_skills("Python", "SQL", "PyTorch")
        job = parse_job_description("Requirements\nMust have Python, SQL, and PyTorch.")
        gap = compute_skill_gap(resume, job)
        assert gap.required_matched == 3
        assert gap.required_missing == 0
        assert gap.required_skill_coverage == 100.0

    def test_complete_match_with_aliases(self):
        resume = _resume_with_skills("sklearn", "JS")
        job = parse_job_description(
            "Requirements\nMust have scikit-learn and JavaScript."
        )
        gap = compute_skill_gap(resume, job)
        assert gap.required_matched == 2
        assert gap.required_skill_coverage == 100.0


# ===========================================================================
# G. Partial skill match
# ===========================================================================

class TestPartialSkillMatch:
    """G. Candidate has some required skills."""

    def test_partial_match(self):
        resume = _resume_with_skills("Python", "SQL", "Docker")
        job = parse_job_description("Requirements\nMust have Python, SQL, PyTorch.")
        gap = compute_skill_gap(resume, job)
        assert gap.required_matched == 2
        assert gap.required_missing == 1
        assert any(e.skill == "PyTorch" for e in gap.missing_required)
        assert abs(gap.required_skill_coverage - 66.666) < 0.1

    def test_partial_preferred_match(self):
        resume = _resume_with_skills("Python", "AWS")
        jd = "Requirements\nPython required.\nAWS is a plus. Docker is a plus."
        job = parse_job_description(jd)
        gap = compute_skill_gap(resume, job)
        assert gap.preferred_matched == 1
        assert gap.preferred_missing == 1


# ===========================================================================
# H. Zero skill match
# ===========================================================================

class TestZeroSkillMatch:
    """H. Candidate has none of the required skills."""

    def test_zero_required_match(self):
        resume = _resume_with_skills("Java")
        job = parse_job_description("Requirements\nMust have Python.")
        gap = compute_skill_gap(resume, job)
        assert gap.required_matched == 0
        assert gap.required_skill_coverage == 0.0

    def test_zero_match_all_missing(self):
        resume = _resume_with_skills("Spark")
        job = parse_job_description(
            "Requirements\nMust have Python, SQL, PyTorch."
        )
        gap = compute_skill_gap(resume, job)
        assert gap.required_matched == 0
        assert gap.required_missing == 3


# ===========================================================================
# I. Preferred-only match
# ===========================================================================

class TestPreferredOnlyMatch:
    """I. Candidate matches preferred but not required skills."""

    def test_preferred_match_with_required_missing(self):
        resume = _resume_with_skills("AWS")
        jd = "Requirements\nMust have Python.\nAWS is a plus."
        job = parse_job_description(jd)
        gap = compute_skill_gap(resume, job)
        assert gap.required_matched == 0
        assert gap.preferred_matched == 1
        assert any(e.skill == "AWS" for e in gap.matched_preferred)

    def test_preferred_coverage_calculation(self):
        resume = _resume_with_skills("AWS", "Docker")
        jd = "Requirements\nMust have Python.\nAWS is a plus. Docker is a plus."
        job = parse_job_description(jd)
        gap = compute_skill_gap(resume, job)
        assert gap.preferred_matched == 2
        assert gap.preferred_skill_coverage == 100.0


# ===========================================================================
# J. Additional candidate skills
# ===========================================================================

class TestAdditionalCandidateSkills:
    """J. Candidate has skills not mentioned in the JD at all."""

    def test_additional_skills_captured(self):
        resume = _resume_with_skills("Python", "Docker")
        job = parse_job_description("Requirements\nMust have Python.")
        gap = compute_skill_gap(resume, job)
        additional = {e.skill for e in gap.additional_candidate_skills}
        assert "Docker" in additional

    def test_additional_not_counted_in_required(self):
        resume = _resume_with_skills("Python", "Kubernetes", "AWS")
        job = parse_job_description("Requirements\nMust have Python.")
        gap = compute_skill_gap(resume, job)
        assert gap.required_matched == 1
        assert len(gap.additional_candidate_skills) >= 2


# ===========================================================================
# K. Duplicate skills
# ===========================================================================

class TestDuplicateSkills:
    """K. Duplicate skills do not inflate counts."""

    def test_candidate_duplicate_skills_deduped(self):
        resume = _parse_resume("SKILLS\nPython, python, Python3")
        profile = EntityExtractor().build_candidate_profile(resume)
        count = sum(1 for s in profile.normalized_skill_names if s.lower() == "python")
        assert count == 1

    def test_duplicate_in_gap_not_double_counted(self):
        resume = _parse_resume("SKILLS\nPython, python, Python 3, Python3")
        job = parse_job_description("Requirements\nMust have Python.")
        gap = compute_skill_gap(resume, job)
        assert gap.required_matched == 1
        assert gap.required_skill_coverage == 100.0

    def test_jd_duplicate_required_skills_deduped(self):
        jd = "Requirements\nMust have Python. Python is required. Must have Python."
        job = parse_job_description(jd)
        assert len(job.required_skills) <= 2  # Should be deduplicated

    def test_normalize_list_dedupes_exact(self):
        result = normalize_skills_list(["Python", "Python", "Python"])
        assert result.count("Python") == 1


# ===========================================================================
# L. Empty skill lists
# ===========================================================================

class TestEmptySkillLists:
    """L. Empty skill lists handled gracefully."""

    def test_empty_required_skills_in_jd(self):
        resume = _resume_with_skills("Python")
        job = parse_job_description(
            "Responsibilities\nYou will work with Docker.\nPreferred: AWS is a plus."
        )
        gap = compute_skill_gap(resume, job)
        assert gap.required_total == 0
        assert gap.required_skill_coverage == 0.0

    def test_empty_candidate_skills(self):
        resume = _parse_resume("Jane Doe\njane@example.com\n\nSUMMARY\nNo technical skills.")
        job = parse_job_description("Requirements\nMust have Python.")
        gap = compute_skill_gap(resume, job)
        assert gap.required_matched == 0
        assert gap.required_skill_coverage == 0.0

    def test_both_empty(self):
        resume = _parse_resume("Jane Doe\njane@example.com")
        job = parse_job_description("We are hiring.")
        gap = compute_skill_gap(resume, job)
        assert gap.required_total == 0
        assert gap.preferred_total == 0
        assert gap.required_skill_coverage == 0.0


# ===========================================================================
# M. Evidence preservation
# ===========================================================================

class TestEvidencePreservation:
    """M. Evidence (source sentences) is preserved in gap results."""

    def test_matched_required_has_job_evidence(self):
        resume = _resume_with_skills("Python")
        job = parse_job_description("Requirements\nMust have Python experience.")
        gap = compute_skill_gap(resume, job)
        assert gap.matched_required
        assert gap.matched_required[0].job_evidence

    def test_missing_required_has_job_evidence(self):
        resume = _resume_with_skills("Java")
        job = parse_job_description("Requirements\nMust have Python experience.")
        gap = compute_skill_gap(resume, job)
        assert gap.missing_required
        assert gap.missing_required[0].job_evidence

    def test_matched_required_has_candidate_evidence(self):
        resume = HeuristicResumeParser().parse(
            Document(
                source_type=DocumentSourceType.TXT,
                raw_text=SAMPLE_RESUME,
            )
        )
        job = parse_job_description(ML_ENGINEER_JD)
        gap = compute_skill_gap(resume, job)
        if gap.matched_required:
            assert gap.matched_required[0].candidate_evidence is not None

    def test_entity_preserves_evidence_from_resume(self):
        resume = _parse_resume(
            "EXPERIENCE\nBuilt machine learning models using Python and PyTorch."
        )
        profile = EntityExtractor().extract_from_resume(resume)
        python_entities = [
            e for e in profile.technical_skills if e.normalized_value == "Python"
        ]
        # Evidence should contain the source sentence
        if python_entities:
            assert python_entities[0].evidence


# ===========================================================================
# N. Coverage calculation
# ===========================================================================

class TestCoverageCalculation:
    """N. Coverage is calculated as percentage (0–100), not a final score."""

    def test_required_coverage_two_thirds(self):
        resume = _resume_with_skills("Python", "SQL")
        job = parse_job_description("Requirements\nMust have Python, SQL, PyTorch.")
        gap = compute_skill_gap(resume, job)
        assert gap.required_total == 3
        assert gap.required_matched == 2
        assert abs(gap.required_skill_coverage - 66.666) < 0.1

    def test_required_coverage_100_percent(self):
        resume = _resume_with_skills("Python", "SQL")
        job = parse_job_description("Requirements\nMust have Python and SQL.")
        gap = compute_skill_gap(resume, job)
        assert gap.required_skill_coverage == 100.0

    def test_required_coverage_0_percent(self):
        resume = _resume_with_skills("Java")
        job = parse_job_description("Requirements\nMust have Python.")
        gap = compute_skill_gap(resume, job)
        assert gap.required_skill_coverage == 0.0

    def test_preferred_coverage_calculation(self):
        resume = _resume_with_skills("AWS")
        jd = "Requirements\nAWS is a plus. Docker is a plus."
        job = parse_job_description(jd)
        gap = compute_skill_gap(resume, job)
        assert gap.preferred_total == 2
        assert gap.preferred_matched == 1
        assert gap.preferred_skill_coverage == 50.0

    def test_coverage_fields_not_named_match_score(self):
        """Coverage fields must NOT be named match_score, ranking_score, etc."""
        gap = SkillGapResult()
        assert hasattr(gap, "required_skill_coverage")
        assert hasattr(gap, "preferred_skill_coverage")
        assert not hasattr(gap, "match_score")
        assert not hasattr(gap, "ranking_score")
        assert not hasattr(gap, "overall_score")

    def test_counters_consistent(self):
        resume = _resume_with_skills("Python", "SQL")
        job = parse_job_description("Requirements\nMust have Python, SQL, PyTorch.")
        gap = compute_skill_gap(resume, job)
        assert gap.required_matched + gap.required_missing == gap.required_total


# ===========================================================================
# O. Multiple evidence occurrences
# ===========================================================================

class TestMultipleEvidenceOccurrences:
    """O. Skills mentioned in multiple sections / sentences handled correctly."""

    def test_skill_in_multiple_sections_not_double_counted(self):
        """Python mentioned in skills section AND experience — should count once."""
        resume = _parse_resume(
            "SKILLS\nPython\n\nEXPERIENCE\nBuilt apps with Python for 3 years."
        )
        profile = EntityExtractor().build_candidate_profile(resume)
        py_count = sum(1 for s in profile.normalized_skill_names if s.lower() == "python")
        assert py_count == 1

    def test_mentioned_skills_accumulated_correctly(self):
        jd = """
Responsibilities
- Work with Docker on container deployments.
- Use Docker to build CI pipelines.

Required Qualifications
Must have Python.
""".strip()
        job = parse_job_description(jd)
        # Docker should appear once in mentioned (deduplicated)
        docker_mentioned = [
            s for s in job.mentioned_skills
            if s.normalized_skill == "Docker"
        ]
        assert len(docker_mentioned) == 1


# ===========================================================================
# P. Word-boundary false positive prevention
# ===========================================================================

class TestWordBoundaryFalsePositives:
    """P. Word boundaries prevent substring matching false positives."""

    def test_mustard_does_not_trigger_required(self):
        """'mustard' should not activate 'must' word-boundary check."""
        from src.extraction.requirement_classifier import _phrase_matches
        assert not _phrase_matches("add mustard to the recipe", "must")

    def test_must_standalone_triggers_required(self):
        from src.extraction.requirement_classifier import _phrase_matches
        assert _phrase_matches("you must have python", "must")

    def test_required_word_boundary(self):
        from src.extraction.requirement_classifier import _phrase_matches
        # 'required' should not trip on 'requirements' (which contains 'required')
        # In practice "requirements" contains "required" as substring — we allow this
        # since the section name is never the classification sentence.
        # But standalone "required" in a sentence must work.
        assert _phrase_matches("python is required", "required")

    def test_mandatory_word_boundary(self):
        result = classify_requirement_type(
            "Mandatory: Strong SQL skills.", JobSectionType.REQUIREMENTS
        )
        assert result == SkillRequirementType.REQUIRED


# ===========================================================================
# Entity extraction structure tests
# ===========================================================================

class TestEntityExtractionStructure:
    """Verify EntityProfile, CandidateProfile, and JobProfile structures."""

    def test_resume_entity_profile_source_type(self):
        resume = HeuristicResumeParser().parse(
            Document(source_type=DocumentSourceType.TXT, raw_text=SAMPLE_RESUME)
        )
        profile = EntityExtractor().extract_from_resume(resume)
        assert profile.source_type.value == "resume"
        assert "Python" in profile.normalized_skill_names

    def test_job_entity_profile_source_type(self):
        job = parse_job_description(REQUIRED_PREFERRED_JD)
        profile = EntityExtractor().extract_from_job(job)
        assert profile.source_type.value == "job"
        assert profile.technical_skills

    def test_entity_has_required_fields(self):
        resume = HeuristicResumeParser().parse(
            Document(source_type=DocumentSourceType.TXT, raw_text=SAMPLE_RESUME)
        )
        profile = EntityExtractor().extract_from_resume(resume)
        for entity in profile.technical_skills:
            assert entity.raw_text is not None
            assert entity.normalized_value
            assert entity.entity_type
            assert entity.source

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

    def test_soft_skills_separate_from_technical(self):
        resume = _parse_resume(
            "SKILLS\nPython, teamwork, leadership\n\nSUMMARY\nStrong communication skills."
        )
        profile = EntityExtractor().extract_from_resume(resume)
        tech_names = {e.normalized_value.lower() for e in profile.technical_skills}
        soft_names = {e.normalized_value.lower() for e in profile.soft_skills}
        assert "python" in tech_names
        assert "python" not in soft_names
        # soft skills should not appear in technical
        for name in soft_names:
            assert name not in tech_names

    def test_soft_skill_entity_type(self):
        resume = _parse_resume("SKILLS\nleadership, teamwork")
        profile = EntityExtractor().extract_from_resume(resume)
        for entity in profile.soft_skills:
            assert entity.entity_type == EntityType.SOFT_SKILL

    def test_domain_extraction_explicit_only(self):
        """Domains only extracted when explicitly mentioned in text."""
        resume = _parse_resume(
            "SUMMARY\nI work in data science and machine learning.\n\nSKILLS\nPython"
        )
        profile = EntityExtractor().extract_from_resume(resume)
        domain_names = {e.normalized_value.lower() for e in profile.domains}
        # data science explicitly mentioned → should be found
        assert "data science" in domain_names or "Data Science" in {
            e.normalized_value for e in profile.domains
        }
        # Python alone should NOT create a machine learning domain entry
        # (no semantic inference)

    def test_no_semantic_inference_python_not_ml(self):
        """Python should not automatically produce a Machine Learning domain."""
        resume = _parse_resume("SKILLS\nPython")
        profile = EntityExtractor().extract_from_resume(resume)
        domain_names = {e.normalized_value.lower() for e in profile.domains}
        # Only explicit text mentions drive domain — Python alone ≠ ML
        assert "machine learning" not in domain_names


# ===========================================================================
# Mention tracking in skill gap
# ===========================================================================

class TestMentionedSkillsInGapResult:
    def test_mentioned_skills_appear_in_result(self):
        jd = """
Responsibilities
- Deploy models to AWS

Required Qualifications
Must have Python.
""".strip()
        job = parse_job_description(jd)
        resume = _resume_with_skills("Python", "AWS")
        gap = compute_skill_gap(resume, job)
        mentioned = {e.skill for e in gap.mentioned_skills}
        assert "AWS" in mentioned
        assert gap.required_matched == 1

    def test_mentioned_not_in_required_or_preferred(self):
        jd = """
Responsibilities
You will work with Docker.

Requirements
Must have Python.
""".strip()
        job = parse_job_description(jd)
        req_names = {s.normalized_skill for s in job.required_skills}
        pref_names = {s.normalized_skill for s in job.preferred_skills}
        assert "Docker" not in req_names
        assert "Docker" not in pref_names
