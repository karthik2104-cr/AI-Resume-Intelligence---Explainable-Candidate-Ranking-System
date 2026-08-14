"""Unit tests for job description parser."""

from __future__ import annotations

import pytest

from src.models.job import SkillRequirementType
from src.parsing.job_parser import HeuristicJobDescriptionParser, parse_job_description
from src.parsing.jd_quality import assess_jd_parsing_quality
from tests.fixtures.jd_samples import (
    EDUCATION_JD,
    EXPERIENCE_JD,
    FALSE_POSITIVE_JD,
    JUNIOR_JD,
    MESSY_JD,
    ML_ENGINEER_JD,
    REQUIRED_PREFERRED_JD,
    SKILL_ALIAS_JD,
)


@pytest.fixture
def parser() -> HeuristicJobDescriptionParser:
    return HeuristicJobDescriptionParser()


class TestJobTitleExtraction:
    def test_title_from_header(self, parser: HeuristicJobDescriptionParser):
        parsed = parser.parse_text(ML_ENGINEER_JD)
        assert parsed.title is not None
        assert "Machine Learning Engineer" in parsed.title
        assert parsed.seniority_level is not None
        assert "senior" in parsed.seniority_level.lower()

    def test_explicit_job_title_label(self, parser: HeuristicJobDescriptionParser):
        parsed = parser.parse_text(JUNIOR_JD)
        assert parsed.job_title is not None
        assert parsed.job_title.seniority_level == "Junior"
        assert "Engineer" in parsed.job_title.raw_title


class TestRequiredPreferred:
    def test_must_have_required(self, parser: HeuristicJobDescriptionParser):
        parsed = parser.parse_text(REQUIRED_PREFERRED_JD)
        required = {s.normalized_skill for s in parsed.required_skills}
        preferred = {s.normalized_skill for s in parsed.preferred_skills}
        assert "Python" in required
        assert "SQL" in required
        assert "AWS" in preferred
        assert "Docker" in preferred

    def test_evidence_preserved(self, parser: HeuristicJobDescriptionParser):
        parsed = parser.parse_text(REQUIRED_PREFERRED_JD)
        aws = next(s for s in parsed.preferred_skills if s.normalized_skill == "AWS")
        assert "plus" in aws.evidence.lower()
        assert aws.source_section


class TestExperienceExtraction:
    def test_plus_years(self, parser: HeuristicJobDescriptionParser):
        parsed = parser.parse_text(ML_ENGINEER_JD)
        assert parsed.experience_requirements
        mins = [r.min_years for r in parsed.experience_requirements if r.min_years is not None]
        assert any(y >= 3 for y in mins)

    def test_range_years(self, parser: HeuristicJobDescriptionParser):
        parsed = parser.parse_text(EXPERIENCE_JD)
        range_req = next((r for r in parsed.experience_requirements if r.max_years), None)
        assert range_req is not None
        assert range_req.min_years == 2
        assert range_req.max_years == 5

    def test_entry_level(self, parser: HeuristicJobDescriptionParser):
        parsed = parser.parse_text(EXPERIENCE_JD)
        assert any(r.is_entry_level for r in parsed.experience_requirements)


class TestEducationExtraction:
    def test_degree_extraction(self, parser: HeuristicJobDescriptionParser):
        parsed = parser.parse_text(EDUCATION_JD)
        assert parsed.education_requirements
        levels = [e.degree_level for e in parsed.education_requirements if e.degree_level]
        assert any("bachelor" in (lvl or "").lower() for lvl in levels)

    def test_preferred_education(self, parser: HeuristicJobDescriptionParser):
        parsed = parser.parse_text(EDUCATION_JD)
        assert any(
            e.requirement_type == SkillRequirementType.PREFERRED
            for e in parsed.education_requirements
        )


class TestResponsibilities:
    def test_responsibilities_extracted(self, parser: HeuristicJobDescriptionParser):
        parsed = parser.parse_text(ML_ENGINEER_JD)
        assert len(parsed.responsibilities) >= 3
        texts = parsed.responsibility_texts
        assert any("machine learning" in t.lower() for t in texts)


class TestMetadata:
    def test_work_mode_and_location(self, parser: HeuristicJobDescriptionParser):
        parsed = parser.parse_text(ML_ENGINEER_JD)
        assert parsed.work_mode == "hybrid"
        assert parsed.location is not None
        assert "Bangalore" in parsed.location

    def test_employment_type(self, parser: HeuristicJobDescriptionParser):
        parsed = parser.parse_text(ML_ENGINEER_JD)
        assert parsed.employment_type is not None
        assert "full" in parsed.employment_type.lower()


class TestSkillAliases:
    def test_sklearn_normalized(self, parser: HeuristicJobDescriptionParser):
        parsed = parser.parse_text(SKILL_ALIAS_JD)
        required = {s.normalized_skill for s in parsed.required_skills}
        assert "Scikit-learn" in required
        assert "JavaScript" in required


class TestFalsePositives:
    def test_no_java_from_javascript(self, parser: HeuristicJobDescriptionParser):
        parsed = parser.parse_text(FALSE_POSITIVE_JD)
        required = {s.normalized_skill for s in parsed.required_skills}
        preferred = {s.normalized_skill for s in parsed.preferred_skills}
        all_skills = required | preferred
        assert "JavaScript" in all_skills
        assert "Java" in preferred  # explicitly mentioned as plus
        assert "Java" not in required


class TestMessyJD:
    def test_messy_jd_still_extracts_skills(self, parser: HeuristicJobDescriptionParser):
        parsed = parser.parse_text(MESSY_JD)
        all_skills = {s.normalized_skill for s in parsed.required_skills + parsed.preferred_skills}
        assert "Python" in all_skills
        assert "JavaScript" in all_skills
        assert parsed.work_mode == "remote"


class TestParsingQuality:
    def test_quality_high_for_structured_jd(self, parser: HeuristicJobDescriptionParser):
        parsed = parser.parse_text(ML_ENGINEER_JD)
        assert parsed.parsing_quality.level in {"high", "medium"}
        assert parsed.parsing_quality.score >= 4

    def test_quality_warnings_for_sparse_jd(self):
        parsed = parse_job_description("Short job ad.")
        quality = assess_jd_parsing_quality(parsed)
        assert quality.level == "low"
        assert quality.warnings


class TestParseJobDescriptionFunction:
    def test_convenience_function(self):
        parsed = parse_job_description(ML_ENGINEER_JD)
        assert parsed.title is not None
        assert parsed.required_skills
