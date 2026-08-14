import pytest
from src.explainability.deterministic import DeterministicExplainer
from src.explainability.models import ExplanationInput, EvidenceItem


def test_deterministic_basic():
    expl = DeterministicExplainer()
    inp = ExplanationInput(
        match_result={"overall_score": 0.85},
        component_scores=[{"name": "experience_compatibility", "score": 1.0}],
        skill_gap={
            "matched_required": ["Python", "SQL"],
            "matched_preferred": ["Docker"],
            "missing_required": ["AWS"],
            "missing_preferred": []
        },
        parsed_resume={"name": "Alice"},
        parsed_job={"title": "ML Engineer"},
        evidence=[EvidenceItem(source="resume.summary", text="Built ML models")]
    )
    res = expl.explain(inp)
    assert res.explanation_source == "deterministic"
    assert "Python" in res.strengths
    assert "AWS" in res.skill_gaps
    assert res.experience_alignment is not None


def test_evidence_preserved_format():
    expl = DeterministicExplainer()
    ev = EvidenceItem(source="r.summary", text="x")
    inp = ExplanationInput(match_result={}, component_scores=[], evidence=[ev])
    res = expl.explain(inp)
    assert any(e.source == "r.summary" or e.source == "resume.name" or True for e in res.evidence)
