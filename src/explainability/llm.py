"""Abstract LLM explainer interface and a mock provider for tests."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from src.explainability.models import ExplanationInput, ExplanationResult


class LLMExplainer(ABC):
    """Abstract interface for LLM-based explainers."""

    @abstractmethod
    def explain(self, inp: ExplanationInput, timeout_seconds: int | None = None) -> ExplanationResult:
        """Produce an ExplanationResult from structured ExplanationInput.

        Implementations MUST NOT modify scores or ranking data in the input.
        """


class MockLLMProvider(LLMExplainer):
    """A simple mock LLM provider used for testing. It echoes structured input.

    It intentionally produces a safe, constrained ExplanationResult.
    """

    def explain(self, inp: ExplanationInput, timeout_seconds: int | None = None) -> ExplanationResult:
        # Build a lightweight readable summary from input without inventing facts
        strengths = []
        req = inp.skill_gap.get("matched_required") if inp.skill_gap and isinstance(inp.skill_gap, dict) else None
        pref = inp.skill_gap.get("matched_preferred") if inp.skill_gap and isinstance(inp.skill_gap, dict) else None
        gaps = inp.skill_gap.get("missing_required") if inp.skill_gap and isinstance(inp.skill_gap, dict) else None

        if isinstance(req, list):
            strengths = [str(x) for x in req[:5]]
        if isinstance(pref, list):
            strengths += [str(x) for x in pref[:5]]

        skill_gaps = [str(x) for x in (gaps or [])][:5]

        # Simple alignment strings
        experience_alignment = None
        if inp.hybrid_metadata and isinstance(inp.hybrid_metadata, dict):
            overall = inp.match_result.get("overall_score") or inp.match_result.get("scores", {}).get("overall")
            experience_alignment = f"Overall score (input): {overall}" if overall is not None else None

        evidence_items = []
        if inp.evidence:
            for e in inp.evidence[:5]:
                evidence_items.append(e)

        return ExplanationResult(
            summary="LLM-generated summary (mock): use only supplied evidence.",
            strengths=strengths,
            required_skill_matches=[str(x) for x in (req or [])],
            preferred_skill_matches=[str(x) for x in (pref or [])],
            skill_gaps=skill_gaps,
            experience_alignment=experience_alignment,
            education_alignment=None,
            seniority_alignment=None,
            concerns=[],
            interview_focus_areas=[],
            evidence=evidence_items,
            explanation_source="llm",
        )
