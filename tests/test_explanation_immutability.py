import copy

from src.explainability.service import ExplanationService
from src.explainability.models import ExplanationInput, ExplanationResult
from src.models.matching import MatchResult, ComponentScores


class MutatingLLM:
    # This mock LLM attempts to mutate the provided input's match_result
    def explain(self, inp: ExplanationInput, timeout_seconds: int = 5):
        # Try to mutate the passed dict/object
        try:
            if isinstance(inp.match_result, dict):
                inp.match_result.setdefault("candidate_id", "mutated")
                scores = inp.match_result.setdefault("scores", {})
                scores["overall"] = 999.0
        except Exception:
            pass
        # Return something incorrect type sometimes for testing
        return ExplanationResult(summary="llm")


def test_explanation_does_not_modify_matchresult():
    # Build a MatchResult and keep a deep copy
    mr = MatchResult(candidate_id="cand1", scores=ComponentScores(overall=0.5))
    original = copy.deepcopy(mr.model_dump())

    # Create ExplanationInput referencing the match result as dict
    inp = ExplanationInput(
        match_result=mr.model_dump(),
        component_scores=[],
        hybrid_metadata={},
        skill_gap={},
        parsed_resume={"candidate_id": "cand1"},
        parsed_job={},
        ranking_info={},
    )

    svc = ExplanationService(llm_provider=MutatingLLM())
    # Force LLM enabled to exercise provider path
    svc._settings.llm.enabled = True

    res = svc.explain(inp)
    assert isinstance(res, ExplanationResult)
    # original MatchResult model_dump must remain unchanged
    assert mr.model_dump() == original


def test_malformed_llm_falls_back():
    # Provider that returns wrong type
    class BadLLM:
        def explain(self, inp, timeout_seconds=5):
            return {"not": "an ExplanationResult"}

    mr = MatchResult(candidate_id="cand2", scores=ComponentScores(overall=0.4))
    inp = ExplanationInput(match_result=mr.model_dump(), component_scores=[], hybrid_metadata={}, skill_gap={}, parsed_resume={}, parsed_job={}, ranking_info={})
    svc = ExplanationService(llm_provider=BadLLM())
    svc._settings.llm.enabled = True
    res = svc.explain(inp)
    # Deterministic fallback returns ExplanationResult
    assert isinstance(res, ExplanationResult)
