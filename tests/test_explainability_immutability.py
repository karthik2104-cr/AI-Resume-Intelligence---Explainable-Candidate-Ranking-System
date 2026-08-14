import copy
from src.explainability.deterministic import DeterministicExplainer
from src.explainability.models import ExplanationInput


def test_explanation_does_not_modify_scores():
    expl = DeterministicExplainer()
    match_result = {"overall_score": 0.9, "scores": {"semantic": 0.8}}
    comp_scores = [{"name": "semantic_similarity", "score": 0.8}]
    inp = ExplanationInput(match_result=match_result, component_scores=comp_scores)
    before = copy.deepcopy(inp.match_result)
    res = expl.explain(inp)
    assert inp.match_result == before, "Explanation must not modify match_result"


def test_explanation_does_not_modify_component_scores():
    expl = DeterministicExplainer()
    comp_scores = [{"name": "semantic_similarity", "score": 0.8}]
    inp = ExplanationInput(match_result={}, component_scores=comp_scores)
    before = copy.deepcopy(inp.component_scores)
    res = expl.explain(inp)
    assert inp.component_scores == before, "Explanation must not modify component_scores"
