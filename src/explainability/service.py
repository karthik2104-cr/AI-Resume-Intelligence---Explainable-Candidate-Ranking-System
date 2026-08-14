"""Explanation service orchestrates deterministic and LLM explainers with validation and fallback."""
from __future__ import annotations

from typing import Optional

from src.explainability.models import ExplanationInput, ExplanationResult
from src.explainability.deterministic import DeterministicExplainer
from src.explainability.llm import LLMExplainer, MockLLMProvider
from src.utils.config import get_settings


import copy


class ExplanationService:
    def __init__(self, llm_provider: Optional[LLMExplainer] | None = None):
        self._settings = get_settings()
        self._deterministic = DeterministicExplainer()
        # use provided provider or a mock placeholder; real provider chosen when settings enabled
        self._llm_provider = llm_provider

    def explain(self, inp: ExplanationInput) -> ExplanationResult:
        # Always validate input is structured and contains match_result/component_scores
        # Do not allow explanations to modify scores — operate on deep copies for safety
        det = self._deterministic

        # Use a deep copy for any external provider to prevent mutation of caller data
        inp_for_provider = copy.deepcopy(inp)
        inp_for_deterministic = copy.deepcopy(inp)

        if not self._settings.llm.enabled or not self._llm_provider:
            return det.explain(inp_for_deterministic)

        # Attempt LLM explanation with strict grounding; fall back to deterministic
        try:
            # Call LLM with timeout using a deep copy
            result = self._llm_provider.explain(inp_for_provider, timeout_seconds=self._settings.llm.timeout_seconds)
            # Validate result type
            if not isinstance(result, ExplanationResult):
                # malformed
                return det.explain(inp_for_deterministic)
            # Always preserve that explanations are presentation only — do not modify input
            return result
        except Exception:
            # provider failure, timeout, or malformed output
            return det.explain(inp_for_deterministic)
