Explainability and LLM-grounded Explanations

Overview
--------
The explainability subsystem provides evidence-grounded, deterministic
explanations for candidate matches and an optional LLM-based synthesizer that
improves readability while being strictly grounded in supplied evidence.

Key principles
--------------
- The deterministic engine is authoritative for facts and evidence.
- The LLM may only rephrase or synthesize supplied evidence; it must NOT
  modify scores or produce new facts.
- If the LLM is unavailable, fails, times out, or produces malformed output,
  the system falls back to the deterministic explainer.
- No protected attributes are inferred or used.

Models
------
- ExplanationInput: structured input containing match_result, component_scores,
  skill_gap, parsed_resume, parsed_job, evidence, and ranking metadata.
- ExplanationResult: strongly-typed Pydantic model describing summary,
  strengths, skill matches, gaps, alignments, concerns, interview focus areas,
  evidence, and explanation_source (deterministic or llm).

LLM usage rules
---------------
When the LLM is used, the prompt must:
- instruct the model to use only provided evidence
- prohibit invention of skills, experience, education, or sensitive attributes
- prohibit changing scores or rankings
- instruct to output JSON matching the ExplanationResult schema

Fallback and validation
-----------------------
LLM outputs are validated against ExplanationResult. If validation fails or the
provider errors, deterministic explainer output is used instead.

Configuration
-------------
Configure LLM usage in configs/config.yaml (llm.enabled, provider, model,
timeout_seconds, fallback_to_deterministic). API keys must be provided via
environment variables and are not stored in config files.

Limitations
-----------
- LLM-based explanations rely on correct and sufficient supplied evidence.
- Deterministic explainer produces conservative, factual statements only.
- This layer is presentation/analysis only; scores and ranking remain the
  authoritative outputs from the hybrid scoring engine.
