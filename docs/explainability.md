# Explainability

## Overview

The explainability layer produces evidence-grounded narratives for ranked
candidates. It is **presentation-only**: explanations never modify scores,
ranking, or match metadata used for hiring decisions.

## Components

| Component | Role |
|-----------|------|
| `DeterministicExplainer` | Always available; composes strengths, gaps, and interview focus from structured evidence |
| `LLMExplainer` | Optional abstraction; disabled by default (`llm.enabled: false`) |
| `ExplanationService` | Orchestrates deterministic + optional LLM with deep-copy isolation and fallback |

## Guarantees

- No invention of skills, experience, or education
- No inference of protected attributes
- Malformed / unavailable LLM responses fall back to deterministic output
- Inputs are deep-copied so providers cannot mutate caller state

## Configuration

See `configs/config.yaml` (`llm.*`). API keys must come from environment variables.

## Limitations

- Quality depends on the structured evidence supplied by ranking / skill-gap
- Deterministic text is conservative and template-like by design
