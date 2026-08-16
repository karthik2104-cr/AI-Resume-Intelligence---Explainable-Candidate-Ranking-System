# Entity Intelligence & Skill Gap Analysis

## Overview

Skill normalization is shared across resumes and job descriptions. Deterministic
skill-gap analysis compares candidate skills to required/preferred JD skills
without embeddings, LLMs, or final match scores.

```
ParsedResume → EntityExtractor → Candidate Profile ─┐
                                                     ├→ Skill Gap Analysis → SkillGapResult
ParsedJobDescription → EntityExtractor → Job Profile ┘
```

## Entity Model

Defined in `src/models/entity.py`:

| Field | Purpose |
|-------|---------|
| `raw_text` | Original mention |
| `normalized_value` | Canonical vocabulary name |
| `entity_type` | programming_language, framework, cloud, soft_skill, domain, … |
| `category` | Human-readable category from `skills.yaml` |
| `source` | Where extracted (e.g. `resume.skills`, `job.required`) |
| `evidence` | Source sentence for explainability |

`EntityProfile` groups entities into `technical_skills`, `soft_skills`, and `domains`.

`CandidateProfile` and `JobProfile` are lightweight derived views used for matching:

| Profile | Contains |
|---------|----------|
| `CandidateProfile` | Normalized skills, soft skills, domains + education/experience/projects/certifications references |
| `JobProfile` | Required/preferred/mentioned skills, soft skills, domains, experience/education requirements, seniority |

## Shared Normalization

Both resume and JD parsers use:

- `configs/skills.yaml` → `skills.vocabulary`
- `src/extraction/skill_normalizer.py`

Examples:

| Raw (Resume) | Raw (JD) | Normalized |
|--------------|----------|------------|
| sklearn | scikit-learn | Scikit-learn |
| js | JavaScript | JavaScript |

Longest-match-first with word boundaries prevents Java ⊂ JavaScript false positives.

## Skill Categories

Configured in `skills.skill_categories`:

- Programming Languages, Frameworks, Databases, Cloud, DevOps
- Machine Learning, Data Science, Web Technologies, Tools

Categories map to entity types (e.g. Cloud → `cloud`, Frameworks → `framework`).

## Requirement Semantics (Critical Fix)

JD skills are classified as:

| Type | When |
|------|------|
| **required** | Explicit phrases ("must have", "required") or required qualification sections |
| **preferred** | Explicit phrases ("is a plus", "preferred") or preferred sections |
| **mentioned** | Neutral mentions in responsibilities/description without requirement language |

Example:

- "Deploy models to AWS" (Responsibilities) → **mentioned**, not required
- "Must have Python" (Requirements) → **required**
- "AWS is a plus" (Preferred) → **preferred**

Implemented in `src/extraction/requirement_classifier.py`.

## Skill Gap Analysis

`src/matching/skill_gap.py` computes set-based differences on **normalized** skills:

| Output | Meaning |
|--------|---------|
| `matched_required` | Candidate has required skill |
| `missing_required` | Required skill absent |
| `matched_preferred` / `missing_preferred` | Same for preferred |
| `mentioned_skills` | JD skills mentioned neutrally (e.g. responsibilities) |
| `additional_candidate_skills` | Candidate skills not in JD requirements |
| `required_skill_coverage` | % — **not** a final match score |

Each entry preserves `candidate_evidence` and `job_evidence`.

## Usage

```python
from src.extraction import EntityExtractor
from src.matching import compute_skill_gap
from src.parsing import ingest_and_parse_resume, parse_job_description

resume = ingest_and_parse_resume(file_bytes, "resume.pdf")
job = parse_job_description(jd_text)

candidate = EntityExtractor().build_candidate_profile(resume)
job_profile = EntityExtractor().build_job_profile(job)
gap = compute_skill_gap(resume, job, candidate, job_profile)

print(gap.required_skill_coverage)  # e.g. 66.7
print(gap.missing_required)
```

## Limitations

1. Vocabulary-bound — unknown skills won't normalize or match
2. No semantic equivalence (Python ≠ Django) — conservative by design
3. No hierarchy matching yet
4. Soft skills extracted separately; excluded from technical gap by default
5. Coverage metrics are skill-set only — not experience, education, or semantic fit

## Related

Semantic matching adds sentence embeddings while retaining this deterministic
gap layer for explainability.
