# Job Description Parsing

## Overview

The JD parser transforms raw job description text into a structured `ParsedJobDescription` using deterministic, evidence-preserving heuristics. No LLM or embeddings are used.

```
Job Description text
        ↓
HeuristicJobDescriptionParser
        ↓
ParsedJobDescription
        ↓
Skill Normalization / Matching / Ranking
```

## Supported Sections

Section headings are detected via configurable aliases in `configs/skills.yaml` (`job_parsing.section_aliases`):

| Canonical | Example headings |
|-----------|------------------|
| `about` | About the Role |
| `description` | Job Description |
| `responsibilities` | What You'll Do, Key Responsibilities |
| `requirements` | Requirements, Minimum Qualifications |
| `required_qualifications` | Required Skills, Must Have |
| `preferred_qualifications` | Preferred Skills, Nice to Have |
| `qualifications` | Qualifications |
| `education` | Education |
| `experience` | Experience Requirements |
| `benefits` | Benefits |

## Required vs Preferred Classification

Classification uses **section context first**, then **sentence-level phrases**:

| Signal | Result |
|--------|--------|
| Section: Required Qualifications / Must Have | REQUIRED |
| Section: Preferred / Nice to Have | PREFERRED |
| Phrase: "must have", "required", "mandatory" | REQUIRED |
| Phrase: "is a plus", "preferred", "nice to have" | PREFERRED |

Every skill retains `evidence` (source sentence) and `source_section`.

## Skill Extraction & Normalization

Skills are extracted via a **shared vocabulary** (`configs/skills.yaml` → `skills.vocabulary`):

- Longest-match-first to avoid substring collisions (JavaScript ≠ Java)
- Word-boundary safe patterns (trailing punctuation allowed)
- Aliases normalized to canonical names (e.g., `sklearn` → `Scikit-learn`)

Technical skills and soft skills are stored separately (`required_skills`, `preferred_skills`, `soft_skills`).

## Experience Extraction

Supported patterns:

| Pattern | Example | Output |
|---------|---------|--------|
| N+ years | `3+ years` | min=3, max=None |
| Range | `2-5 years` | min=2, max=5 |
| Minimum | `at least 4 years` | min=4 |
| Entry level | `entry level`, `fresh graduates` | is_entry_level=True |

Experience context distinguishes professional vs relevant/domain experience where wording allows.

## Education Extraction

Detects degree levels via configurable aliases (`degree_aliases`):

- bachelor: B.Tech, B.E., Bachelor's, …
- master: M.Tech, MBA, Master's, …
- doctorate: Ph.D, PhD, …

Preferred education is detected via preferred phrases in the sentence.

## Seniority, Employment, Location

| Field | Source |
|-------|--------|
| `seniority_level` | Title or explicit keywords (Junior, Senior, Lead, …) |
| `employment_type` | full-time, contract, internship, … |
| `work_mode` | remote, hybrid, onsite |
| `location` | Location: … or known city names |

These are extracted only when explicitly stated — not inferred from unrelated text.

## Parsing Quality

Rule-based score (not ML confidence):

| Level | Typical signals |
|-------|-----------------|
| high | Title + sections + required skills + experience + responsibilities |
| medium | Partial structure |
| low | Minimal extraction, warnings issued |

Returns `JobParsingQuality` with `score`, `level`, and `warnings`.

## Usage

```python
from src.parsing import parse_job_description

parsed = parse_job_description(jd_text)
print(parsed.title)
print(parsed.all_required_skill_names)
print(parsed.required_skills[0].evidence)
```

## Known Limitations

1. Heuristic section detection — non-standard layouts may miss sections
2. Skill vocabulary is finite — unknown technologies won't be normalized
3. Domain classification only when explicit domain terms appear in config
4. Location extraction limited to labeled lines and a small city list
5. Responsibilities require bullet-like or line-separated format
6. No LLM fallback for ambiguous requirements

## Evidence Preservation

All skill requirements include:

```python
SkillRequirement(
    raw_skill="sklearn",
    normalized_skill="Scikit-learn",
    requirement_type="required",
    evidence="Experience building models using Python and sklearn.",
    source_section="Required Qualifications",
)
```

This supports future explainability without fabricating reasons.
