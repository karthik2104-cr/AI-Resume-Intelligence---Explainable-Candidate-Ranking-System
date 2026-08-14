"""Shared skill vocabulary, normalization, and boundary-safe extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from src.utils.config import V2_ROOT, get_settings

SKILLS_CONFIG_PATH = V2_ROOT / "configs" / "skills.yaml"
BOUNDARY_LEFT = r"(?<![A-Za-z0-9_+#-])"
BOUNDARY_RIGHT = r"(?![A-Za-z0-9_+#-])"


@dataclass(frozen=True)
class SkillMatch:
    canonical: str
    matched_text: str
    start: int
    end: int


def _load_vocabulary_raw() -> dict[str, list[str]]:
    settings = get_settings()
    if settings.skills.vocabulary:
        return settings.skills.vocabulary

    if SKILLS_CONFIG_PATH.exists():
        with SKILLS_CONFIG_PATH.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data.get("skills", {}).get("vocabulary", {})
    return {}


@lru_cache(maxsize=1)
def get_vocabulary_entries() -> list[tuple[str, str]]:
    """Return (canonical, form) pairs sorted longest-form first."""
    vocabulary = _load_vocabulary_raw()
    entries: list[tuple[str, str]] = []
    for canonical, aliases in vocabulary.items():
        forms = {canonical.lower(), canonical}
        for alias in aliases:
            forms.add(alias)
            forms.add(alias.lower())
        for form in forms:
            if form.strip():
                entries.append((canonical, form))
    entries.sort(key=lambda item: len(item[1]), reverse=True)
    return entries


def normalize_skill(raw_skill: str) -> str:
    """Map a raw skill token to its canonical vocabulary name if known."""
    raw_lower = raw_skill.strip().lower()
    for canonical, form in get_vocabulary_entries():
        if raw_lower == form.lower():
            return canonical
    return raw_skill.strip()


def _compile_pattern(form: str) -> re.Pattern[str]:
    escaped = re.escape(form)
    return re.compile(rf"{BOUNDARY_LEFT}{escaped}{BOUNDARY_RIGHT}", re.IGNORECASE)


def find_skills_in_text(text: str) -> list[SkillMatch]:
    """
    Extract skills using longest-match-first with word-boundary safety.

    Prevents substring false positives (e.g., Java inside JavaScript).
    """
    matches: list[SkillMatch] = []
    occupied: list[tuple[int, int]] = []

    for canonical, form in get_vocabulary_entries():
        pattern = _compile_pattern(form)
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if _overlaps(span, occupied):
                continue
            occupied.append(span)
            matches.append(
                SkillMatch(
                    canonical=canonical,
                    matched_text=match.group(0),
                    start=span[0],
                    end=span[1],
                )
            )

    matches.sort(key=lambda m: m.start)
    return matches


def _overlaps(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    for start, end in occupied:
        if not (span[1] <= start or span[0] >= end):
            return True
    return False


def dedupe_skill_matches(matches: list[SkillMatch]) -> list[SkillMatch]:
    """Keep first occurrence per canonical skill."""
    seen: set[str] = set()
    unique: list[SkillMatch] = []
    for match in matches:
        key = match.canonical.lower()
        if key not in seen:
            seen.add(key)
            unique.append(match)
    return unique


def get_skill_category(canonical: str) -> str | None:
    """Return configured category label for a canonical skill."""
    settings = get_settings()
    return settings.skills.skill_categories.get(canonical)


def entity_type_for_category(category: str | None) -> str:
    """Map skill category label to entity type slug."""
    if not category:
        return "technical_skill"
    mapping = {
        "Programming Languages": "programming_language",
        "Frameworks": "framework",
        "Libraries": "library",
        "Databases": "database",
        "Cloud": "cloud",
        "DevOps": "tool",
        "Machine Learning": "technology",
        "Data Science": "technology",
        "Web Technologies": "technology",
        "Tools": "tool",
    }
    return mapping.get(category, "technical_skill")


def normalize_skills_list(raw_skills: list[str]) -> list[str]:
    """Normalize a list of raw skill tokens, deduplicated."""
    seen: set[str] = set()
    result: list[str] = []
    for raw in raw_skills:
        normalized = normalize_skill(raw)
        key = normalized.lower()
        if key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def extract_sentence_evidence(text: str, start: int, end: int) -> str:
    """Return the sentence containing a skill match for explainability."""
    sentence_breaks = re.split(r"(?<=[.!?])\s+", text)
    position = 0
    for sentence in sentence_breaks:
        sentence_start = text.find(sentence, position)
        if sentence_start == -1:
            break
        sentence_end = sentence_start + len(sentence)
        if sentence_start <= start < sentence_end:
            return sentence.strip()
        position = sentence_end
    # Fallback: surrounding line
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end].strip()
