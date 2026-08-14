"""Unit tests for skill normalizer false-positive prevention."""

from src.extraction.skill_normalizer import find_skills_in_text, normalize_skill


def test_javascript_not_java():
    text = "Experience working with JavaScript applications"
    matches = find_skills_in_text(text)
    canonical = {m.canonical for m in matches}
    assert "JavaScript" in canonical
    assert "Java" not in canonical


def test_java_and_javascript_together():
    text = "Must have Java for backend and JavaScript for frontend"
    matches = find_skills_in_text(text)
    canonical = {m.canonical for m in matches}
    assert "Java" in canonical
    assert "JavaScript" in canonical


def test_cpp_not_c():
    text = "Experience with C++ programming"
    matches = find_skills_in_text(text)
    canonical = {m.canonical for m in matches}
    assert "C++" in canonical
    assert "C" not in canonical


def test_sklearn_alias_normalization():
    assert normalize_skill("sklearn") == "Scikit-learn"
    assert normalize_skill("scikit-learn") == "Scikit-learn"


def test_js_alias():
    assert normalize_skill("js") == "JavaScript"
