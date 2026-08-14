"""Unit tests for entry extractors."""

from src.parsing.entry_extractors import (
    extract_education_entries,
    extract_experience_entries,
    extract_skills,
    estimate_years_experience,
)


def test_extract_skills_from_comma_list():
    content = "Python, Django, PostgreSQL, REST APIs"
    skills = extract_skills(content)
    assert "Python" in skills
    assert "Django" in skills
    assert len(skills) >= 4


def test_extract_skills_from_bullets():
    content = "• Python\n• SQL\n• Machine Learning"
    skills = extract_skills(content)
    assert "Python" in skills
    assert "Machine Learning" in skills


def test_extract_experience_with_dates():
    content = """
Senior Developer at TechCorp
Jan 2020 - Present
Built APIs with Django
""".strip()
    entries = extract_experience_entries(content)
    assert len(entries) == 1
    assert entries[0].title is not None
    assert "TechCorp" in (entries[0].organization or "")
    assert entries[0].start_date is not None


def test_extract_internship_flag():
    content = """
Software Engineering Intern at BigCo
Jun 2023 - Aug 2023
Built tools
""".strip()
    entries = extract_experience_entries(content)
    assert entries[0].is_internship is True


def test_extract_education_degree():
    content = """
B.Tech Computer Science
State University
2015 - 2019
""".strip()
    entries = extract_education_entries(content)
    assert len(entries) == 1
    assert entries[0].degree is not None
    assert "B" in entries[0].degree


def test_estimate_years_experience():
    content = """
Developer at Co
Jan 2018 - Dec 2020
Work
""".strip()
    entries = extract_experience_entries(content)
    years = estimate_years_experience(entries)
    assert years is not None
    assert years >= 2.0
