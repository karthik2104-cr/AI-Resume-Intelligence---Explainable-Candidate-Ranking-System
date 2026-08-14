"""Unit tests for section heading detection."""

from src.parsing.section_patterns import detect_section_heading, normalize_heading_line
from src.parsing.section_patterns import build_heading_patterns
from src.models.resume import ResumeSectionType
from src.utils.config import ParsingConfig


def test_normalize_markdown_heading():
    assert normalize_heading_line("## Skills") == "Skills"


def test_detect_skills_heading():
    patterns = build_heading_patterns()
    match = detect_section_heading("SKILLS", 5, patterns, max_header_line_length=60)
    assert match is not None
    assert match.section_type == ResumeSectionType.SKILLS


def test_detect_experience_with_colon():
    patterns = build_heading_patterns()
    match = detect_section_heading("Work Experience:", 3, patterns, max_header_line_length=60)
    assert match is not None
    assert match.section_type == ResumeSectionType.EXPERIENCE


def test_non_heading_line_returns_none():
    patterns = build_heading_patterns()
    long_line = "Built REST APIs with Django and PostgreSQL for enterprise clients worldwide"
    match = detect_section_heading(long_line, 10, patterns, max_header_line_length=60)
    assert match is None
