"""Unit tests for text preprocessing."""

from src.preprocessing.text_cleaner import clean_text
from src.utils.config import PreprocessingConfig


def test_clean_text_removes_urls():
    text = "Visit http://example.com for details about Python."
    result = clean_text(text, PreprocessingConfig())
    assert "http" not in result
    assert "python" in result


def test_clean_text_lowercases():
    text = "Python Developer SKILLS"
    result = clean_text(text, PreprocessingConfig(lowercase=True))
    assert result == result.lower()


def test_clean_text_empty_input():
    assert clean_text("") == ""
    assert clean_text("   ") == ""


def test_clean_text_normalizes_whitespace():
    text = "Python    SQL\n\nMachine   Learning"
    result = clean_text(text, PreprocessingConfig())
    assert "  " not in result


def test_clean_text_removes_non_ascii():
    text = "Skills • Python • SQL • R"
    result = clean_text(text, PreprocessingConfig(remove_non_ascii=True))
    assert "•" not in result
    assert "python" in result
