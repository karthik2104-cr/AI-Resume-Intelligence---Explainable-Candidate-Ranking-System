import pytest

from src.models.document import Document, DocumentSourceType
from src.parsing.resume_parser import HeuristicResumeParser


def make_doc(text: str) -> Document:
    return Document(source_type=DocumentSourceType.TXT, raw_text=text, filename="test.txt")


def test_extracts_achievements_and_languages_and_soft_skills():
    text = """
    John Doe
    Summary:
    Experienced machine learning engineer.

    Skills:
    - Python, scikit-learn, leadership, communication

    Achievements:
    - Won Best Paper Award at XYZ 2020
    - Reduced latency by 40% across service

    Languages:
    - English, Spanish

    Experience:
    Software Engineer at Acme Corp | Jan 2018 - Present
    - Built predictive models using scikit-learn
    """
    doc = make_doc(text)
    parser = HeuristicResumeParser()
    parsed = parser.parse(doc)

    assert "Won Best Paper Award at XYZ 2020" in parsed.achievements
    assert "Reduced latency by 40% across service" in parsed.achievements
    assert "English" in parsed.languages
    assert "Spanish" in parsed.languages
    # soft skills normalized should include canonical names
    assert "communication" in [s.lower() for s in parsed.soft_skills]
    assert "leadership" in [s.lower() for s in parsed.soft_skills]


def test_detects_employment_and_location_and_notice():
    text = """
    Jane Smith
    Location: Bengaluru, India
    Employment Type: Full-time
    Notice period: 2 months

    Experience:
    Senior Data Scientist at DataCorp | 2016 - Present
    - Led team in deploying ML models
    """
    doc = make_doc(text)
    parser = HeuristicResumeParser()
    parsed = parser.parse(doc)

    assert parsed.employment_type is not None
    assert parsed.location is not None
    assert parsed.notice_period is not None
    assert parsed.location.lower().startswith("bengaluru")
    assert "2 months" in parsed.notice_period.lower()
