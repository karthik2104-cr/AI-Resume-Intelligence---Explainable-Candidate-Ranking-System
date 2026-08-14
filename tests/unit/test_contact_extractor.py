"""Unit tests for contact extraction."""

from src.parsing.contact_extractor import extract_contact_info, extract_email, extract_phone


def test_extract_email():
    assert extract_email("Contact: john.smith@email.com") == "john.smith@email.com"


def test_extract_phone():
    phone = extract_phone("Phone: +1-555-123-4567")
    assert phone is not None
    assert "555" in phone


def test_extract_name_from_header():
    header = "John Smith\njohn.smith@email.com\n+1-555-123-4567"
    contact = extract_contact_info(header)
    assert contact.name == "John Smith"
    assert contact.email == "john.smith@email.com"
    assert contact.phone is not None
