"""Contact information extraction from resume header text."""

from __future__ import annotations

import re
from dataclasses import dataclass

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)
PHONE_PATTERN = re.compile(
    r"(?:\+?\d{1,3}[\s\-.]?)?"
    r"(?:\(?\d{2,4}\)?[\s\-.]?)?"
    r"\d{3}[\s\-.]?\d{3,4}[\s\-.]?\d{3,4}"
)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
LINKEDIN_PATTERN = re.compile(r"linkedin\.com/in/[\w\-]+", re.IGNORECASE)


@dataclass
class ContactInfo:
    name: str | None = None
    email: str | None = None
    phone: str | None = None


def extract_email(text: str) -> str | None:
    match = EMAIL_PATTERN.search(text)
    return match.group(0) if match else None


def extract_phone(text: str) -> str | None:
    for match in PHONE_PATTERN.finditer(text):
        candidate = match.group(0).strip()
        digits = re.sub(r"\D", "", candidate)
        if 7 <= len(digits) <= 15:
            return candidate
    return None


def extract_name(header_lines: list[str], email: str | None, phone: str | None) -> str | None:
    """Infer candidate name from the first plausible header line."""
    for line in header_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if email and email in stripped:
            continue
        if phone and phone in stripped:
            continue
        if URL_PATTERN.search(stripped) or LINKEDIN_PATTERN.search(stripped):
            continue
        if EMAIL_PATTERN.search(stripped) or PHONE_PATTERN.search(stripped):
            continue
        if len(stripped) > 80:
            continue
        if len(stripped.split()) > 5:
            continue
        if stripped.lower().startswith(("resume", "curriculum vitae", "cv")):
            continue
        # Skip lines that look like section headers (all caps short lines)
        if stripped.isupper() and len(stripped.split()) <= 5:
            continue
        return stripped
    return None


def extract_contact_info(header_text: str) -> ContactInfo:
    """Extract name, email, and phone from the resume header block."""
    lines = header_text.splitlines()
    email = extract_email(header_text)
    phone = extract_phone(header_text)
    name = extract_name(lines, email, phone)
    return ContactInfo(name=name, email=email, phone=phone)
