"""Split resume text into labeled sections."""

from __future__ import annotations

from src.models.resume import ResumeSection, ResumeSectionType
from src.parsing.section_patterns import (
    SectionHeadingMatch,
    build_heading_patterns,
    detect_section_heading,
)
from src.utils.config import ParsingConfig, get_settings


def split_into_sections(
    text: str,
    config: ParsingConfig | None = None,
) -> tuple[str, list[ResumeSection]]:
    """
    Split resume text into a header block and structured sections.

    Returns:
        header_text: content before the first detected section heading
        sections: ordered list of ResumeSection objects
    """
    cfg = config or get_settings().parsing
    patterns = build_heading_patterns(cfg)
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    headings: list[SectionHeadingMatch] = []
    for index, line in enumerate(lines):
        match = detect_section_heading(line, index, patterns, cfg.max_header_line_length)
        if match:
            headings.append(match)

    if not headings:
        return text.strip(), []

    header_lines = lines[: headings[0].line_index]
    header_text = "\n".join(header_lines).strip()

    sections: list[ResumeSection] = []
    for i, heading in enumerate(headings):
        start_line = heading.line_index + 1
        end_line = headings[i + 1].line_index if i + 1 < len(headings) else len(lines)
        content = "\n".join(lines[start_line:end_line]).strip()
        sections.append(
            ResumeSection(
                section_type=heading.section_type,
                title=heading.title,
                content=content,
                start_line=start_line,
                end_line=end_line,
            )
        )

    return header_text, sections


def section_content_by_type(
    sections: list[ResumeSection],
    section_type: ResumeSectionType,
) -> str:
    """Concatenate content from all sections of a given type."""
    parts = [s.content for s in sections if s.section_type == section_type and s.content]
    return "\n\n".join(parts).strip()
