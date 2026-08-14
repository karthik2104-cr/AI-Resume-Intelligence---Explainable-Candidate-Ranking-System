"""Split job description text into structured sections."""

from __future__ import annotations

from src.models.job import JobSection, JobSectionType
from src.parsing.jd_section_patterns import build_jd_heading_patterns, detect_jd_section_heading
from src.utils.config import JobParsingConfig, get_settings


def split_jd_into_sections(
    text: str,
    config: JobParsingConfig | None = None,
) -> tuple[str, list[JobSection]]:
    cfg = config or get_settings().job_parsing
    patterns = build_jd_heading_patterns(cfg)
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    headings = []
    for index, line in enumerate(lines):
        match = detect_jd_section_heading(line, index, patterns, cfg.max_header_line_length)
        if match:
            headings.append(match)

    if not headings:
        return text.strip(), []

    header_lines = lines[: headings[0].line_index]
    header_text = "\n".join(header_lines).strip()

    sections: list[JobSection] = []
    for i, heading in enumerate(headings):
        start_line = heading.line_index + 1
        end_line = headings[i + 1].line_index if i + 1 < len(headings) else len(lines)
        content = "\n".join(lines[start_line:end_line]).strip()
        sections.append(
            JobSection(
                section_type=heading.section_type,
                title=heading.title,
                content=content,
                start_line=start_line,
                end_line=end_line,
            )
        )
    return header_text, sections


def section_content_by_type(
    sections: list[JobSection],
    section_type: JobSectionType,
) -> str:
    parts = [s.content for s in sections if s.section_type == section_type and s.content]
    return "\n\n".join(parts).strip()
