"""Streamlit demo for AI Resume Screening V2.

Presentation layer only — all screening logic is delegated to ScreeningService.
"""
from __future__ import annotations

import streamlit as st

from src.ingestion.factory import IngestionFactory
from src.parsing.job_parser import parse_job_description
from src.parsing.resume_parser import HeuristicResumeParser
from src.services.resume_screening_service import ScreeningService
from src.utils.config import get_settings


def mask_name(name: str | None) -> str:
    if not name:
        return "Candidate"
    parts = name.split()
    return parts[0]


def score_bar(label: str, value: float | None) -> None:
    pct = max(0.0, min(1.0, float(value or 0.0)))
    st.markdown(f"**{label}**")
    st.progress(pct, text=f"{pct * 100:.0f}%")


def render_explanation(explanation: dict) -> None:
    summary = explanation.get("summary")
    if summary:
        st.markdown(summary)

    strengths = explanation.get("strengths") or []
    if strengths:
        st.markdown("**Strengths**")
        for item in strengths[:5]:
            st.markdown(f"- {item}")

    gaps = explanation.get("skill_gaps") or []
    if gaps:
        st.markdown("**Skill gaps**")
        for item in gaps[:8]:
            st.markdown(f"- {item}")

    focus = explanation.get("interview_focus_areas") or []
    if focus:
        st.markdown("**Interview focus**")
        for item in focus[:5]:
            st.markdown(f"- {item}")


def main() -> None:
    st.set_page_config(
        page_title="AI Resume Screening",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("AI Resume Screening")
    st.caption("Intelligent candidate matching and explainability")

    with st.sidebar:
        st.header("Settings")
        top_k = st.number_input("Retrieval top-K", min_value=1, max_value=50, value=10)
        show_explanations = st.checkbox("Show explanations", value=True)

    col_jd, col_resumes = st.columns(2)

    with col_jd:
        st.subheader("Job Description")
        job_text = st.text_area(
            "Paste job description",
            height=220,
            placeholder="Paste the full job description here…",
            label_visibility="collapsed",
        )
        job_title = st.text_input("Job title (optional)", placeholder="e.g. Senior ML Engineer")

    with col_resumes:
        st.subheader("Candidate Resumes")
        uploaded = st.file_uploader(
            "Upload resumes (PDF, DOCX, TXT)",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt"],
            label_visibility="collapsed",
        )
        if uploaded:
            st.caption(f"{len(uploaded)} file(s) selected")

    run = st.button("Screen Candidates", type="primary", use_container_width=True)

    if not run:
        return

    if not job_text or not job_text.strip():
        st.error("Please paste a job description.")
        return
    if not uploaded:
        st.error("Please upload at least one resume.")
        return

    settings = get_settings()
    settings.retrieval.top_k = int(top_k)

    ingester = IngestionFactory()
    parser = HeuristicResumeParser()
    parsed_resumes = []
    candidate_ids = []

    with st.spinner("Parsing resumes and running screening pipeline…"):
        for i, up in enumerate(uploaded):
            try:
                raw = up.read()
                doc = ingester.ingest(raw, up.name)
                parsed = parser.parse(doc)
                parsed.email = None
                parsed.phone = None
                parsed_resumes.append(parsed)
                candidate_ids.append(f"cand_{i}")
            except Exception as exc:
                st.error(f"Could not process **{up.name}**: {exc}")
                return

        job = parse_job_description(job_text, title=job_title or None)
        svc = ScreeningService()
        out = svc.screen(job, parsed_resumes, candidate_ids=candidate_ids)

    ranking = out.get("ranking_result")
    explanations = out.get("explanations") if show_explanations else None
    retrieval_results = out.get("retrieval_results") or []

    if not ranking or not ranking.entries:
        st.warning("No candidates were ranked. Check parsing quality or upload more resumes.")
        return

    st.divider()
    st.subheader("Screening Results")
    st.caption(
        f"Retrieved {len(retrieval_results)} candidate(s) · "
        f"Ranked {len(ranking.entries)} · "
        "Scores are hybrid matches (not retrieval-only)."
    )

    for entry in ranking.entries:
        mr = entry.match_result
        display_name = mask_name(mr.candidate_name)
        overall_pct = entry.overall_score * 100

        with st.container(border=True):
            header_left, header_right = st.columns([3, 1])
            with header_left:
                st.markdown(f"### #{entry.rank} · {display_name}")
                st.caption(f"ID: {entry.candidate_id}")
            with header_right:
                st.metric("Overall Match", f"{overall_pct:.0f}%")

            c1, c2, c3 = st.columns(3)
            with c1:
                score_bar("Skills Match", getattr(mr.scores, "skill", 0.0))
            with c2:
                score_bar("Semantic Match", getattr(mr.scores, "semantic", 0.0))
            with c3:
                score_bar("Experience", getattr(mr.scores, "experience", 0.0))

            meta = mr.metadata or {}
            matched = meta.get("matched_skills") or []
            missing_req = meta.get("missing_required_skills") or []
            retrieval = meta.get("retrieval") or {}

            detail_cols = st.columns(2)
            with detail_cols[0]:
                if matched:
                    st.markdown("**Matched skills**")
                    st.write(", ".join(matched[:12]))
            with detail_cols[1]:
                if missing_req:
                    st.markdown("**Missing required skills**")
                    st.write(", ".join(missing_req[:12]))
                if retrieval.get("similarity") is not None:
                    st.caption(
                        f"Retrieval similarity: {retrieval['similarity']:.3f} "
                        f"(rank {retrieval.get('rank', '—')})"
                    )

            if show_explanations and explanations and entry.candidate_id in explanations:
                with st.expander("View explanation"):
                    render_explanation(explanations[entry.candidate_id])


if __name__ == "__main__":
    main()
