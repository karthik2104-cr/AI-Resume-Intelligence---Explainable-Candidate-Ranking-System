"""Streamlit demo for AI Resume Screening V2.

Presentation layer only — all screening logic is delegated to ScreeningService.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when launched via `streamlit run app/...`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.services.document_pipeline import parse_resume_bytes, parse_job
from src.services.resume_screening_service import ScreeningService


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


def _skills_from_entry(meta: dict) -> tuple[list, list]:
    matched = meta.get("matched_skills")
    missing_req = meta.get("missing_required_skills")
    if matched is not None and missing_req is not None:
        return matched, missing_req

    skill_gap = (meta.get("hybrid") or {}).get("skill_gap") or {}
    if matched is None:
        matched = [
            e.get("skill") if isinstance(e, dict) else str(e)
            for e in (skill_gap.get("matched_required") or [])
            + (skill_gap.get("matched_preferred") or [])
        ]
    if missing_req is None:
        missing_req = [
            e.get("skill") if isinstance(e, dict) else str(e)
            for e in (skill_gap.get("missing_required") or [])
        ]
    return matched or [], missing_req or []


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

    parsed_resumes = []
    candidate_ids = []

    with st.spinner("Parsing resumes and running screening pipeline…"):
        for i, up in enumerate(uploaded):
            try:
                parsed = parse_resume_bytes(up.read(), up.name, strip_pii=True)
                parsed_resumes.append(parsed)
                candidate_ids.append(f"cand_{i}")
            except Exception as exc:
                st.error(f"Could not process **{up.name}**: {exc}")
                return

        job = parse_job(job_text, title=job_title or None)
        svc = ScreeningService()
        out = svc.screen(job, parsed_resumes, candidate_ids=candidate_ids, top_k=int(top_k))

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
            matched, missing_req = _skills_from_entry(meta)
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
