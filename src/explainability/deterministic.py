"""Deterministic explainer: produces grounded explanations purely from structured evidence."""
from __future__ import annotations

from typing import List

from src.explainability.models import ExplanationInput, ExplanationResult, EvidenceItem


class DeterministicExplainer:
    """Generates deterministic, evidence-grounded explanations.

    The explainer must not invent skills, experience, or education items. It
    composes a concise summary, strengths, gaps, concerns and interview focus
    areas from the provided structured inputs.
    """

    def explain(self, inp: ExplanationInput) -> ExplanationResult:
        strengths: List[str] = []
        req_matches: List[str] = []
        pref_matches: List[str] = []
        gaps: List[str] = []
        concerns: List[str] = []
        interview: List[str] = []
        evidence_items: List[EvidenceItem] = []

        # skill_gap is expected as dict with lists
        sg = inp.skill_gap or {}
        if isinstance(sg, dict):
            matched_required = sg.get("matched_required") or []
            matched_preferred = sg.get("matched_preferred") or []
            missing_required = sg.get("missing_required") or []
            missing_preferred = sg.get("missing_preferred") or []

            for m in matched_required:
                # accept string or dict
                if isinstance(m, dict):
                    name = m.get("skill") or str(m)
                else:
                    name = str(m)
                req_matches.append(name)
                strengths.append(name)

            for m in matched_preferred:
                if isinstance(m, dict):
                    name = m.get("skill") or str(m)
                else:
                    name = str(m)
                pref_matches.append(name)
                strengths.append(name)

            for m in missing_required:
                if isinstance(m, dict):
                    name = m.get("skill") or str(m)
                else:
                    name = str(m)
                gaps.append(name)

            for m in missing_preferred:
                if isinstance(m, dict):
                    name = m.get("skill") or str(m)
                else:
                    name = str(m)
                gaps.append(name)

        # Component scores: list[dict] with name, score, availability
        comps = inp.component_scores or []
        comp_map = {c.get("name"): c for c in comps if isinstance(c, dict)}

        # Experience
        exp_align = None
        exp_comp = comp_map.get("experience_compatibility")
        if exp_comp and exp_comp.get("score") is not None:
            val = exp_comp.get("score")
            if val >= 0.9:
                exp_align = "Experience aligns well with requirement"
            elif val >= 0.5:
                exp_align = "Experience partially aligns with requirement"
            else:
                exp_align = "Experience likely below requirement"

        # Education
        edu_align = None
        edu_comp = comp_map.get("education_compatibility")
        if edu_comp and edu_comp.get("score") is not None:
            val = edu_comp.get("score")
            if val >= 0.9:
                edu_align = "Education matches requirement"
            elif val >= 0.5:
                edu_align = "Education partially matches requirement"
            else:
                edu_align = "Education likely below requirement"

        # Seniority
        sen_align = None
        sen_comp = comp_map.get("seniority_compatibility")
        if sen_comp and sen_comp.get("score") is not None:
            val = sen_comp.get("score")
            if val >= 0.9:
                sen_align = "Seniority aligns"
            elif val >= 0.5:
                sen_align = "Seniority roughly compatible"
            else:
                sen_align = "Seniority may be insufficient"

        # Evidence: prefer explicit evidence list
        if inp.evidence:
            for e in inp.evidence:
                if isinstance(e, EvidenceItem):
                    evidence_items.append(e)
                elif isinstance(e, dict):
                    evidence_items.append(EvidenceItem(**e))

        # also collect from parsed_resume/resume fields where present
        if inp.parsed_resume and isinstance(inp.parsed_resume, dict):
            name = inp.parsed_resume.get("name")
            if name:
                evidence_items.append(EvidenceItem(source="resume.name", text=str(name)))

        # Build summary
        summary_parts = []
        overall = None
        if inp.match_result:
            # match_result can be dict or object; expect dict
            overall = inp.match_result.get("overall_score") or None
        if strengths:
            summary_parts.append("Strong match on skills: " + ", ".join(strengths[:3]))
        if overall is not None:
            summary_parts.append(f"Overall score: {float(overall):.2f}")

        summary = "; ".join(summary_parts) if summary_parts else "No strong positive signals detected."

        # Interview focuses: propose missing required skills first
        if gaps:
            interview += [f"Ask about experience with {g}" for g in gaps[:5]]

        return ExplanationResult(
            summary=summary,
            strengths=strengths,
            required_skill_matches=req_matches,
            preferred_skill_matches=pref_matches,
            skill_gaps=gaps,
            experience_alignment=exp_align,
            education_alignment=edu_align,
            seniority_alignment=sen_align,
            concerns=concerns,
            interview_focus_areas=interview,
            evidence=evidence_items,
            explanation_source="deterministic",
        )
