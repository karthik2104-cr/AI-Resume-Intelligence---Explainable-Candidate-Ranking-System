from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional

from src.api.schemas import ScreenResponse, CandidateScore
from src.services.document_pipeline import parse_resume_bytes, parse_job
from src.services.resume_screening_service import ScreeningService
from src.utils.config import get_settings
from src.ingestion.errors import (
    UnsupportedFileTypeError,
    FileTooLargeError,
    CorruptedFileError,
    EmptyDocumentError,
)

router = APIRouter()


@router.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@router.post("/screen", response_model=ScreenResponse)
async def screen(
    job_text: str = Form(...),
    job_title: Optional[str] = Form(None),
    resumes: Optional[List[UploadFile]] = File(None),
    top_k: Optional[int] = Form(None),
    explain: Optional[bool] = Form(False),
):
    # Validate before constructing ScreeningService (avoids loading embeddings on bad requests)
    if not job_text or not job_text.strip():
        raise HTTPException(status_code=400, detail="job_text cannot be empty")

    if not resumes:
        raise HTTPException(status_code=400, detail="At least one resume file must be uploaded")

    if top_k is not None and top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be > 0")

    settings = get_settings()
    effective_top_k = top_k if top_k is not None else settings.retrieval.top_k

    parsed_resumes = []
    candidate_ids = []

    for i, upload in enumerate(resumes):
        filename = upload.filename or f"uploaded_{i}"
        content = await upload.read()
        try:
            parsed = parse_resume_bytes(content, filename, strip_pii=True)
        except UnsupportedFileTypeError:
            raise HTTPException(status_code=415, detail=f"Unsupported file type for {filename}")
        except FileTooLargeError:
            raise HTTPException(status_code=413, detail=f"File too large: {filename}")
        except CorruptedFileError:
            raise HTTPException(status_code=422, detail=f"Corrupted or unreadable file: {filename}")
        except EmptyDocumentError:
            raise HTTPException(status_code=400, detail=f"Empty document: {filename}")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to ingest {filename}: {exc}")

        parsed_resumes.append(parsed)
        candidate_ids.append(f"cand_{i}")

    service = ScreeningService()
    job = parse_job(job_text, title=job_title)
    result = service.screen(job, parsed_resumes, candidate_ids=candidate_ids, top_k=effective_top_k)

    ranking = result.get("ranking_result")
    explanations = result.get("explanations") if explain else None

    if not ranking:
        return ScreenResponse(
            job_title=job_title,
            top_k=effective_top_k,
            candidates=[],
            explanations=None,
        )

    candidates_out = []
    for entry in ranking.entries:
        mr = entry.match_result
        meta = mr.metadata or {}
        hybrid = meta.get("hybrid") or {}
        skill_gap = hybrid.get("skill_gap") or {}

        matched = meta.get("matched_skills")
        if matched is None:
            matched = [
                e.get("skill") if isinstance(e, dict) else str(e)
                for e in (skill_gap.get("matched_required") or [])
                + (skill_gap.get("matched_preferred") or [])
            ]

        missing_req = meta.get("missing_required_skills")
        if missing_req is None:
            missing_req = [
                e.get("skill") if isinstance(e, dict) else str(e)
                for e in (skill_gap.get("missing_required") or [])
            ]

        missing_pref = meta.get("missing_preferred_skills")
        if missing_pref is None:
            missing_pref = [
                e.get("skill") if isinstance(e, dict) else str(e)
                for e in (skill_gap.get("missing_preferred") or [])
            ]

        comp_scores = {
            "semantic": float(getattr(mr.scores, "semantic", None) or 0.0),
            "skill": float(getattr(mr.scores, "skill", None) or 0.0),
            "experience": float(getattr(mr.scores, "experience", None) or 0.0),
            "education": float(getattr(mr.scores, "education", None) or 0.0),
        }
        retrieval_meta = meta.get("retrieval", {})
        candidates_out.append(
            CandidateScore(
                candidate_id=entry.candidate_id,
                overall_score=entry.overall_score,
                component_scores=comp_scores,
                matched_skills=matched,
                missing_required_skills=missing_req,
                missing_preferred_skills=missing_pref,
                parsing_quality=meta.get("parsing_quality"),
                retrieval_similarity=retrieval_meta.get("similarity"),
            )
        )

    return ScreenResponse(
        job_title=job_title,
        top_k=effective_top_k,
        candidates=candidates_out,
        explanations=explanations,
    )
