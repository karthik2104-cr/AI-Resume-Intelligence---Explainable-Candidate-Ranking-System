from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional

from src.api.schemas import ScreenRequest, ScreenResponse, CandidateScore
from src.ingestion.factory import IngestionFactory
from src.parsing.resume_parser import HeuristicResumeParser
from src.parsing.job_parser import parse_job_description
from src.services.resume_screening_service import ScreeningService
from src.utils.config import get_settings

router = APIRouter()


def get_screening_service() -> ScreeningService:
    # create a new ScreeningService instance using default retriever
    return ScreeningService()


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
    service: ScreeningService = Depends(get_screening_service),
):
    # Basic validation
    if not job_text or not job_text.strip():
        raise HTTPException(status_code=400, detail="job_text cannot be empty")

    if not resumes or len(resumes) == 0:
        raise HTTPException(status_code=400, detail="At least one resume file must be uploaded")

    if top_k is not None and top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be > 0")

    settings = get_settings()

    ingester = IngestionFactory()
    parser = HeuristicResumeParser()

    parsed_resumes = []
    candidate_ids = []

    for i, upload in enumerate(resumes):
        filename = upload.filename or f"uploaded_{i}"
        content = await upload.read()
        try:
            doc = ingester.ingest(content, filename=filename)
        except Exception as exc:
            # Map domain-specific ingestion errors to appropriate HTTP responses
            from src.ingestion.errors import (
                UnsupportedFileTypeError,
                FileTooLargeError,
                CorruptedFileError,
                EmptyDocumentError,
            )

            if isinstance(exc, UnsupportedFileTypeError):
                raise HTTPException(status_code=415, detail=f"Unsupported file type for {filename}")
            if isinstance(exc, FileTooLargeError):
                raise HTTPException(status_code=413, detail=f"File too large: {filename}")
            if isinstance(exc, (CorruptedFileError,)):
                raise HTTPException(status_code=422, detail=f"Corrupted or unreadable file: {filename}")
            if isinstance(exc, EmptyDocumentError):
                raise HTTPException(status_code=400, detail=f"Empty document: {filename}")

            # Generic ingestion error
            raise HTTPException(status_code=400, detail=f"Failed to ingest {filename}: {exc}")

        parsed = parser.parse(doc)
        # clear sensitive fields before passing through API response
        parsed.email = None
        parsed.phone = None
        parsed_resumes.append(parsed)
        candidate_ids.append(f"cand_{i}")

    # Parse job description
    job = parse_job_description(job_text, title=job_title)

    # Optionally override retrieval top_k via request
    if top_k is not None:
        settings.retrieval.top_k = top_k

    result = service.screen(job, parsed_resumes, candidate_ids=candidate_ids)

    ranking = result.get("ranking_result")
    retrieval_results = result.get("retrieval_results", [])
    explanations = result.get("explanations") if explain else None

    if not ranking:
        return ScreenResponse(job_title=job_title, top_k=settings.retrieval.top_k, candidates=[], explanations=None)

    candidates_out = []
    for entry in ranking.entries:
        mr = entry.match_result
        comp_scores = {
            "semantic": float(getattr(mr.scores, "semantic", None) or 0.0),
            "skill": float(getattr(mr.scores, "skill", None) or 0.0),
            "experience": float(getattr(mr.scores, "experience", None) or 0.0),
        }
        retrieval_meta = mr.metadata.get("retrieval", {}) if mr.metadata else {}
        cs = CandidateScore(
            candidate_id=entry.candidate_id,
            overall_score=entry.overall_score,
            component_scores=comp_scores,
            matched_skills=mr.metadata.get("matched_skills", []),
            missing_required_skills=mr.metadata.get("missing_required_skills", []),
            missing_preferred_skills=mr.metadata.get("missing_preferred_skills", []),
            parsing_quality=mr.metadata.get("parsing_quality", None),
            retrieval_similarity=retrieval_meta.get("similarity"),
        )
        candidates_out.append(cs)

    return ScreenResponse(job_title=job_title, top_k=settings.retrieval.top_k, candidates=candidates_out, explanations=explanations)
