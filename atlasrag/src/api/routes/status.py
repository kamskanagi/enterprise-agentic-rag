"""Status Endpoint

GET /status/{job_id} — Get the status of a document ingestion job.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from atlasrag.src.api.dependencies import get_ingestion_pipeline
from atlasrag.src.api.models import JobStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/status/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str, pipeline=Depends(get_ingestion_pipeline)):
    """Get the current status of an ingestion job."""
    job = pipeline.get_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    return JobStatusResponse(
        job_id=job.job_id,
        document_id=job.document_id,
        status=job.status,
        total_chunks=job.total_chunks,
        stored_chunks=job.stored_chunks,
        progress=job.progress,
        error_message=job.error_message,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )
