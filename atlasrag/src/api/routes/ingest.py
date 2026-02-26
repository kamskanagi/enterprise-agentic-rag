"""Ingest Endpoint

POST /ingest — Upload a document for ingestion into the knowledge base.
"""

import logging
import os
import tempfile

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile

from atlasrag.src.api.dependencies import get_current_settings, get_ingestion_pipeline
from atlasrag.src.api.models import IngestResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def _run_ingestion(pipeline, file_path: str, job_id: str):
    """Background task that runs the ingestion pipeline."""
    try:
        pipeline.ingest_file(file_path)
    except Exception as e:
        logger.error("Background ingestion failed for job %s: %s", job_id, e)
    finally:
        # Clean up temp file
        try:
            os.unlink(file_path)
        except OSError:
            pass


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    pipeline=Depends(get_ingestion_pipeline),
    settings=Depends(get_current_settings),
):
    """Upload and ingest a document into the knowledge base.

    The file is saved to a temporary location and ingested asynchronously
    via a background task. Returns a job ID for tracking progress.
    """
    # Validate file extension
    if file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in settings.supported_file_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Supported: {settings.supported_file_types}",
            )

    # Save uploaded file to temp location
    try:
        suffix = os.path.splitext(file.filename or "upload.txt")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    # Create a job record synchronously, then run ingestion in background
    import uuid

    job_id = str(uuid.uuid4())

    from datetime import datetime

    from atlasrag.src.ingestion.models import IngestionJob

    pipeline.jobs[job_id] = IngestionJob(
        job_id=job_id,
        document_id=str(uuid.uuid4()),
        status="pending",
        started_at=datetime.now(),
        progress=0,
    )

    background_tasks.add_task(_run_ingestion, pipeline, tmp_path, job_id)

    logger.info("Ingestion job %s queued for file: %s", job_id, file.filename)

    return IngestResponse(
        job_id=job_id,
        status="pending",
        message=f"Document '{file.filename}' queued for ingestion",
    )
