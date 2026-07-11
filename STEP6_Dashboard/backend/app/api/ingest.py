from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_desktop_token
from app.core.config import get_settings
from app.db.session import get_db
from app.models import ApiToken, ScrapingJob, ScrapingLog, SyncBatch, User
from app.schemas import IngestBatchIn, IngestBatchResult
from app.services.ai import analyze_news_items_by_id
from app.services.ingest import ingest_batch


router = APIRouter(tags=["ingest"])
settings = get_settings()


@router.post("/ingest/batches", response_model=IngestBatchResult)
def create_ingest_batch(
    payload: IngestBatchIn,
    background_tasks: BackgroundTasks,
    api_token: ApiToken = Depends(get_desktop_token),
    db: Session = Depends(get_db),
) -> dict:
    result, item_ids = ingest_batch(db, payload, token_id=api_token.id)
    if settings.label_on_ingest and item_ids:
        background_tasks.add_task(analyze_news_items_by_id, item_ids)
    return result


@router.get("/ingest/batches")
def list_batches(_: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[dict]:
    batches = db.query(SyncBatch).order_by(SyncBatch.created_at.desc()).limit(100).all()
    return [
        {
            "id": batch.id,
            "batch_id": batch.batch_id,
            "total_received": batch.total_received,
            "inserted": batch.inserted_count,
            "updated": batch.updated_count,
            "skipped": batch.skipped_count,
            "failed": batch.failed_count,
            "status": batch.status,
            "created_at": batch.created_at,
        }
        for batch in batches
    ]


@router.post("/scraping-jobs")
def create_scraping_job(payload: dict, api_token: ApiToken = Depends(get_desktop_token), db: Session = Depends(get_db)) -> dict:
    job = ScrapingJob(
        client_job_id=payload.get("client_job_id"),
        status=payload.get("status", "running"),
        total_collected=payload.get("total_collected", 0),
        created_by_token_id=api_token.id,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return {"id": job.id}


@router.patch("/scraping-jobs/{job_id}")
def update_scraping_job(job_id: int, payload: dict, _: ApiToken = Depends(get_desktop_token), db: Session = Depends(get_db)) -> dict:
    job = db.get(ScrapingJob, job_id)
    if job:
        for field in ["status", "total_collected", "total_synced", "error_message", "finished_at"]:
            if field in payload:
                setattr(job, field, payload[field])
        db.add(job)
        db.commit()
    return {"ok": True}


@router.post("/scraping-jobs/{job_id}/logs")
def create_scraping_log(job_id: int, payload: dict, _: ApiToken = Depends(get_desktop_token), db: Session = Depends(get_db)) -> dict:
    db.add(ScrapingLog(job_id=job_id, level=payload.get("level", "info"), message=payload.get("message", ""), context_json=payload.get("context")))
    db.commit()
    return {"ok": True}
