from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models import Report, User
from app.schemas import ReportCreate, ReportRead
from app.services.reports import generate_report


router = APIRouter(prefix="/reports", tags=["reports"])


@router.post("", response_model=ReportRead)
def create_report(payload: ReportCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> Report:
    fmt = "xlsx" if payload.format == "excel" else payload.format
    if fmt not in {"csv", "xlsx", "pdf"}:
        raise HTTPException(status_code=400, detail="Unsupported format")
    report = Report(
        report_type=payload.report_type,
        date_from=payload.date_from,
        date_to=payload.date_to,
        format=fmt,
        generated_by_user_id=user.id,
        filters_json=payload.filters,
        status="running",
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    report.file_path = generate_report(db, report.id, fmt, payload.date_from, payload.date_to)
    report.status = "completed"
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


@router.get("", response_model=list[ReportRead])
def list_reports(_: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[Report]:
    return db.query(Report).order_by(Report.created_at.desc()).limit(100).all()


@router.get("/{report_id}/download")
def download_report(report_id: int, _: User = Depends(get_current_user), db: Session = Depends(get_db)) -> FileResponse:
    report = db.get(Report, report_id)
    if not report or not report.file_path or not Path(report.file_path).exists():
        raise HTTPException(status_code=404, detail="Report file not found")
    return FileResponse(report.file_path, filename=Path(report.file_path).name)
