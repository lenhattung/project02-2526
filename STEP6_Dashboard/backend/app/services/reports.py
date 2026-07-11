from datetime import datetime
from pathlib import Path
import csv

from openpyxl import Workbook
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import NewsItem


def get_report_rows(db: Session, date_from: datetime | None, date_to: datetime | None) -> list[NewsItem]:
    query = db.query(NewsItem).filter(NewsItem.deleted_at.is_(None))
    if date_from:
        query = query.filter(NewsItem.created_at >= date_from)
    if date_to:
        query = query.filter(NewsItem.created_at <= date_to)
    return query.order_by(NewsItem.created_at.desc()).all()


def generate_report(db: Session, report_id: int, fmt: str, date_from: datetime | None, date_to: datetime | None) -> str:
    settings = get_settings()
    out_dir = Path(settings.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = get_report_rows(db, date_from, date_to)
    path = out_dir / f"ctsv_report_{report_id}.{fmt}"

    if fmt == "csv":
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["ID", "Nguon", "Noi dung", "Chu de", "Muc do", "Trang thai", "Ngay thu thap"])
            for item in rows:
                writer.writerow([item.id, item.source_id, item.content, item.topic, item.importance_level, item.status, item.collected_at])
    elif fmt == "xlsx":
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "CTSV News"
        sheet.append(["ID", "Nguon", "Noi dung", "Chu de", "Muc do", "Trang thai", "Ngay thu thap"])
        for item in rows:
            sheet.append([item.id, item.source_id, item.content, item.topic, item.importance_level, item.status, str(item.collected_at or "")])
        workbook.save(path)
    elif fmt == "pdf":
        pdf = canvas.Canvas(str(path), pagesize=A4)
        width, height = A4
        y = height - 48
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(48, y, "CTSV News Report")
        y -= 28
        pdf.setFont("Helvetica", 9)
        for item in rows[:80]:
            text = f"#{item.id} [{item.importance_level}] {item.content[:100]}"
            pdf.drawString(48, y, text)
            y -= 16
            if y < 48:
                pdf.showPage()
                pdf.setFont("Helvetica", 9)
                y = height - 48
        pdf.save()
    else:
        raise ValueError("Unsupported report format")

    return str(path)
