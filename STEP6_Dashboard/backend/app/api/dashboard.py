from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models import NewsItem, User


router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/summary")
def summary(_: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    today = datetime.utcnow().date()
    total = db.query(NewsItem).filter(NewsItem.deleted_at.is_(None)).count()
    today_count = db.query(NewsItem).filter(func.date(NewsItem.created_at) == today, NewsItem.deleted_at.is_(None)).count()
    important = db.query(NewsItem).filter(NewsItem.importance_level.in_(["high", "critical"]), NewsItem.deleted_at.is_(None)).count()
    pending = db.query(NewsItem).filter(NewsItem.status.in_(["new", "pending"]), NewsItem.deleted_at.is_(None)).count()
    analyzed = db.query(NewsItem).filter(NewsItem.voted_label.is_not(None), NewsItem.deleted_at.is_(None)).count()
    return {"total": total, "today": today_count, "important": important, "pending": pending, "analyzed": analyzed}


@router.get("/trends")
def trends(_: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[dict]:
    start = datetime.utcnow().date() - timedelta(days=13)
    rows = (
        db.query(func.date(NewsItem.created_at).label("day"), func.count(NewsItem.id))
        .filter(func.date(NewsItem.created_at) >= start, NewsItem.deleted_at.is_(None))
        .group_by(func.date(NewsItem.created_at))
        .order_by(func.date(NewsItem.created_at))
        .all()
    )
    return [{"date": str(day), "count": count} for day, count in rows]


@router.get("/distributions")
def distributions(_: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    topics = db.query(NewsItem.topic, func.count(NewsItem.id)).filter(NewsItem.deleted_at.is_(None)).group_by(NewsItem.topic).all()
    importance = db.query(NewsItem.importance_level, func.count(NewsItem.id)).filter(NewsItem.deleted_at.is_(None)).group_by(NewsItem.importance_level).all()
    sentiments = db.query(NewsItem.voted_label, func.count(NewsItem.id)).filter(NewsItem.deleted_at.is_(None)).group_by(NewsItem.voted_label).all()
    return {
        "topics": [{"name": topic or "chua_phan_loai", "value": count} for topic, count in topics],
        "importance": [{"name": name or "normal", "value": count} for name, count in importance],
        "sentiments": [{"name": "chua_label" if label is None else str(label), "value": count} for label, count in sentiments],
    }


@router.get("/alerts")
def alerts(_: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[dict]:
    items = (
        db.query(NewsItem)
        .filter(NewsItem.importance_level.in_(["high", "critical"]), NewsItem.deleted_at.is_(None))
        .order_by(NewsItem.created_at.desc())
        .limit(10)
        .all()
    )
    return [{"id": item.id, "content": item.content, "importance_level": item.importance_level, "created_at": item.created_at} for item in items]
