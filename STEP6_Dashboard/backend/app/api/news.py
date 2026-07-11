from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import or_
from sqlalchemy.orm import Session, joinedload

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models import NewsItem, User
from app.schemas import BulkStatusRequest, NewsItemRead, NewsListResponse, NewsUpdate


router = APIRouter(prefix="/news", tags=["news"])


@router.get("", response_model=NewsListResponse)
def list_news(
    search: str | None = None,
    source_id: int | None = None,
    status: str | None = None,
    topic: str | None = None,
    importance: str | None = None,
    voted_label: int | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    sort: str = "-created_at",
    page: int = 1,
    page_size: int = 20,
    _: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> NewsListResponse:
    query = db.query(NewsItem).options(joinedload(NewsItem.comments), joinedload(NewsItem.ai_results)).filter(NewsItem.deleted_at.is_(None))
    if search:
        like = f"%{search}%"
        query = query.filter(or_(NewsItem.content.ilike(like), NewsItem.normalized_content.ilike(like)))
    if source_id:
        query = query.filter(NewsItem.source_id == source_id)
    if status:
        query = query.filter(NewsItem.status == status)
    if topic:
        query = query.filter(NewsItem.topic == topic)
    if importance:
        query = query.filter(NewsItem.importance_level == importance)
    if voted_label is not None:
        query = query.filter(NewsItem.voted_label == voted_label)
    if date_from:
        query = query.filter(NewsItem.created_at >= date_from)
    if date_to:
        query = query.filter(NewsItem.created_at <= date_to)

    total = query.count()
    order_col = NewsItem.created_at
    if sort.lstrip("-") == "posted_at":
        order_col = NewsItem.posted_at
    if sort.lstrip("-") == "importance_level":
        order_col = NewsItem.importance_level
    if sort.lstrip("-") == "voted_label":
        order_col = NewsItem.voted_label
    query = query.order_by(order_col.desc() if sort.startswith("-") else order_col.asc())
    items = query.offset((page - 1) * page_size).limit(page_size).all()
    return NewsListResponse(items=items, total=total, page=page, page_size=page_size)


@router.get("/{news_id}", response_model=NewsItemRead)
def get_news(news_id: int, _: User = Depends(get_current_user), db: Session = Depends(get_db)) -> NewsItem:
    item = db.query(NewsItem).options(joinedload(NewsItem.comments), joinedload(NewsItem.ai_results)).filter(NewsItem.id == news_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="News item not found")
    return item


@router.patch("/{news_id}", response_model=NewsItemRead)
def update_news(news_id: int, payload: NewsUpdate, _: User = Depends(get_current_user), db: Session = Depends(get_db)) -> NewsItem:
    item = db.get(NewsItem, news_id)
    if not item:
        raise HTTPException(status_code=404, detail="News item not found")
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(item, key, value)
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@router.delete("/{news_id}")
def delete_news(news_id: int, _: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    item = db.get(NewsItem, news_id)
    if item:
        item.deleted_at = datetime.utcnow()
        db.add(item)
        db.commit()
    return {"ok": True}


@router.post("/bulk-status")
def bulk_status(payload: BulkStatusRequest, _: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    updated = db.query(NewsItem).filter(NewsItem.id.in_(payload.ids)).update({"status": payload.status}, synchronize_session=False)
    db.commit()
    return {"updated": updated}
