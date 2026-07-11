from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models import AIAnalysisResult, NewsItem, User
from app.schemas import AIResultRead
from app.services.ai import persist_analysis


router = APIRouter(prefix="/ai", tags=["ai"])


def save_analysis(db: Session, item: NewsItem) -> AIAnalysisResult:
    record = persist_analysis(db, item)
    db.commit()
    db.refresh(record)
    return record


@router.post("/analyze/{news_id}", response_model=AIResultRead)
def analyze_one(news_id: int, _: User = Depends(get_current_user), db: Session = Depends(get_db)) -> AIAnalysisResult:
    item = db.get(NewsItem, news_id)
    if not item:
        raise HTTPException(status_code=404, detail="News item not found")
    return save_analysis(db, item)


@router.post("/analyze-batch")
def analyze_batch(limit: int = 50, _: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    items = db.query(NewsItem).filter(NewsItem.deleted_at.is_(None)).order_by(NewsItem.created_at.desc()).limit(limit).all()
    for item in items:
        persist_analysis(db, item)
    db.commit()
    return {"analyzed": len(items)}


@router.get("/results", response_model=list[AIResultRead])
def list_results(_: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[AIAnalysisResult]:
    return db.query(AIAnalysisResult).order_by(AIAnalysisResult.created_at.desc()).limit(200).all()
