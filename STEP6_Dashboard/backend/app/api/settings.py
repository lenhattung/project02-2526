from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_admin
from app.db.session import get_db
from app.models import ScrapedSource, SystemSetting, User
from app.schemas import SettingIn, SettingRead, SourceIn, SourceRead


router = APIRouter(tags=["settings"])


@router.get("/settings", response_model=list[SettingRead])
def list_settings(_: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[SystemSetting]:
    return db.query(SystemSetting).order_by(SystemSetting.key).all()


@router.put("/settings/{key}", response_model=SettingRead)
def upsert_setting(key: str, payload: SettingIn, user: User = Depends(require_admin), db: Session = Depends(get_db)) -> SystemSetting:
    setting = db.query(SystemSetting).filter(SystemSetting.key == key).first()
    if not setting:
        setting = SystemSetting(key=key)
    setting.value_json = payload.value_json
    setting.is_secret = payload.is_secret
    setting.updated_by_user_id = user.id
    db.add(setting)
    db.commit()
    db.refresh(setting)
    return setting


@router.get("/sources", response_model=list[SourceRead])
def list_sources(_: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[ScrapedSource]:
    return db.query(ScrapedSource).order_by(ScrapedSource.name).all()


@router.post("/sources", response_model=SourceRead)
def create_source(payload: SourceIn, _: User = Depends(require_admin), db: Session = Depends(get_db)) -> ScrapedSource:
    source = db.query(ScrapedSource).filter(ScrapedSource.url == payload.url).first() or ScrapedSource()
    for key, value in payload.model_dump().items():
        setattr(source, key, value)
    db.add(source)
    db.commit()
    db.refresh(source)
    return source


@router.patch("/sources/{source_id}", response_model=SourceRead)
def update_source(source_id: int, payload: SourceIn, _: User = Depends(require_admin), db: Session = Depends(get_db)) -> ScrapedSource:
    source = db.get(ScrapedSource, source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    for key, value in payload.model_dump().items():
        setattr(source, key, value)
    db.add(source)
    db.commit()
    db.refresh(source)
    return source
