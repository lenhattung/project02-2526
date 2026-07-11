import secrets

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import require_admin
from app.core.security import hash_secret
from app.db.session import get_db
from app.models import ApiToken, User
from app.schemas import ApiTokenCreate, ApiTokenCreated, ApiTokenRead


router = APIRouter(prefix="/tokens", tags=["tokens"])


@router.get("", response_model=list[ApiTokenRead])
def list_tokens(_: User = Depends(require_admin), db: Session = Depends(get_db)) -> list[ApiToken]:
    return db.query(ApiToken).order_by(ApiToken.created_at.desc()).all()


@router.post("", response_model=ApiTokenCreated)
def create_token(payload: ApiTokenCreate, _: User = Depends(require_admin), db: Session = Depends(get_db)) -> ApiTokenCreated:
    raw_token = f"ctsv_{secrets.token_urlsafe(32)}"
    token = ApiToken(name=payload.name, scope=payload.scope, token_hash=hash_secret(raw_token), is_active=True)
    db.add(token)
    db.commit()
    db.refresh(token)
    data = ApiTokenRead.model_validate(token).model_dump()
    return ApiTokenCreated(**data, token=raw_token)


@router.patch("/{token_id}", response_model=ApiTokenRead)
def toggle_token(token_id: int, is_active: bool, _: User = Depends(require_admin), db: Session = Depends(get_db)) -> ApiToken:
    token = db.get(ApiToken, token_id)
    if not token:
        raise HTTPException(status_code=404, detail="Token not found")
    token.is_active = is_active
    db.add(token)
    db.commit()
    db.refresh(token)
    return token


@router.delete("/{token_id}")
def delete_token(token_id: int, _: User = Depends(require_admin), db: Session = Depends(get_db)) -> dict:
    token = db.get(ApiToken, token_id)
    if token:
        token.is_active = False
        db.add(token)
        db.commit()
    return {"ok": True}
