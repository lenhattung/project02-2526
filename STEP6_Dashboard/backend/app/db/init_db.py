from datetime import datetime, timedelta
import json
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import get_password_hash, hash_secret
from app.models import ApiToken, NewsItem, ScrapedSource, User


def init_db(db: Session) -> None:
    settings = get_settings()

    admin = db.query(User).filter(User.email == settings.admin_email).first()
    if not admin:
        admin = User(
            email=settings.admin_email,
            password_hash=get_password_hash(settings.admin_password),
            full_name=settings.admin_full_name,
            role="admin",
            is_active=True,
        )
        db.add(admin)

    token_hash = hash_secret(settings.desktop_api_token)
    token = db.query(ApiToken).filter(ApiToken.token_hash == token_hash).first()
    if not token:
        db.add(ApiToken(name="Default Desktop Tool", token_hash=token_hash, scope="desktop:ingest", is_active=True))

    source = db.query(ScrapedSource).filter(ScrapedSource.url == "https://www.facebook.com/DNTUConfession/").first()
    if not source:
        source = ScrapedSource(
            name="DNTU Confession",
            url="https://www.facebook.com/DNTUConfession/",
            source_type="facebook_page",
            platform="facebook",
            metadata_json={"seed": True},
        )
        db.add(source)
        db.flush()

    seed_path = Path(__file__).with_name("seed_data.json")
    if db.query(NewsItem).count() == 0 and seed_path.exists():
        samples = json.loads(seed_path.read_text(encoding="utf-8"))
        for index, sample in enumerate(samples, start=1):
            db.add(
                NewsItem(
                    source_id=source.id,
                    content=sample["content"],
                    normalized_content=sample["content"].lower(),
                    posted_at=datetime.utcnow() - timedelta(days=index),
                    collected_at=datetime.utcnow(),
                    like_count=sample.get("like_count", 0),
                    comment_count=sample.get("comment_count", 0),
                    topic=sample.get("topic"),
                    importance_level=sample.get("importance_level", "normal"),
                    status="new",
                    content_hash=hash_secret(f"seed-{index}-{sample['content']}"),
                    raw_payload_json={"seed": True},
                )
            )

    db.commit()
