from datetime import datetime
import hashlib
import re

from sqlalchemy.orm import Session

from app.models import Comment, NewsItem, ScrapedSource, SyncBatch
from app.schemas import IngestBatchIn, NewsItemIn


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def make_hash(*parts: object) -> str:
    raw = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_count(value: int | str | None) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    value = str(value).replace(",", ".").strip()
    match = re.search(r"([\d.]+)", value)
    if not match:
        return None
    number = float(match.group(1))
    lowered = value.lower()
    if "k" in lowered or "ngan" in lowered:
        number *= 1000
    if "m" in lowered or "trieu" in lowered:
        number *= 1_000_000
    return int(number)


def ensure_source(db: Session, payload: IngestBatchIn) -> ScrapedSource:
    source = db.query(ScrapedSource).filter(ScrapedSource.url == payload.source.url).first()
    if source:
        source.name = payload.source.name
        source.source_type = payload.source.source_type
        source.platform = payload.source.platform
        source.is_active = payload.source.is_active
        source.metadata_json = payload.source.metadata_json
        return source
    source = ScrapedSource(**payload.source.model_dump())
    db.add(source)
    db.flush()
    return source


def upsert_post(db: Session, source: ScrapedSource, item: NewsItemIn) -> tuple[str, NewsItem]:
    normalized = normalize_text(item.normalized_content or item.content)
    content_hash = item.content_hash or make_hash(source.url, item.external_id or item.url or "", normalized, item.posted_at or "")
    post = db.query(NewsItem).filter(NewsItem.content_hash == content_hash).first()
    action = "updated" if post else "inserted"
    if not post:
        post = NewsItem(content_hash=content_hash)
        db.add(post)
    post.source_id = source.id
    post.external_id = item.external_id
    post.url = item.url
    post.content = item.content
    post.normalized_content = normalized
    post.posted_at = item.posted_at
    post.collected_at = item.collected_at or datetime.utcnow()
    post.like_count = parse_count(item.like_count)
    post.comment_count = parse_count(item.comment_count)
    post.raw_payload_json = item.raw_payload_json
    db.flush()

    for comment_in in item.comments:
        text = normalize_text(comment_in.normalized_content or comment_in.content)
        if not text:
            continue
        comment_hash = comment_in.content_hash or make_hash(post.content_hash, text)
        exists = db.query(Comment).filter(Comment.news_item_id == post.id, Comment.content_hash == comment_hash).first()
        if exists:
            continue
        db.add(
            Comment(
                news_item_id=post.id,
                content=comment_in.content,
                normalized_content=text,
                collected_at=comment_in.collected_at or item.collected_at,
                content_hash=comment_hash,
            )
        )
    return action, post


def ingest_batch(db: Session, payload: IngestBatchIn, token_id: int | None) -> tuple[dict, list[int]]:
    existing_batch = db.query(SyncBatch).filter(SyncBatch.batch_id == payload.batch_id).first()
    if existing_batch:
        return (
            {
                "batch_id": payload.batch_id,
                "inserted": 0,
                "updated": 0,
                "skipped": existing_batch.total_received,
                "failed": 0,
                "total_received": existing_batch.total_received,
                "errors": ["Batch was already processed"],
            },
            [],
        )

    inserted = updated = skipped = failed = 0
    errors: list[str] = []
    item_ids: list[int] = []
    source = ensure_source(db, payload)
    for index, item in enumerate(payload.posts, start=1):
        try:
            if not normalize_text(item.content):
                skipped += 1
                continue
            action, post = upsert_post(db, source, item)
            item_ids.append(post.id)
            if action == "inserted":
                inserted += 1
            else:
                updated += 1
        except Exception as exc:
            failed += 1
            errors.append(f"post[{index}]: {exc}")

    batch = SyncBatch(
        batch_id=payload.batch_id,
        token_id=token_id,
        total_received=len(payload.posts),
        inserted_count=inserted,
        updated_count=updated,
        skipped_count=skipped,
        failed_count=failed,
        status="partial_failed" if failed else "success",
        error_json={"errors": errors} if errors else None,
    )
    db.add(batch)
    db.commit()
    return (
        {
            "batch_id": payload.batch_id,
            "inserted": inserted,
            "updated": updated,
            "skipped": skipped,
            "failed": failed,
            "total_received": len(payload.posts),
            "errors": errors,
        },
        item_ids,
    )
