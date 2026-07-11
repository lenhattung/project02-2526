from __future__ import annotations

from datetime import datetime
import hashlib
import sqlite3
import uuid

from app.anonymizer import anonymize_content
from app.config import AppConfig


def normalize_text(text: str | None) -> str:
    return " ".join((text or "").split())


def make_hash(*parts: object) -> str:
    raw = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else None


def parse_datetime(value: object) -> str | None:
    if value is None or value == "":
        return None
    raw = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw[:19], fmt).isoformat()
        except ValueError:
            continue
    return None


def get_local_counts(config: AppConfig) -> dict[str, int | str]:
    try:
        conn = sqlite3.connect(config.output_db_path)
        try:
            posts = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
            comments = conn.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
            return {"posts": int(posts), "comments": int(comments), "status": "ok"}
        finally:
            conn.close()
    except Exception as exc:
        return {"posts": 0, "comments": 0, "status": str(exc)}


def build_ingest_batch(config: AppConfig, limit: int | None = None) -> dict:
    limit = limit or config.sync_limit
    conn = sqlite3.connect(config.output_db_path)
    conn.row_factory = sqlite3.Row
    try:
        posts = conn.execute(
            "SELECT id, content, source, posted_at, like_count, comment_count, collected_at FROM posts ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        post_ids = [row["id"] for row in posts]
        comments_by_post: dict[int, list[dict]] = {post_id: [] for post_id in post_ids}
        if post_ids:
            placeholders = ",".join("?" for _ in post_ids)
            for row in conn.execute(f"SELECT post_id, content, collected_at FROM comments WHERE post_id IN ({placeholders})", post_ids):
                anonymized_content = anonymize_content(row["content"])
                text = normalize_text(anonymized_content)
                comments_by_post[row["post_id"]].append(
                    {
                        "content": anonymized_content,
                        "normalized_content": text,
                        "collected_at": row["collected_at"],
                        "content_hash": make_hash(row["post_id"], text),
                    }
                )

        payload_posts = []
        for row in posts:
            anonymized_content = anonymize_content(row["content"])
            normalized = normalize_text(anonymized_content)
            if not normalized:
                continue
            source_url = row["source"] or config.source_url
            payload_posts.append(
                {
                    "external_id": f"local-post-{row['id']}",
                    "url": source_url,
                    "content": anonymized_content,
                    "normalized_content": normalized,
                    "posted_at": parse_datetime(row["posted_at"]),
                    "collected_at": parse_datetime(row["collected_at"]) or datetime.utcnow().isoformat(),
                    "like_count": parse_int(row["like_count"]),
                    "comment_count": parse_int(row["comment_count"]),
                    "content_hash": make_hash(source_url, row["id"], normalized, row["posted_at"]),
                    "raw_payload_json": {"local_sqlite_id": row["id"], "anonymized_by": "STEP2_Anonymize/Anonymize_CRF.py"},
                    "comments": comments_by_post.get(row["id"], []),
                }
            )

        return {
            "batch_id": f"desktop-{uuid.uuid4()}",
            "source": {
                "name": config.source_name,
                "url": config.source_url,
                "source_type": "facebook_page",
                "platform": "facebook",
                "is_active": True,
                "metadata_json": {"client": "desktop_tool", "output_db": config.output_db_path},
            },
            "posts": payload_posts,
        }
    finally:
        conn.close()
