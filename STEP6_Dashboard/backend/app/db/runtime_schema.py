from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine


def ensure_runtime_schema(engine: Engine) -> None:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if "ai_analysis_results" in tables:
        columns = {column["name"] for column in inspector.get_columns("ai_analysis_results")}
        if "sentiment_code" not in columns:
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE ai_analysis_results ADD COLUMN sentiment_code INTEGER NOT NULL DEFAULT 1"))

    if "news_items" not in tables:
        return

    news_columns = {column["name"] for column in inspector.get_columns("news_items")}
    statements = []
    if "gemini_label" not in news_columns:
        statements.append("ALTER TABLE news_items ADD COLUMN gemini_label INTEGER")
    if "simcse_label" not in news_columns:
        statements.append("ALTER TABLE news_items ADD COLUMN simcse_label INTEGER")
    if "phobert_label" not in news_columns:
        statements.append("ALTER TABLE news_items ADD COLUMN phobert_label INTEGER")
    if "bgem3_label" not in news_columns:
        statements.append("ALTER TABLE news_items ADD COLUMN bgem3_label INTEGER")
    if "voted_label" not in news_columns:
        statements.append("ALTER TABLE news_items ADD COLUMN voted_label INTEGER")
    if "label_status" not in news_columns:
        statements.append("ALTER TABLE news_items ADD COLUMN label_status VARCHAR(50) NOT NULL DEFAULT 'pending'")
    if "label_error_json" not in news_columns:
        statements.append("ALTER TABLE news_items ADD COLUMN label_error_json JSON")
    if "labeled_at" not in news_columns:
        statements.append("ALTER TABLE news_items ADD COLUMN labeled_at TIMESTAMP")

    if statements:
        with engine.begin() as connection:
            for statement in statements:
                connection.execute(text(statement))
