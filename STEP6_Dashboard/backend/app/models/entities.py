from datetime import datetime
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.db.session import Base


JsonType = JSON().with_variant(JSONB, "postgresql")


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    deleted_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(50), default="viewer", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class ApiToken(Base, TimestampMixin):
    __tablename__ = "api_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    scope: Mapped[str] = mapped_column(String(100), default="desktop:ingest", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_used_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)


class ScrapedSource(Base, TimestampMixin):
    __tablename__ = "scraped_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_type: Mapped[str] = mapped_column(String(100), default="facebook_page", nullable=False)
    url: Mapped[str] = mapped_column(String(1024), unique=True, index=True, nullable=False)
    platform: Mapped[str] = mapped_column(String(100), default="facebook", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JsonType, nullable=True)

    news_items: Mapped[list["NewsItem"]] = relationship(back_populates="source")


class NewsItem(Base, TimestampMixin):
    __tablename__ = "news_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("scraped_sources.id"), index=True, nullable=True)
    external_id: Mapped[str] = mapped_column(String(255), index=True, nullable=True)
    url: Mapped[str] = mapped_column(String(2048), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_content: Mapped[str] = mapped_column(Text, nullable=True)
    posted_at: Mapped[datetime] = mapped_column(DateTime, index=True, nullable=True)
    collected_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    like_count: Mapped[int] = mapped_column(Integer, nullable=True)
    comment_count: Mapped[int] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="new", index=True, nullable=False)
    topic: Mapped[str] = mapped_column(String(100), index=True, nullable=True)
    importance_level: Mapped[str] = mapped_column(String(50), default="normal", index=True, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    gemini_label: Mapped[int] = mapped_column(Integer, nullable=True)
    simcse_label: Mapped[int] = mapped_column(Integer, nullable=True)
    phobert_label: Mapped[int] = mapped_column(Integer, nullable=True)
    bgem3_label: Mapped[int] = mapped_column(Integer, nullable=True)
    voted_label: Mapped[int] = mapped_column(Integer, index=True, nullable=True)
    label_status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)
    label_error_json: Mapped[dict] = mapped_column(JsonType, nullable=True)
    labeled_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    raw_payload_json: Mapped[dict] = mapped_column(JsonType, nullable=True)

    source: Mapped[ScrapedSource] = relationship(back_populates="news_items")
    comments: Mapped[list["Comment"]] = relationship(back_populates="news_item", cascade="all, delete-orphan")
    ai_results: Mapped[list["AIAnalysisResult"]] = relationship(back_populates="news_item", cascade="all, delete-orphan")


class Comment(Base):
    __tablename__ = "comments"
    __table_args__ = (UniqueConstraint("news_item_id", "content_hash", name="uq_comment_per_post_hash"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    news_item_id: Mapped[int] = mapped_column(ForeignKey("news_items.id"), index=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_content: Mapped[str] = mapped_column(Text, nullable=True)
    author_hash: Mapped[str] = mapped_column(String(64), nullable=True)
    collected_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    news_item: Mapped[NewsItem] = relationship(back_populates="comments")


class ScrapingJob(Base):
    __tablename__ = "scraping_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_job_id: Mapped[str] = mapped_column(String(255), index=True, nullable=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("scraped_sources.id"), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="running", nullable=False)
    total_collected: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_synced: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str] = mapped_column(Text, nullable=True)
    created_by_token_id: Mapped[int] = mapped_column(ForeignKey("api_tokens.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class ScrapingLog(Base):
    __tablename__ = "scraping_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("scraping_jobs.id"), index=True, nullable=True)
    level: Mapped[str] = mapped_column(String(50), default="info", nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    context_json: Mapped[dict] = mapped_column(JsonType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True, nullable=False)


class SyncBatch(Base):
    __tablename__ = "sync_batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    token_id: Mapped[int] = mapped_column(ForeignKey("api_tokens.id"), nullable=True)
    total_received: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    inserted_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    skipped_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="success", nullable=False)
    error_json: Mapped[dict] = mapped_column(JsonType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class AIAnalysisResult(Base, TimestampMixin):
    __tablename__ = "ai_analysis_results"
    __table_args__ = (UniqueConstraint("news_item_id", "provider", "model_name", name="uq_ai_result_provider_model"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    news_item_id: Mapped[int] = mapped_column(ForeignKey("news_items.id"), index=True, nullable=False)
    provider: Mapped[str] = mapped_column(String(100), default="mock", nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), default="rule-based-demo", nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    importance_score: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    attention_required: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    suggested_action: Mapped[str] = mapped_column(Text, nullable=False)
    sentiment_label: Mapped[str] = mapped_column(String(50), default="neutral", nullable=False)
    sentiment_code: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    status_label: Mapped[str] = mapped_column(String(50), default="new", nullable=False)
    raw_result_json: Mapped[dict] = mapped_column(JsonType, nullable=True)

    news_item: Mapped[NewsItem] = relationship(back_populates="ai_results")


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    report_type: Mapped[str] = mapped_column(String(50), nullable=False)
    date_from: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    date_to: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    format: Mapped[str] = mapped_column(String(20), nullable=False)
    file_path: Mapped[str] = mapped_column(String(2048), nullable=True)
    generated_by_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=True)
    filters_json: Mapped[dict] = mapped_column(JsonType, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="completed", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class SystemSetting(Base):
    __tablename__ = "system_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    value_json: Mapped[dict] = mapped_column(JsonType, default=dict, nullable=False)
    is_secret: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    updated_by_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
