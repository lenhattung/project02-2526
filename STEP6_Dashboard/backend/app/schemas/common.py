from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    email: str
    password: str


class UserRead(ORMModel):
    id: int
    email: str
    full_name: str
    role: str
    is_active: bool


class ApiTokenCreate(BaseModel):
    name: str = Field(min_length=2, max_length=255)
    scope: str = "desktop:ingest"


class ApiTokenRead(ORMModel):
    id: int
    name: str
    scope: str
    is_active: bool
    last_used_at: datetime | None
    expires_at: datetime | None
    created_at: datetime


class ApiTokenCreated(ApiTokenRead):
    token: str


class SourceIn(BaseModel):
    name: str
    url: str
    source_type: str = "facebook_page"
    platform: str = "facebook"
    is_active: bool = True
    metadata_json: dict[str, Any] | None = None


class SourceRead(ORMModel):
    id: int
    name: str
    url: str
    source_type: str
    platform: str
    is_active: bool
    metadata_json: dict[str, Any] | None = None


class CommentIn(BaseModel):
    content: str
    normalized_content: str | None = None
    collected_at: datetime | None = None
    content_hash: str | None = None


class NewsItemIn(BaseModel):
    external_id: str | None = None
    url: str | None = None
    content: str
    normalized_content: str | None = None
    posted_at: datetime | None = None
    collected_at: datetime | None = None
    like_count: int | None = None
    comment_count: int | None = None
    content_hash: str | None = None
    raw_payload_json: dict[str, Any] | None = None
    comments: list[CommentIn] = []


class IngestBatchIn(BaseModel):
    batch_id: str
    client_job_id: str | None = None
    source: SourceIn
    posts: list[NewsItemIn]


class IngestBatchResult(BaseModel):
    batch_id: str
    inserted: int
    updated: int
    skipped: int
    failed: int
    total_received: int
    errors: list[str] = []


class CommentRead(ORMModel):
    id: int
    content: str
    collected_at: datetime | None


class AIResultRead(ORMModel):
    id: int
    provider: str
    model_name: str
    summary: str
    category: str
    importance_score: float
    attention_required: bool
    suggested_action: str
    sentiment_label: str
    sentiment_code: int
    status_label: str
    created_at: datetime


class NewsItemRead(ORMModel):
    id: int
    source_id: int | None
    external_id: str | None
    url: str | None
    content: str
    normalized_content: str | None
    posted_at: datetime | None
    collected_at: datetime | None
    like_count: int | None
    comment_count: int | None
    status: str
    topic: str | None
    importance_level: str
    content_hash: str
    gemini_label: int | None = None
    simcse_label: int | None = None
    phobert_label: int | None = None
    bgem3_label: int | None = None
    voted_label: int | None = None
    label_status: str
    label_error_json: dict[str, Any] | None = None
    labeled_at: datetime | None = None
    created_at: datetime
    comments: list[CommentRead] = []
    ai_results: list[AIResultRead] = []


class NewsListResponse(BaseModel):
    items: list[NewsItemRead]
    total: int
    page: int
    page_size: int


class NewsUpdate(BaseModel):
    status: str | None = None
    topic: str | None = None
    importance_level: str | None = None


class BulkStatusRequest(BaseModel):
    ids: list[int]
    status: str


class ReportCreate(BaseModel):
    report_type: str = "monthly"
    format: str = "csv"
    date_from: datetime | None = None
    date_to: datetime | None = None
    filters: dict[str, Any] | None = None


class ReportRead(ORMModel):
    id: int
    report_type: str
    format: str
    file_path: str | None
    status: str
    created_at: datetime


class SettingIn(BaseModel):
    key: str
    value_json: dict[str, Any]
    is_secret: bool = False


class SettingRead(ORMModel):
    key: str
    value_json: dict[str, Any]
    is_secret: bool
