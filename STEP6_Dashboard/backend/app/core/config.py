from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_LOCAL_CORS_ORIGINS = [
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "http://localhost:4174",
    "http://127.0.0.1:4174",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:18080",
    "http://127.0.0.1:18080",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=("../.env", ".env"), env_file_encoding="utf-8", extra="ignore")

    app_name: str = "CTSV News Intelligence API"
    environment: str = "local"
    api_prefix: str = "/api"
    secret_key: str = Field(default="change-me-in-production")
    access_token_expire_minutes: int = 60 * 12
    database_url: str = "sqlite:///./ctsv_dashboard.db"
    cors_origins: list[str] = DEFAULT_LOCAL_CORS_ORIGINS.copy()

    admin_email: str = "admin@example.com"
    admin_password: str = "Admin@123456"
    admin_full_name: str = "CTSV Administrator"
    desktop_api_token: str = "ctsv-demo-desktop-token"

    ai_provider: str = "mock"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-1.5-flash"
    local_models_dir: str = "/app/models"
    enable_local_models: bool = True
    label_on_ingest: bool = True

    reports_dir: str = "reports"

    @field_validator("cors_origins", mode="after")
    @classmethod
    def merge_local_cors_origins(cls, value: list[str]) -> list[str]:
        merged: list[str] = []
        for origin in [*(value or []), *DEFAULT_LOCAL_CORS_ORIGINS]:
            cleaned = origin.strip()
            if cleaned and cleaned not in merged:
                merged.append(cleaned)
        return merged


@lru_cache
def get_settings() -> Settings:
    return Settings()
