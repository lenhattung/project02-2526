from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path

try:
    import keyring
except Exception:  # pragma: no cover
    keyring = None


APP_DIR = Path(os.getenv("APPDATA", str(Path.home()))) / "CTSVScraperTool"
CONFIG_FILE = APP_DIR / "config.json"
KEYRING_SERVICE = "ctsv_scraper_tool"
KEYRING_USER = "desktop_api_token"


@dataclass
class AppConfig:
    project_dir: str = r"F:\project02-2526\STEP1_ScrapingData"
    scraper_path: str = r"F:\project02-2526\STEP1_ScrapingData\scraper.py"
    requirements_path: str = r"F:\project02-2526\STEP1_ScrapingData\requirements-scraper-core.txt"
    output_db_path: str = r"F:\project02-2526\STEP1_ScrapingData\data\posts.db"
    cookies_path: str = r"F:\project02-2526\STEP1_ScrapingData\cookies.json"
    backend_url: str = "http://localhost:8010/api"
    source_name: str = "DNTU Confession"
    source_url: str = "https://www.facebook.com/DNTUConfession/"
    timeout_seconds: int = 7200
    max_posts_per_page: int = 2500
    duplicate_stop_threshold: int = 10
    auto_sync_after_scrape: bool = True
    sync_limit: int = 500
    schedule_enabled: bool = False
    schedule_type: str = "daily"
    schedule_value: str = "22:00"


def parse_schedule(raw: str) -> tuple[str, str]:
    value = raw.strip()
    if value.startswith("daily:"):
        return "daily", value.split(":", 1)[1]
    if value.startswith("hours:"):
        return "hours", value.split(":", 1)[1]
    return "daily", value or "22:00"


def load_config() -> AppConfig:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_FILE.exists():
        save_config(AppConfig())
    data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    return AppConfig(**{**asdict(AppConfig()), **data})


def save_config(config: AppConfig) -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8")


def get_api_token() -> str:
    if keyring:
        try:
            return keyring.get_password(KEYRING_SERVICE, KEYRING_USER) or ""
        except Exception:
            return ""
    token_file = APP_DIR / ".token"
    return token_file.read_text(encoding="utf-8").strip() if token_file.exists() else ""


def set_api_token(token: str) -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    if keyring:
        try:
            keyring.set_password(KEYRING_SERVICE, KEYRING_USER, token)
            return
        except Exception:
            pass
    (APP_DIR / ".token").write_text(token, encoding="utf-8")
