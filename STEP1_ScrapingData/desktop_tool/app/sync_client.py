from __future__ import annotations

import time

import httpx

from app.config import AppConfig


class SyncClient:
    def __init__(self, config: AppConfig, api_token: str):
        self.config = config
        self.api_token = api_token

    def headers(self) -> dict[str, str]:
        return {"X-API-Token": self.api_token}

    def test_connection(self) -> dict:
        with httpx.Client(timeout=10) as client:
            health_url = self.config.backend_url.replace("/api", "") + "/health"
            response = client.get(health_url)
            response.raise_for_status()
            return response.json()

    def send_batch(self, payload: dict, retries: int = 3) -> dict:
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                with httpx.Client(timeout=60) as client:
                    url = f"{self.config.backend_url}/ingest/batches"
                    response = client.post(url, headers=self.headers(), json=payload)
                    if response.status_code in {401, 403}:
                        raise RuntimeError(f"Backend từ chối API token tại {url}: HTTP {response.status_code} - {response.text[:300]}")
                    if response.status_code >= 400:
                        raise RuntimeError(f"Lỗi đồng bộ tới {url}: HTTP {response.status_code} - {response.text[:300]}")
                    response.raise_for_status()
                    return response.json()
            except Exception as exc:
                last_error = exc
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"Sync failed after {retries} retries: {last_error}")
