# End-to-End Test Checklist

- Backend starts and `/health` returns `ok`.
- Swagger opens at `/docs`.
- Admin can login from Dashboard.
- Dashboard Overview displays seed data.
- Desktop Tool opens on Windows.
- Desktop Tool can save config and API token.
- Desktop Tool `Test API Connection` succeeds.
- Desktop Tool reads `STEP1_ScrapingData\data\posts.db`.
- `Sync Now` returns inserted/updated/skipped counts.
- Re-running `Sync Now` does not duplicate posts because `content_hash` is unique.
- News Management search/filter works.
- AI batch analysis creates summaries and importance scores.
- Report export creates CSV/XLSX/PDF and download works.
- Docker Compose starts `db`, `backend`, and `frontend`.
- No secret/API token is committed outside `.env.example`.
