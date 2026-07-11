# CTSV News Collection, Analysis, and Reporting System

This repository now contains an end-to-end demo product for collecting Facebook/news data on a CTSV staff machine and managing it through an online dashboard.

## Modules

- `STEP1_ScrapingData/scraper.py`: existing local Facebook scraper.
- `STEP1_ScrapingData/desktop_tool`: PySide6 Desktop Tool for running scraper, scheduling, logs, and sync.
- `STEP6_Dashboard/backend`: FastAPI backend with PostgreSQL schema, auth, ingest, AI mock, reports.
- `STEP6_Dashboard/frontend`: React TypeScript dashboard.
- `deploy`: Docker Compose and environment template for VPS.
- `docs`: local, desktop, VPS, and E2E guides.

## Quick Start

Read [docs/local-runbook.md](docs/local-runbook.md).
