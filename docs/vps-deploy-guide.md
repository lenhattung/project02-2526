# VPS Deploy Guide

## Prepare

```bash
cd /opt/project02-2526/deploy
cp .env.example .env
nano .env
```

Change:

- `POSTGRES_PASSWORD`
- `SECRET_KEY`
- `ADMIN_PASSWORD`
- `DESKTOP_API_TOKEN`
- `POSTGRES_HOST_PORT`
- `BACKEND_HOST_PORT`
- `FRONTEND_HOST_PORT`
- `VITE_API_BASE_URL`

Recommended defaults in this repo avoid the occupied ports you listed:

- PostgreSQL host port: `55432`
- Backend host port: `18000`
- Frontend host port: `18080`

## Start

```bash
docker compose up -d --build
docker compose logs -f backend
```

Services:

- Backend API: `http://SERVER_IP:18000`
- Swagger: `http://SERVER_IP:18000/docs`
- Dashboard: `http://SERVER_IP:18080`
- PostgreSQL: internal service `db:5432`

## Desktop Tool

On the CTSV staff machine, set:

- Backend API URL: `http://SERVER_IP:18000/api`
- API token: value from `.env` or Dashboard Settings

The VPS does not run the Facebook scraper.
