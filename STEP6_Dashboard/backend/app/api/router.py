from fastapi import APIRouter

from app.api import ai, auth, dashboard, ingest, news, reports, settings, tokens


api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(tokens.router)
api_router.include_router(ingest.router)
api_router.include_router(news.router)
api_router.include_router(dashboard.router)
api_router.include_router(ai.router)
api_router.include_router(reports.router)
api_router.include_router(settings.router)
