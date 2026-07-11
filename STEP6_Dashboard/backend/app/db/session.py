from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import get_settings


settings = get_settings()
database_url = settings.database_url
connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
try:
    engine = create_engine(database_url, pool_pre_ping=True, connect_args=connect_args)
except ImportError:
    fallback_url = "sqlite:///./ctsv_dashboard.db"
    connect_args = {"check_same_thread": False}
    engine = create_engine(fallback_url, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
