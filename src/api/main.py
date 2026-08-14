# pyrefly: ignore [missing-import]
from fastapi import FastAPI
from src.api.routes import screening
from src.utils.logging_config import configure_logging
from src.utils.config import get_settings


def create_app() -> FastAPI:
    # Configure structured logging early using configured level
    settings = get_settings()
    level = getattr(settings.logging, "level", "INFO") if hasattr(settings, "logging") else "INFO"
    configure_logging(level=level)

    app = FastAPI(title="AI Resume Screening V2")
    app.include_router(screening.router, prefix="/api")
    return app


app = create_app()
