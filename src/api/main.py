# pyrefly: ignore [missing-import]
from fastapi import FastAPI
from src.api.routes import screening
from src.utils.logging_config import configure_logging
from src.utils.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    level = getattr(settings.observability, "log_level", "INFO")
    configure_logging(level=level)

    app = FastAPI(title="AI Resume Screening V2")
    app.include_router(screening.router, prefix="/api")
    return app


app = create_app()
