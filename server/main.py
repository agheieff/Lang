from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi import HTTPException

from .db import DB_PATH, init_db, SessionLocal
from .config import MSP_ENABLE
from .auth import CookieUserMiddleware, Account
from .routes.health import router as health_router
from .routes.reading import router as reading_router
from .routes.reading_text_log import router as reading_text_log_router
from .routes.srs import router as srs_router
from .routes.pages import router as pages_router
from .routes.user import router as user_router
from .routes.admin import router as admin_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    # Validate required environment variables
    _validate_environment()

    # Background task references (to prevent garbage collection)
    background_tasks = []

    # Start background worker
    try:
        from .services.background_worker import (
            background_worker,
            startup_generation,
        )

        # Run startup pre-generation
        startup_langs = os.getenv("ARC_STARTUP_LANGS", "").split(",")
        startup_langs = [l.strip() for l in startup_langs if l.strip()]
        texts_per_lang = int(os.getenv("ARC_STARTUP_TEXTS_PER_LANG", "2"))

        if startup_langs:
            startup_task = asyncio.create_task(
                startup_generation(startup_langs, texts_per_lang)
            )
            background_tasks.append(startup_task)

        # Start background worker loop
        worker_task = asyncio.create_task(background_worker())
        background_tasks.append(worker_task)
        logger.info("Background worker started")
    except Exception as e:
        logger.error(f"Failed to start background worker: {e}", exc_info=True)

    yield

    # Cleanup: cancel all background tasks
    for task in background_tasks:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    logger.info("Background tasks cleaned up")


def _validate_environment():
    """Validate critical environment variables on startup."""
    if os.getenv("ENV") == "production":
        # Critical production checks
        required_vars = [
            "ARC_LANG_JWT_SECRET",
        ]
        missing = [v for v in required_vars if not os.getenv(v)]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )
    else:
        # Development mode warnings
        if not os.getenv("OPENROUTER_API_KEY"):
            logger.warning(
                "OPENROUTER_API_KEY not set - LLM features will be limited. "
                "Set this environment variable for full functionality."
            )
        if not os.getenv("ARC_LANG_JWT_SECRET"):
            logger.warning(
                "ARC_LANG_JWT_SECRET not set - using insecure development secret. "
                "Set this environment variable in production."
            )


app = FastAPI(lifespan=lifespan, title="Arcadia Lang", version="0.1.0")


@app.exception_handler(404)
async def not_found_exception_handler(request, exc):
    return RedirectResponse(url="/", status_code=302)


# Configure CORS - restrict in production
allowed_origins = os.getenv(
    "ARC_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lightweight auth context via cookie
try:
    secret = os.getenv("ARC_LANG_JWT_SECRET")
    if not secret:
        if os.getenv("ENV") != "dev":
            raise ValueError(
                "ARC_LANG_JWT_SECRET environment variable is required in production"
            )
        secret = "dev-secret-change"
        logger.warning("Using development JWT secret")

    app.add_middleware(
        CookieUserMiddleware,
        session_factory=SessionLocal,
        UserModel=Account,
        secret_key=secret,
        algorithm="HS256",
        cookie_name="access_token",
    )
except Exception as e:
    # Non-fatal during dev
    logger.error(f"Failed to setup auth middleware: {e}")

# Optional auth API (best-effort)
try:
    from .auth import AuthSettings, create_auth_router, create_sqlite_repo

    # Ensure JWT secret is set
    jwt_secret = os.getenv("ARC_LANG_JWT_SECRET")
    if not jwt_secret and os.getenv("ENV") != "dev":
        logger.warning("ARC_LANG_JWT_SECRET not set, using development key")

    app.include_router(
        create_auth_router(
            create_sqlite_repo(f"sqlite:///{DB_PATH}"),
            AuthSettings(secret_key=jwt_secret or "dev-secret-change"),
        )
    )
except Exception as e:
    logger.warning(f"Auth router not available: {e}")

# Built-in routers
app.include_router(health_router)
app.include_router(user_router)
# app.include_router(wordlists_router, prefix="/api")  # Disabled - word lists are generated dynamically

# Feature routers
app.include_router(reading_router)
app.include_router(reading_text_log_router)
app.include_router(srs_router)
app.include_router(pages_router)
app.include_router(admin_router)

# Static assets
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount(
        "/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static"
    )
