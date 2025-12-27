from __future__ import annotations

import asyncio
import os
from pathlib import Path
from contextlib import asynccontextmanager

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
from .routes.srs import router as srs_router
from .routes.pages import router as pages_router
from .routes.user import router as user_router
from .routes.admin import router as admin_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    # Start background worker
    try:
        from .services.background_worker import (
            start_background_worker,
            startup_generation,
        )

        # Run startup pre-generation
        startup_langs = os.getenv("ARC_STARTUP_LANGS", "").split(",")
        startup_langs = [l.strip() for l in startup_langs if l.strip()]
        texts_per_lang = int(os.getenv("ARC_STARTUP_TEXTS_PER_LANG", "2"))

        if startup_langs:
            asyncio.create_task(startup_generation(startup_langs, texts_per_lang))

        # Start background worker loop
        start_background_worker()
    except Exception as e:
        import logging

        logging.getLogger(__name__).error(f"Failed to start background worker: {e}")

    yield


app = FastAPI(lifespan=lifespan, title="Arcadia Lang", version="0.1.0")


@app.exception_handler(404)
async def not_found_exception_handler(request, exc):
    return RedirectResponse(url="/", status_code=302)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lightweight auth context via cookie
try:
    secret = os.getenv("ARC_LANG_JWT_SECRET", "dev-secret-change")
    app.add_middleware(
        CookieUserMiddleware,
        session_factory=SessionLocal,
        UserModel=Account,
        secret_key=secret,
        algorithm="HS256",
        cookie_name="access_token",
    )
except Exception:
    # Non-fatal during dev
    pass

# Optional auth API (best-effort)
try:
    from server.auth import AuthSettings, create_auth_router, create_sqlite_repo  # type: ignore

    app.include_router(
        create_auth_router(
            create_sqlite_repo(f"sqlite:///{DB_PATH}"),
            AuthSettings(secret_key="dev-secret-change"),
        )
    )
except Exception:
    pass

# Seed OpenRouter models (best-effort)
try:
    from openrouter import get_default_catalog, seed_sqlite  # type: ignore

    seed_sqlite(str(DB_PATH), table="models", cat=get_default_catalog())
except Exception:
    pass

# Built-in routers
app.include_router(health_router)
app.include_router(user_router)
# app.include_router(wordlists_router, prefix="/api")  # Disabled - word lists are generated dynamically

# Feature routers
app.include_router(reading_router)
app.include_router(srs_router)
app.include_router(pages_router)
app.include_router(admin_router)

# Static assets
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount(
        "/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static"
    )
