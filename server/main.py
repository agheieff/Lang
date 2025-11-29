from __future__ import annotations

from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi import HTTPException

from .db import DB_PATH, init_db
from .api.profile import router as profile_router
from .api.tiers import router as tiers_router
# from .api.wordlists import router as wordlists_router  # Disabled - word lists are generated dynamically
from .config import MSP_ENABLE
from .middleware.auth import install_auth
from .middleware.rate_limit import install_rate_limit
from .routes.health import router as health_router
from .routes.reading import router as reading_router
from .routes.srs import router as srs_router
from .routes.ui import router as ui_router
from .routes.settings import router as settings_router
from .routes.user_models import router as user_models_router, htmx_router as user_models_htmx_router
from .services.background_worker import get_background_worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    
    # Start background worker for pool management
    worker = get_background_worker(interval_seconds=30)
    worker.start()
    
    yield
    
    # Stop background worker on shutdown
    worker.stop()


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
install_auth(app)

# Optional auth API (best-effort)
try:
    from server.auth import AuthSettings, create_auth_router, create_sqlite_repo  # type: ignore

    app.include_router(
        create_auth_router(
            create_sqlite_repo(f"sqlite:///{DB_PATH}"), AuthSettings(secret_key="dev-secret-change")
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
app.include_router(profile_router)
# app.include_router(wordlists_router, prefix="/api")  # Disabled - word lists are generated dynamically
app.include_router(tiers_router)

# Feature routers
app.include_router(reading_router)
app.include_router(srs_router)
app.include_router(ui_router)
app.include_router(settings_router)
app.include_router(user_models_router)
app.include_router(user_models_htmx_router)

# Optional module stream processing API
if MSP_ENABLE:
    try:
        from .api.mstream import router as msp_router

        if msp_router is not None:  # type: ignore
            app.include_router(msp_router, prefix="/msp")  # type: ignore[arg-type]
    except Exception:
        pass

# Install rate limiting middleware after routers
install_rate_limit(app)

# Static assets
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

