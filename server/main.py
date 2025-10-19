from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List, Callable
import os
import time
from collections import deque, defaultdict

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Language engine
from Lang.parsing.registry import ENGINES
from Lang.parsing.morph_format import format_morph_label
from Lang.parsing.dicts.provider import DictionaryProviderChain, StarDictProvider
from Lang.parsing.dicts.cedict import CedictProvider
from .db import get_db, init_db
from .models import User, Profile, SubscriptionTier, ProfilePref
from sqlalchemy.orm import Session
import jwt
from datetime import datetime, timedelta
from argon2 import PasswordHasher
from Lang.tokenize.registry import TOKENIZERS
from Lang.tokenize.base import Token


class LookupRequest(BaseModel):
    source_lang: str = Field(..., description="BCP-47 or ISO code, e.g., 'es'")
    target_lang: str = Field(..., description="BCP-47 or ISO code for output, e.g., 'en'")
    surface: str = Field(..., description="Surface form as clicked in text")
    context: Optional[str] = Field(None, description="Optional sentence context for disambiguation")


app = FastAPI(title="Arcadia Lang", version="0.1.0")

# Allow local dev in browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple per-minute rate limit by IP or user tier for heavy endpoints
_RATE_LIMITS = {"free": 60, "premium": 300, "pro": 1000}  # requests/minute; admin unlimited via bypass
_RATE_BUCKETS: dict[str, deque] = defaultdict(deque)


@app.middleware("http")
async def _rate_limit(request, call_next):
    path = request.url.path
    if not (path.startswith("/api/lookup") or path.startswith("/api/parse")):
        return await call_next(request)
    tier = "free"
    key: Optional[str] = None
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1]
        try:
            data = jwt.decode(token, _JWT_SECRET, algorithms=["HS256"])  # type: ignore
            uid = data.get("sub")
            if uid:
                key = f"u:{uid}:{path}"
                from .db import SessionLocal
                db = SessionLocal()
                try:
                    u = db.get(User, int(uid))
                    if u and u.subscription_tier in _RATE_LIMITS:
                        tier = u.subscription_tier
                finally:
                    db.close()
        except Exception:
            pass
    if not key:
        client = request.client.host if request.client else "unknown"
        key = f"ip:{client}:{path}"
    # Admin users are unlimited
    if tier == "admin":
        return await call_next(request)
    limit = _RATE_LIMITS.get(tier, _RATE_LIMITS["free"])
    now = time.time()
    window = 60.0
    dq = _RATE_BUCKETS[key]
    while dq and now - dq[0] > window:
        dq.popleft()
    if len(dq) >= limit:
        raise HTTPException(429, "Rate limit exceeded")
    dq.append(now)
    return await call_next(request)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


 # Shared dictionary provider chain (lazy-init on first call)
_DICT_CHAIN: Optional[DictionaryProviderChain] = None
_PH = PasswordHasher()
_JWT_SECRET = os.getenv("ARC_LANG_JWT_SECRET", "dev-secret-change")  # replace in production
_ENGINES = ENGINES


def get_dict_chain() -> DictionaryProviderChain:
    global _DICT_CHAIN
    if _DICT_CHAIN is None:
        # Prefer CEDICT for Chinese, then StarDict fallback
        _DICT_CHAIN = DictionaryProviderChain(providers=[CedictProvider(), StarDictProvider()])
    return _DICT_CHAIN


@app.post("/api/lookup")
def lookup(req: LookupRequest) -> Dict[str, Any]:
    engine = _ENGINES.get(req.source_lang)
    if not engine:
        raise HTTPException(status_code=400, detail="language not supported yet")

    # Analyze word (lemma, pos, morph)
    analysis = engine.analyze_word(req.surface, context=req.context)

    # Attempt dictionary translations using lemma when available, else surface
    lemma = analysis.get("lemma") or req.surface
    translations = get_dict_chain().translations(req.source_lang, req.target_lang, lemma)

    morph = analysis.get("morph") or {}
    label = format_morph_label(analysis.get("pos"), morph)

    mode = "translation" if translations else "analysis"

    return {
        "mode": mode,
        "surface": req.surface,
        "lemma": lemma,
        "pos": analysis.get("pos"),
        "morph": morph,
        "label": label,
        "translations": translations,
        "script": analysis.get("script"),
        "pronunciation": analysis.get("pronunciation"),
    }
class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"


def _create_access_token(user_id: int) -> str:
    payload = {"sub": str(user_id), "typ": "access", "exp": datetime.utcnow() + timedelta(days=7)}
    return jwt.encode(payload, _JWT_SECRET, algorithm="HS256")


def _create_refresh_token(user_id: int) -> str:
    payload = {"sub": str(user_id), "typ": "refresh", "exp": datetime.utcnow() + timedelta(days=30)}
    return jwt.encode(payload, _JWT_SECRET, algorithm="HS256")


def _get_current_user(db: Session = Depends(get_db), authorization: str | None = None) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Not authenticated")
    token = authorization.split(" ", 1)[1]
    try:
        data = jwt.decode(token, _JWT_SECRET, algorithms=["HS256"])  # type: ignore
        user_id = int(data.get("sub"))
    except Exception:
        raise HTTPException(401, "Invalid token")
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(401, "User not found")
    return user
def _ensure_default_tiers(db: Session) -> None:
    existing = {t.name for t in db.query(SubscriptionTier).all()}
    defaults = [
        ("free", "Free plan"),
        ("premium", "Premium plan"),
        ("pro", "Pro plan"),
        ("admin", "Administrator (no limits)"),
    ]
    created = False
    for name, desc in defaults:
        if name not in existing:
            db.add(SubscriptionTier(name=name, description=desc))
            created = True
    if created:
        db.commit()


def _is_supported_lang(code: str) -> bool:
    return code in _ENGINES


def require_tier(allowed: set[str]) -> Callable[[User], User]:
    def dep(user: User = Depends(_get_current_user)) -> User:
        if user.subscription_tier not in allowed:
            raise HTTPException(403, "Insufficient subscription tier")
        return user
    return dep


@app.post("/auth/register", response_model=TokenResponse)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    init_db()
    _ensure_default_tiers(db)
    existing = db.query(User).filter(User.email == req.email).first()
    if existing:
        raise HTTPException(400, "Email already registered")
    pwd = _PH.hash(req.password)
    user = User(email=req.email, password_hash=pwd)
    db.add(user)
    db.commit()
    db.refresh(user)
    return TokenResponse(access_token=_create_access_token(user.id), refresh_token=_create_refresh_token(user.id))


@app.post("/auth/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    init_db()
    _ensure_default_tiers(db)
    user = db.query(User).filter(User.email == req.email).first()
    if not user:
        raise HTTPException(401, "Invalid credentials")
    try:
        _PH.verify(user.password_hash, req.password)
    except Exception:
        raise HTTPException(401, "Invalid credentials")
    return TokenResponse(access_token=_create_access_token(user.id), refresh_token=_create_refresh_token(user.id))


class RefreshRequest(BaseModel):
    refresh_token: str


@app.post("/auth/refresh", response_model=TokenResponse)
def refresh_token(req: RefreshRequest, db: Session = Depends(get_db)):
    try:
        data = jwt.decode(req.refresh_token, _JWT_SECRET, algorithms=["HS256"])  # type: ignore
        if data.get("typ") != "refresh":
            raise HTTPException(401, "Invalid token type")
        user_id = int(data.get("sub"))
    except Exception:
        raise HTTPException(401, "Invalid refresh token")
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(401, "User not found")
    return TokenResponse(access_token=_create_access_token(user.id), refresh_token=_create_refresh_token(user.id))


class ProfileRequest(BaseModel):
    lang: str
    subscription_tier: Optional[str] = None  # e.g., free|premium|pro from DB
    settings: Optional[Dict[str, Any]] = None


@app.post("/me/profile")
def upsert_profile(req: ProfileRequest, db: Session = Depends(get_db), user: User = Depends(_get_current_user)):
    init_db()
    _ensure_default_tiers(db)
    if not _is_supported_lang(req.lang):
        raise HTTPException(400, "Unsupported language code")
    prof = db.query(Profile).filter(Profile.user_id == user.id, Profile.lang == req.lang).first()
    if not prof:
        prof = Profile(user_id=user.id, lang=req.lang)
        db.add(prof)
        db.flush()
    if req.subscription_tier:
        # Validate against DB-backed tiers
        tier = db.query(SubscriptionTier).filter(SubscriptionTier.name == req.subscription_tier).first()
        if not tier:
            raise HTTPException(400, "Invalid subscription tier")
        user.subscription_tier = req.subscription_tier
    if req.settings is not None:
        pref = db.query(ProfilePref).filter(ProfilePref.profile_id == prof.id).first()
        if not pref:
            pref = ProfilePref(profile_id=prof.id, data=req.settings)
            db.add(pref)
        else:
            pref.data = req.settings
    db.commit()
    return {"ok": True, "user_id": user.id, "lang": prof.lang, "subscription_tier": user.subscription_tier}


class ProfileOut(BaseModel):
    lang: str
    created_at: datetime
    settings: Optional[Dict[str, Any]] = None


@app.get("/me/profiles", response_model=List[ProfileOut])
def list_profiles(db: Session = Depends(get_db), user: User = Depends(_get_current_user)):
    init_db()
    profiles = db.query(Profile).filter(Profile.user_id == user.id).all()
    out: List[ProfileOut] = []
    for p in profiles:
        pref = db.query(ProfilePref).filter(ProfilePref.profile_id == p.id).first()
        out.append(ProfileOut(lang=p.lang, created_at=p.created_at, settings=(pref.data if pref else None)))
    return out


@app.delete("/me/profile")
def delete_profile(lang: str, db: Session = Depends(get_db), user: User = Depends(_get_current_user)):
    init_db()
    prof = db.query(Profile).filter(Profile.user_id == user.id, Profile.lang == lang).first()
    if not prof:
        raise HTTPException(404, "Profile not found")
    pref = db.query(ProfilePref).filter(ProfilePref.profile_id == prof.id).first()
    if pref:
        db.delete(pref)
    db.delete(prof)
    db.commit()
    return {"ok": True}


class TierOut(BaseModel):
    name: str
    description: Optional[str] = None


@app.get("/tiers", response_model=List[TierOut])
def list_tiers(db: Session = Depends(get_db)):
    init_db()
    _ensure_default_tiers(db)
    tiers = db.query(SubscriptionTier).order_by(SubscriptionTier.id.asc()).all()
    return [TierOut(name=t.name, description=t.description) for t in tiers]


class TierSetRequest(BaseModel):
    name: str


@app.get("/me/tier", response_model=TierOut)
def get_my_tier(db: Session = Depends(get_db), user: User = Depends(_get_current_user)):
    init_db()
    _ensure_default_tiers(db)
    t = db.query(SubscriptionTier).filter(SubscriptionTier.name == user.subscription_tier).first()
    if not t:
        t = db.query(SubscriptionTier).first()
    return TierOut(name=t.name, description=t.description if t else None)


@app.post("/me/tier", response_model=TierOut)
def set_my_tier(req: TierSetRequest, db: Session = Depends(get_db), user: User = Depends(_get_current_user)):
    init_db()
    _ensure_default_tiers(db)
    t = db.query(SubscriptionTier).filter(SubscriptionTier.name == req.name).first()
    if not t:
        raise HTTPException(400, "Invalid subscription tier")
    user.subscription_tier = t.name
    db.commit()
    return TierOut(name=t.name, description=t.description)


class ParseRequest(BaseModel):
    lang: str
    text: str


@app.post("/api/parse")
def parse(req: ParseRequest) -> Dict[str, Any]:
    tok = TOKENIZERS.get(req.lang, TOKENIZERS["default"])
    words: list[Token] = tok.tokenize(req.text)
    tokens_out: list[Dict[str, Any]] = []

    def add_sep(start: int, end: int):
        if end > start:
            tokens_out.append({
                "text": req.text[start:end],
                "start": start,
                "end": end,
                "is_word": False,
                "is_mwe": False,
            })

    # Reinsert separators to preserve layout
    last = 0
    for w in words:
        if w.start > last:
            add_sep(last, w.start)
        entry: Dict[str, Any] = {
            "text": w.text,
            "start": w.start,
            "end": w.end,
            "is_word": True,
            "is_mwe": w.is_mwe,
        }
        if req.lang.startswith("zh"):
            # per-character pinyin
            try:
                from pypinyin import lazy_pinyin, Style  # type: ignore
                chars = []
                for i, ch in enumerate(w.text):
                    p_mark = lazy_pinyin(ch, style=Style.TONE)
                    p_num = lazy_pinyin(ch, style=Style.TONE3)
                    chars.append({
                        "ch": ch,
                        "start": w.start + i,
                        "end": w.start + i + 1,
                        "pinyin": p_mark[0] if p_mark else None,
                        "pinyin_num": p_num[0] if p_num else None,
                    })
                entry["chars"] = chars
            except Exception:
                pass
        tokens_out.append(entry)
        last = w.end
    add_sep(last, len(req.text))

    return {"tokens": tokens_out}


# Static site (frontend build output). We mount after API routes so they take priority.
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
