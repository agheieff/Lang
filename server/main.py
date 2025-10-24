from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List, Callable, Literal, Tuple
import os
import time
from collections import deque, defaultdict
import math

from fastapi import FastAPI, HTTPException, Depends, Request, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from fastapi.responses import RedirectResponse

# Language engine
from Lang.parsing.registry import ENGINES
from Lang.parsing.morph_format import format_morph_label
from Lang.parsing.dicts.provider import DictionaryProviderChain, StarDictProvider
from Lang.parsing.dicts.cedict import CedictProvider
from .db import get_db, init_db, SessionLocal, DB_PATH
from .models import User, Profile, SubscriptionTier, ProfilePref, Lexeme, UserLexeme, WordEvent, UserLexemeContext, LexemeInfo, LexemeVariant, ReadingText, GenerationLog, ReadingTextTranslation, TranslationLog, LLMModel
from .level import update_level_if_stale, update_level_for_profile
from .level import get_level_summary
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from Lang.tokenize.registry import TOKENIZERS
from Lang.tokenize.base import Token
from .llm import PromptSpec, build_reading_prompt, chat_complete, pick_words, compose_level_hint, urgent_words_detailed, build_translation_prompt, TranslationSpec
from arcadia_auth import decode_token, create_auth_router, AuthSettings, mount_cookie_agent_middleware  # type: ignore
from fastapi.templating import Jinja2Templates

# SRS parameters
_SRS_ALPHA_CLICK = 0.2
_SRS_BETA_EXPOSURE = 0.02
_DIVERSITY_K = 8.0
_SRS_BETA_NONLOOKUP = 2
_SRS_GAMMA_NONLOOKUP = 0.08

# SRS config (env-overridable)
_W_CLICK = float(os.getenv("ARC_SRS_W_CLICK", "1.0"))
_W_NONLOOK = float(os.getenv("ARC_SRS_W_NONLOOK", "1.5"))
_W_EXPOSURE = float(os.getenv("ARC_SRS_W_EXPOSURE", "0.1"))
_HL_CLICK_D = float(os.getenv("ARC_SRS_HL_CLICK_DAYS", "14"))
_HL_NONLOOK_D = float(os.getenv("ARC_SRS_HL_NONLOOK_DAYS", "30"))
_HL_EXPOSURE_D = float(os.getenv("ARC_SRS_HL_EXPOSURE_DAYS", "7"))
_FSRS_TARGET_R = float(os.getenv("ARC_SRS_TARGET_RETENTION", "0.9"))
_FSRS_FAIL_F = float(os.getenv("ARC_SRS_FAIL_FACTOR", "0.4"))
_G_PASS_WEAK = float(os.getenv("ARC_SRS_G_PASS_WEAK", "0.15"))
_G_PASS_NORM = float(os.getenv("ARC_SRS_G_PASS_NORM", "0.30"))
_G_PASS_STRONG = float(os.getenv("ARC_SRS_G_PASS_STRONG", "0.45"))

# Exposure gating
_SESSION_MIN = int(os.getenv("ARC_SRS_SESSION_MINUTES", "30"))
_EXPOSURE_WEAK_W = float(os.getenv("ARC_SRS_EXPOSURE_WEAK", "0.05"))
_DISTINCT_PROMOTE = int(os.getenv("ARC_SRS_DISTINCTS_PROMOTE", "2"))
_FREQ_LOW_THRESH = int(os.getenv("ARC_SRS_FREQ_LOW_THRESHOLD", "10000"))
_DIFF_HIGH = float(os.getenv("ARC_SRS_DIFFICULTY_HIGH", "1.2"))

# Synthetic nonlookup promotion
_SYN_NL_ENABLE = os.getenv("ARC_SRS_SYN_NONLOOK_ENABLE", "1") != "0"
_SYN_NL_MIN_DISTINCT = int(os.getenv("ARC_SRS_SYN_NONLOOK_MIN_DISTINCT", "3"))
_SYN_NL_MIN_DAYS = float(os.getenv("ARC_SRS_SYN_NONLOOK_MIN_DAYS", "2"))
_SYN_NL_COOLDOWN_DAYS = float(os.getenv("ARC_SRS_SYN_NONLOOK_COOLDOWN_DAYS", "7"))


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

# Attach lightweight cookie agent middleware (sets request.state.user/agent from JWT cookie)
try:
    mount_cookie_agent_middleware(app, secret_key=os.getenv("ARC_LANG_JWT_SECRET", "dev-secret-change"))
except Exception:
    pass

# Integrate shared auth router from libs/auth
try:
    # Ensure DB schema exists before attaching auth routes
    init_db()
    # Ensure secret is available before using AuthSettings
    _JWT_SECRET = os.getenv("ARC_LANG_JWT_SECRET", "dev-secret-change")
    # Use SQLite-backed auth repository from libs/auth (decoupled from app DB)
    from arcadia_auth import create_sqlite_repo  # type: ignore
    _auth_settings = AuthSettings(secret_key=_JWT_SECRET, multi_profile=True)
    app.include_router(create_auth_router(create_sqlite_repo("sqlite:///data/auth.db"), _auth_settings))
except Exception:
    # non-fatal during dev if auth lib unavailable
    pass

# Populate/refresh LLM model catalog via OpenRouter sqlite helpers (best-effort)
try:
    from openrouter import seed_sqlite, get_default_catalog  # type: ignore
    seed_sqlite(str(DB_PATH), table="models", cat=get_default_catalog())
except Exception:
    pass

# Simple per-minute rate limit by IP or user tier for heavy endpoints
_RATE_LIMITS = {"free": 60, "premium": 300, "pro": 1000}  # requests/minute; admin unlimited via bypass
_RATE_BUCKETS: dict[str, deque] = defaultdict(deque)


@app.middleware("http")
async def _rate_limit(request, call_next):
    path = request.url.path
    if not (path.startswith("/api/lookup") or path.startswith("/api/parse") or path.startswith("/translate")):
        return await call_next(request)
    tier = "free"
    key: Optional[str] = None
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1]
        try:
            data = decode_token(token, _JWT_SECRET, ["HS256"])  # type: ignore
            uid = (data or {}).get("sub")
            if uid:
                key = f"u:{uid}:{path}"
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
_JWT_SECRET = os.getenv("ARC_LANG_JWT_SECRET", "dev-secret-change")  # replace in production
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_templates_env: Optional[Jinja2Templates] = None
_ENGINES = ENGINES


def get_dict_chain() -> DictionaryProviderChain:
    global _DICT_CHAIN
    if _DICT_CHAIN is None:
        # Prefer CEDICT for Chinese, then StarDict fallback
        _DICT_CHAIN = DictionaryProviderChain(providers=[CedictProvider(), StarDictProvider()])
    return _DICT_CHAIN


def _templates() -> Jinja2Templates:
    global _templates_env
    if _templates_env is None:
        try:
            # Reuse directory ensured by UI style
            tdir = TEMPLATES_DIR
            _templates_env = Jinja2Templates(directory=str(tdir))
        except Exception:
            _templates_env = Jinja2Templates(directory=str(TEMPLATES_DIR))
    return _templates_env


def _hash_context(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    try:
        import hashlib
        h = hashlib.sha256(text.encode('utf-8')).hexdigest()
        return h[:32]
    except Exception:
        return None


def _decay_posterior(ul: UserLexeme, now: datetime, hl_days: float) -> None:
    if not hl_days or hl_days <= 0:
        ul.last_decay_at = now
        return
    ref = ul.last_decay_at or ul.last_seen_at or ul.created_at or now
    dt = (now - ref).total_seconds() / 86400.0
    if dt <= 0:
        ul.last_decay_at = now
        return
    k = pow(0.5, dt / hl_days)
    ul.alpha = float(max(0.0, (ul.alpha or 1.0) * k))
    ul.beta = float(max(0.0, (ul.beta or 9.0) * k))
    ul.last_decay_at = now


def _retention_now(ul: UserLexeme, now: datetime) -> float:
    S = max(0.5, float(ul.stability or 1.0))
    ref = ul.last_seen_at or ul.first_seen_at or ul.created_at or now
    dt = max(0.0, (now - ref).total_seconds() / 86400.0)
    try:
        return math.exp(-dt / S)
    except Exception:
        return 0.5


def _schedule_next(ul: UserLexeme, quality: int, now: datetime) -> None:
    S = max(0.5, float(ul.stability or 1.0))
    D = float(getattr(ul, "difficulty", 1.0) or 1.0)
    if quality <= 0:
        S = max(0.5, S * _FSRS_FAIL_F)
        nxt = now + timedelta(days=1)
        D = min(1.5, D + 0.05)
    else:
        g = _G_PASS_WEAK if quality == 1 else _G_PASS_NORM if quality == 2 else _G_PASS_STRONG
        S = S * (1.0 + g * (2.0 - D))
        try:
            days = S * (-math.log(_FSRS_TARGET_R))
        except Exception:
            days = max(1.0, S * 0.1)
        nxt = now + timedelta(days=days)
        D = max(0.3, min(1.5, D - (0.02 if quality == 3 else 0.01)))
    ul.stability = float(min(3650.0, S))
    ul.difficulty = float(D)
    ul.next_due_at = nxt


# Cached OpenCC converters (graceful if unavailable)
try:  # type: ignore
    from opencc import OpenCC  # type: ignore
    _OPENCC_T2S = OpenCC("t2s")
    _OPENCC_S2T = OpenCC("s2t")
except Exception:  # noqa: BLE001
    _OPENCC_T2S = None  # type: ignore
    _OPENCC_S2T = None  # type: ignore


def _resolve_lexeme(db: Session, lang: str, lemma: str, pos: Optional[str]) -> Lexeme:
    # For Chinese, normalize to unified 'zh' with simplified lemma; capture Hans/Hant variants
    is_zh = lang.startswith("zh")
    canon_lang = "zh" if is_zh else lang
    canon_lemma = lemma
    hans_form = None
    hant_form = None
    if is_zh:
        try:
            hans = _OPENCC_T2S.convert(lemma) if _OPENCC_T2S else lemma
            hant = _OPENCC_S2T.convert(hans) if _OPENCC_S2T else None
            canon_lemma = hans
            hans_form = hans
            hant_form = hant
        except Exception:
            hans_form = lemma
    # Try to reuse existing lexeme via variants to avoid duplicates
    lex = db.query(Lexeme).filter(Lexeme.lang == canon_lang, Lexeme.lemma == canon_lemma, Lexeme.pos == pos).first()
    if not lex and is_zh:
        # If a variant already exists globally, adopt its lexeme
        v = None
        if hans_form:
            v = db.query(LexemeVariant).filter(LexemeVariant.script == "Hans", LexemeVariant.form == hans_form).first()
        if not v and hant_form:
            v = db.query(LexemeVariant).filter(LexemeVariant.script == "Hant", LexemeVariant.form == hant_form).first()
        if v:
            candidate = db.get(Lexeme, v.lexeme_id)
            if candidate:
                lex = candidate
    if not lex:
        lex = Lexeme(lang=canon_lang, lemma=canon_lemma, pos=pos)
        db.add(lex)
        db.flush()
    if is_zh:
        # Build unique target pairs to insert (avoid duplicates within same session, no autoflush)
        to_check: set[tuple[str, str]] = set()
        if hans_form:
            to_check.add(("Hans", hans_form))
        if hant_form:
            to_check.add(("Hant", hant_form))
        orig_script = "Hant" if lang.endswith("Hant") else "Hans"
        if (orig_script, lemma) not in to_check:
            to_check.add((orig_script, lemma))
        for sc, fm in to_check:
            exists = db.query(LexemeVariant).filter(LexemeVariant.script == sc, LexemeVariant.form == fm).first()
            if not exists:
                db.add(LexemeVariant(lexeme_id=lex.id, script=sc, form=fm))
    db.flush()
    return lex


def _get_or_create_userlexeme(db: Session, user: User, profile: Profile, lexeme: Lexeme) -> UserLexeme:
    ul = db.query(UserLexeme).filter(UserLexeme.user_id == user.id, UserLexeme.profile_id == profile.id, UserLexeme.lexeme_id == lexeme.id).first()
    if ul:
        return ul
    # Initialize importance from frequency when available
    imp = 0.5
    try:
        li = db.query(LexemeInfo).filter(LexemeInfo.lexeme_id == lexeme.id).first()
        if li and li.freq_rank:
            imp = min(1.0, 1.0 / (1.0 + li.freq_rank / 500.0))
    except Exception:
        pass
    ul = UserLexeme(user_id=user.id, profile_id=profile.id, lexeme_id=lexeme.id, importance=imp)
    db.add(ul)
    db.flush()
    return ul


def _srs_click(db: Session, user: User, lang: str, lemma: str, pos: Optional[str], surface: Optional[str], context_hash: Optional[str], text_id: Optional[int] = None):
    # Pick or create profile for the lang
    prof = db.query(Profile).filter(Profile.user_id == user.id, Profile.lang == lang).first()
    if not prof:
        prof = Profile(user_id=user.id, lang=lang)
        db.add(prof)
        db.commit()
        db.refresh(prof)
    lex = _resolve_lexeme(db, lang, lemma, pos)
    ul = _get_or_create_userlexeme(db, user, prof, lex)
    # Update Beta and stability
    ul.a_click = (ul.a_click or 0) + 1
    ul.clicks = (ul.clicks or 0) + 1
    ul.last_clicked_at = datetime.utcnow()
    now = datetime.utcnow()
    _decay_posterior(ul, now, _HL_CLICK_D)
    ul.alpha = float((ul.alpha or 1.0) + _W_CLICK)
    _schedule_next(ul, 0, now)
    # Event row
    ev = WordEvent(ts=now, user_id=user.id, profile_id=prof.id, lexeme_id=lex.id, event_type="click", count=1, surface=surface, context_hash=context_hash, source="manual", meta={}, text_id=text_id)
    db.add(ev)
    db.commit()


@app.post("/api/lookup")
def lookup(req: LookupRequest, request: Request, db: Session = Depends(get_db)) -> Dict[str, Any]:
    engine = _ENGINES.get(req.source_lang)
    if not engine:
        raise HTTPException(status_code=400, detail="language not supported yet")

    # Analyze word (lemma, pos, morph)
    analysis = engine.analyze_word(req.surface, context=req.context)

    # Attempt dictionary translations using lemma when available, else surface
    lemma = analysis.get("lemma") or req.surface
    translations = get_dict_chain().translations(req.source_lang, req.target_lang, lemma)
    # Fallback: if lemma yields nothing, try surface form as well (zh special handling remains elsewhere)
    if not translations and req.surface != lemma:
        more = get_dict_chain().translations(req.source_lang, req.target_lang, req.surface)
        if more:
            translations = more

    morph = analysis.get("morph") or {}
    label = format_morph_label(analysis.get("pos"), morph)

    mode = "translation" if translations else "analysis"

    resp = {
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
    # Auto-log click event for SRS if authenticated header is present
    try:
        auth = request.headers.get("authorization") or request.headers.get("Authorization")
        if auth and auth.lower().startswith("bearer "):
            data = decode_token(auth.split(" ", 1)[1], _JWT_SECRET, ["HS256"])  # type: ignore
            uid = int((data or {}).get("sub")) if data and data.get("sub") is not None else None
            if uid is not None:
                user = db.get(User, uid)
                if user:
                    _srs_click(db, user, req.source_lang, lemma=lemma, pos=analysis.get("pos"), surface=req.surface, context_hash=_hash_context(req.context))
    except Exception:
        pass
    return resp
def _get_current_user(request: Request, db: Session = Depends(get_db), authorization: Optional[str] = Header(default=None)) -> User:
    token: Optional[str] = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1]
    else:
        token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(401, "Not authenticated")
    try:
        data = decode_token(token, _JWT_SECRET, ["HS256"])  # type: ignore
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




class ProfileRequest(BaseModel):
    lang: str
    subscription_tier: Optional[str] = None  # e.g., free|premium|pro from DB
    settings: Optional[Dict[str, Any]] = None
    level_value: Optional[float] = None
    level_var: Optional[float] = None
    level_code: Optional[str] = None
    preferred_script: Optional[str] = None  # 'Hans' | 'Hant' for zh


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
    # Optional preferred script for Chinese
    if req.preferred_script is not None and req.lang.startswith("zh"):
        ps = req.preferred_script
        if ps not in ("Hans", "Hant"):
            raise HTTPException(400, "preferred_script must be 'Hans' or 'Hant'")
        prof.preferred_script = ps
    # Optional level updates
    if req.level_value is not None:
        prof.level_value = float(req.level_value)
    if req.level_var is not None:
        prof.level_var = float(req.level_var)
    if req.level_code is not None:
        prof.level_code = req.level_code
    db.commit()
    return {"ok": True, "user_id": user.id, "lang": prof.lang, "subscription_tier": user.subscription_tier, "level_value": prof.level_value, "level_var": prof.level_var, "level_code": prof.level_code}


class ProfileOut(BaseModel):
    lang: str
    created_at: datetime
    settings: Optional[Dict[str, Any]] = None
    level_value: float
    level_var: float
    level_code: Optional[str] = None
    preferred_script: Optional[str] = None


@app.get("/me/profiles", response_model=List[ProfileOut])
def list_profiles(db: Session = Depends(get_db), user: User = Depends(_get_current_user)):
    init_db()
    profiles = db.query(Profile).filter(Profile.user_id == user.id).all()
    out: List[ProfileOut] = []
    for p in profiles:
        pref = db.query(ProfilePref).filter(ProfilePref.profile_id == p.id).first()
        out.append(ProfileOut(lang=p.lang, created_at=p.created_at, settings=(pref.data if pref else None), level_value=p.level_value, level_var=p.level_var, level_code=p.level_code, preferred_script=p.preferred_script))
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


class MeOut(BaseModel):
    id: int
    email: str
    subscription_tier: str


@app.get("/me", response_model=MeOut)
def get_me(user: User = Depends(_get_current_user)):
    return MeOut(id=user.id, email=user.email, subscription_tier=user.subscription_tier)


# ---- UI Theme and Prefs (minimal) ----
class ThemeIn(BaseModel):
    name: Optional[str] = None
    vars: Optional[Dict[str, Any]] = None
    clear: Optional[bool] = None


class UIPrefsIn(BaseModel):
    motion: Optional[bool] = None
    density: Optional[str] = None  # 'comfortable' | 'compact'
    scale: Optional[float] = None  # 0.9–1.2
    clear: Optional[bool] = None


def _get_or_create_profile(db: Session, user: User, lang: str) -> Profile:
    prof = db.query(Profile).filter(Profile.user_id == user.id, Profile.lang == lang).first()
    if prof:
        return prof
    prof = Profile(user_id=user.id, lang=lang)
    db.add(prof)
    db.commit()
    db.refresh(prof)
    return prof


def _get_pref_row(db: Session, profile_id: int) -> ProfilePref:
    pref = db.query(ProfilePref).filter(ProfilePref.profile_id == profile_id).first()
    if not pref:
        pref = ProfilePref(profile_id=profile_id, data={})
        db.add(pref)
        db.commit()
        db.refresh(pref)
    return pref


@app.get("/theme")
def get_theme(lang: str, db: Session = Depends(get_db), user: User = Depends(_get_current_user)):
    prof = _get_or_create_profile(db, user, lang)
    pref = _get_pref_row(db, prof.id)
    data = dict(pref.data or {})
    theme = data.get("theme") or {}
    return {"name": theme.get("name"), "vars": theme.get("vars") or {}}


@app.post("/theme")
def set_theme(payload: ThemeIn, lang: str, db: Session = Depends(get_db), user: User = Depends(_get_current_user)):
    prof = _get_or_create_profile(db, user, lang)
    pref = _get_pref_row(db, prof.id)
    data = dict(pref.data or {})
    if payload.clear:
        data.pop("theme", None)
    else:
        cur = data.get("theme") or {}
        if payload.name is not None:
            cur["name"] = payload.name
        if payload.vars is not None:
            cur["vars"] = payload.vars
        data["theme"] = cur
    pref.data = data
    db.commit()
    return {"ok": True}


@app.get("/prefs")
def get_ui_prefs(lang: str, db: Session = Depends(get_db), user: User = Depends(_get_current_user)):
    prof = _get_or_create_profile(db, user, lang)
    pref = _get_pref_row(db, prof.id)
    data = dict(pref.data or {})
    return data.get("ui_prefs") or {}


@app.post("/prefs")
def set_ui_prefs(payload: UIPrefsIn, lang: str, db: Session = Depends(get_db), user: User = Depends(_get_current_user)):
    prof = _get_or_create_profile(db, user, lang)
    pref = _get_pref_row(db, prof.id)
    data = dict(pref.data or {})
    if payload.clear:
        data.pop("ui_prefs", None)
    else:
        cur = data.get("ui_prefs") or {}
        if payload.motion is not None:
            cur["motion"] = bool(payload.motion)
        if payload.density:
            cur["density"] = payload.density
        if payload.scale is not None:
            cur["scale"] = float(payload.scale)
        data["ui_prefs"] = cur
    pref.data = data
    db.commit()
    return {"ok": True}


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
            try:
                from pypinyin import lazy_pinyin, Style  # type: ignore
                p_mark_list = lazy_pinyin(w.text, style=Style.TONE)
                p_num_list = lazy_pinyin(w.text, style=Style.TONE3)
                chars = []
                for i, ch in enumerate(w.text):
                    p_mark = p_mark_list[i] if i < len(p_mark_list) else None
                    p_num = p_num_list[i] if i < len(p_num_list) else None
                    chars.append({
                        "ch": ch,
                        "start": w.start + i,
                        "end": w.start + i + 1,
                        "pinyin": p_mark,
                        "pinyin_num": p_num,
                    })
                entry["chars"] = chars
            except Exception:
                pass
        tokens_out.append(entry)
        last = w.end
    add_sep(last, len(req.text))

    return {"tokens": tokens_out}


# -------- LLM reading generation --------
class GenRequest(BaseModel):
    lang: str
    length: Optional[int] = None  # UI length hint; unit is decided per-language
    include_words: Optional[List[str]] = None
    model: Optional[str] = None
    provider: Optional[str] = "openrouter"  # default to openrouter; alt: lmstudio
    base_url: str = "http://localhost:1234/v1"


@app.post("/gen/reading")
def gen_reading(req: GenRequest, db: Session = Depends(get_db), user: User = Depends(_get_current_user)):
    init_db()
    # Preferred script: pick from user's profile when available
    script = None
    if req.lang.startswith("zh"):
        prof = db.query(Profile).filter(Profile.user_id == user.id, Profile.lang == req.lang).first()
        if prof and getattr(prof, "preferred_script", None) in ("Hans", "Hant"):
            script = prof.preferred_script
        else:
            script = "Hans"
    # Auto-pick words if not provided
    words = req.include_words or pick_words(db, user, req.lang, count=12)
    level_hint = compose_level_hint(db, user, req.lang)
    # Per-language unit and default length
    unit = "chars" if req.lang.startswith("zh") else "words"
    approx_len = req.length if req.length is not None else (300 if unit == "chars" else 180)
    spec = PromptSpec(lang=req.lang, unit=unit, approx_len=approx_len, user_level_hint=level_hint, include_words=words, script=script)
    messages = build_reading_prompt(spec)
    text: str
    try:
        text = chat_complete(
            messages,
            provider=req.provider,
            model=req.model,
            base_url=req.base_url,
        )
    except Exception as e:
        # Graceful error: return a sanitized message without stack traces or placeholder text
        import logging
        logging.getLogger("uvicorn.error").warning("LLM generation failed: %s", e)
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="No LLM backend available")
    # Save text and generation log (unlimited tier check placeholder)
    source = "llm"
    rt = ReadingText(user_id=user.id, lang=req.lang, content=text, source=source)
    db.add(rt)
    db.flush()
    pid = db.query(Profile.id).filter(Profile.user_id == user.id, Profile.lang == req.lang).scalar()
    gl = GenerationLog(user_id=user.id, profile_id=pid, text_id=rt.id, model=req.model, base_url=req.base_url, prompt={"messages": messages, "provider": (req.provider or None)}, words={"include": words}, level_hint=level_hint, approx_len=approx_len, unit=unit)
    db.add(gl)
    db.commit()
    return {"prompt": messages, "text": text, "level_hint": level_hint, "words": words, "text_id": rt.id}


# -------- Urgent words (selection only) --------
@app.get("/srs/urgent")
def srs_urgent(lang: str, total: int = 12, new_ratio: float = 0.3, db: Session = Depends(get_db), user: User = Depends(_get_current_user)):
    init_db()
    items = urgent_words_detailed(db, user, lang, total=total, new_ratio=new_ratio)
    return {"words": [it["form"] for it in items], "items": items}


# -------- Lexeme Info (freq/levels) --------
class LexemeInfoItem(BaseModel):
    lang: str
    lemma: str
    pos: Optional[str] = None
    freq_rank: Optional[int] = None
    freq_score: Optional[float] = None
    level_code: Optional[str] = None
    source: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None


class LexemeInfoUpsertRequest(BaseModel):
    items: List[LexemeInfoItem]


@app.post("/dict/lexeme_info/upsert")
def upsert_lexeme_info(req: LexemeInfoUpsertRequest, db: Session = Depends(get_db), _admin: User = Depends(require_tier({"admin"}))):
    init_db()
    updated = 0
    for it in req.items:
        lex = _resolve_lexeme(db, it.lang, it.lemma, it.pos)
        li = db.query(LexemeInfo).filter(LexemeInfo.lexeme_id == lex.id).first()
        if not li:
            li = LexemeInfo(lexeme_id=lex.id)
            db.add(li)
        if it.freq_rank is not None:
            li.freq_rank = it.freq_rank
        if it.freq_score is not None:
            li.freq_score = it.freq_score
        if it.level_code is not None:
            li.level_code = it.level_code
        if it.source is not None:
            li.source = it.source
        if it.tags is not None:
            li.tags = it.tags
        updated += 1
    db.commit()
    return {"ok": True, "updated": updated}


@app.get("/dict/lexeme_info")
def get_lexeme_info(lang: str, lemma: str, pos: Optional[str] = None, db: Session = Depends(get_db)):
    init_db()
    lex = db.query(Lexeme).filter(Lexeme.lang == lang, Lexeme.lemma == lemma, Lexeme.pos == pos).first()
    if not lex:
        return {}
    li = db.query(LexemeInfo).filter(LexemeInfo.lexeme_id == lex.id).first()
    if not li:
        return {}
    return {"lexeme_id": lex.id, "freq_rank": li.freq_rank, "freq_score": li.freq_score, "level_code": li.level_code, "source": li.source, "tags": li.tags}


# -------- SRS Endpoints --------
class ExposureItem(BaseModel):
    lemma: Optional[str] = None
    pos: Optional[str] = None
    surface: Optional[str] = None
    context: Optional[str] = None


class ExposuresRequest(BaseModel):
    lang: str
    items: List[ExposureItem]
    text_id: Optional[int] = None


def _srs_exposure(db: Session, user: User, lang: str, lemma: str, pos: Optional[str], surface: Optional[str], context_hash: Optional[str], text_id: Optional[int] = None):
    prof = db.query(Profile).filter(Profile.user_id == user.id, Profile.lang == lang).first()
    if not prof:
        prof = Profile(user_id=user.id, lang=lang)
        db.add(prof)
        db.commit()
        db.refresh(prof)
    lex = _resolve_lexeme(db, lang, lemma, pos)
    ul = _get_or_create_userlexeme(db, user, prof, lex)
    now = datetime.utcnow()
    # determine exposure weight with gating
    # 1) session collapse: skip additional weight if last exposure was recent
    recent_ev = db.query(WordEvent).filter(
        WordEvent.user_id == user.id,
        WordEvent.profile_id == prof.id,
        WordEvent.lexeme_id == lex.id,
        WordEvent.event_type == "exposure",
    ).order_by(WordEvent.ts.desc()).first()
    session_skip = False
    if recent_ev:
        try:
            mins = (now - recent_ev.ts).total_seconds() / 60.0
            if mins < _SESSION_MIN:
                session_skip = True
        except Exception:
            session_skip = False
    # 2) distinct texts gating (use before increment)
    distincts_before = int(ul.distinct_texts or 0)
    w_exp = _W_EXPOSURE
    quality = 2  # normal pass
    if distincts_before < _DISTINCT_PROMOTE:
        w_exp = min(w_exp, _EXPOSURE_WEAK_W)
        quality = 1  # weak pass
    # 3) difficulty/frequency adjustment
    li = db.query(LexemeInfo).filter(LexemeInfo.lexeme_id == lex.id).first()
    if (getattr(ul, "difficulty", 1.0) or 1.0) >= _DIFF_HIGH or (li and li.freq_rank and li.freq_rank > _FREQ_LOW_THRESH):
        w_exp = min(w_exp, _EXPOSURE_WEAK_W)
        quality = min(quality, 1)
    # 4) apply updates
    ul.b_nonclick = (ul.b_nonclick or 0) + 1
    ul.exposures = (ul.exposures or 0) + 1
    if not ul.first_seen_at:
        ul.first_seen_at = now
    ul.last_seen_at = now
    _decay_posterior(ul, now, _HL_EXPOSURE_D)
    if not session_skip and w_exp > 0:
        ul.beta = float((ul.beta or 9.0) + w_exp)
    _schedule_next(ul, quality, now)
    # Diversity (distinct contexts)
    if context_hash:
        exists = db.query(UserLexemeContext).filter(UserLexemeContext.user_lexeme_id == ul.id, UserLexemeContext.context_hash == context_hash).first()
        if not exists:
            db.add(UserLexemeContext(user_lexeme_id=ul.id, context_hash=context_hash))
            ul.distinct_texts = (ul.distinct_texts or 0) + 1
    # Event
    db.add(WordEvent(ts=now, user_id=user.id, profile_id=prof.id, lexeme_id=lex.id, event_type="exposure", count=1, surface=surface, context_hash=context_hash, source="manual", meta={}, text_id=text_id))

    # Synthetic nonlookup promotion (conservative)
    if _SYN_NL_ENABLE:
        try:
            if (ul.clicks or 0) == 0 and int(ul.distinct_texts or 0) >= _SYN_NL_MIN_DISTINCT and ul.first_seen_at:
                days_seen = (now - ul.first_seen_at).total_seconds() / 86400.0
                if days_seen >= _SYN_NL_MIN_DAYS:
                    # cooldown: no recent nonlookup
                    recent_nl = db.query(WordEvent).filter(
                        WordEvent.user_id == user.id,
                        WordEvent.profile_id == prof.id,
                        WordEvent.lexeme_id == lex.id,
                        WordEvent.event_type == "nonlookup",
                        WordEvent.ts >= (now - timedelta(days=_SYN_NL_COOLDOWN_DAYS))
                    ).first()
                    if not recent_nl:
                        _srs_nonlookup(db, user, lang, lemma, pos, surface, context_hash, text_id)
        except Exception:
            pass


@app.post("/srs/event/exposures")
def srs_exposures(req: ExposuresRequest, db: Session = Depends(get_db), user: User = Depends(_get_current_user)):
    init_db()
    count = 0
    for it in req.items:
        lemma = it.lemma
        pos = it.pos
        if not lemma and it.surface:
            engine = _ENGINES.get(req.lang)
            if engine:
                analysis = engine.analyze_word(it.surface, context=None)
                lemma = analysis.get("lemma") or it.surface
                pos = pos or analysis.get("pos")
        if not lemma:
            continue
        _srs_exposure(db, user, req.lang, lemma, pos, it.surface, _hash_context(it.context), req.text_id)
        count += 1
    db.commit()
    # Update bulk level estimate if stale (best-effort)
    try:
        update_level_if_stale(db, user.id, req.lang)
        db.commit()
    except Exception:
        pass
    return {"ok": True, "count": count}


class NonLookupRequest(BaseModel):
    lang: str
    items: List[ExposureItem]
    text_id: Optional[int] = None


def _srs_nonlookup(db: Session, user: User, lang: str, lemma: str, pos: Optional[str], surface: Optional[str], context_hash: Optional[str], text_id: Optional[int] = None):
    # ensure profile and lexeme
    prof = db.query(Profile).filter(Profile.user_id == user.id, Profile.lang == lang).first()
    if not prof:
        prof = Profile(user_id=user.id, lang=lang)
        db.add(prof)
        db.commit()
        db.refresh(prof)
    lex = _resolve_lexeme(db, lang, lemma, pos)
    ul = _get_or_create_userlexeme(db, user, prof, lex)
    # explicit non-click success: add strong nonlookup weight and schedule
    now = datetime.utcnow()
    _decay_posterior(ul, now, _HL_NONLOOK_D)
    ul.b_nonclick = (ul.b_nonclick or 0) + _SRS_BETA_NONLOOKUP
    ul.beta = float((ul.beta or 9.0) + _W_NONLOOK)
    ul.last_seen_at = now
    _schedule_next(ul, 3, now)
    db.add(WordEvent(ts=now, user_id=user.id, profile_id=prof.id, lexeme_id=lex.id, event_type="nonlookup", count=1, surface=surface, context_hash=context_hash, source="manual", meta={}, text_id=text_id))


@app.post("/srs/event/nonlookup")
def srs_nonlookup(req: NonLookupRequest, db: Session = Depends(get_db), user: User = Depends(_get_current_user)):
    init_db()
    count = 0
    for it in req.items:
        lemma = it.lemma
        pos = it.pos
        if not lemma and it.surface:
            engine = _ENGINES.get(req.lang)
            if engine:
                analysis = engine.analyze_word(it.surface, context=None)
                lemma = analysis.get("lemma") or it.surface
                pos = pos or analysis.get("pos")
        if not lemma:
            continue
        _srs_nonlookup(db, user, req.lang, lemma, pos, it.surface, _hash_context(it.context), req.text_id)
        count += 1
    db.commit()
    try:
        update_level_if_stale(db, user.id, req.lang)
        db.commit()
    except Exception:
        pass
    return {"ok": True, "count": count}


class ClickRequest(BaseModel):
    lang: str
    lemma: Optional[str] = None
    pos: Optional[str] = None
    surface: Optional[str] = None
    context: Optional[str] = None
    text_id: Optional[int] = None


@app.post("/srs/event/click")
def srs_click(req: ClickRequest, db: Session = Depends(get_db), user: User = Depends(_get_current_user)):
    init_db()
    lemma = req.lemma
    pos = req.pos
    if not lemma and req.surface:
        engine = _ENGINES.get(req.lang)
        if engine:
            analysis = engine.analyze_word(req.surface, context=None)
            lemma = analysis.get("lemma") or req.surface
            pos = pos or analysis.get("pos")
    if not lemma:
        raise HTTPException(400, "lemma or surface required")
    _srs_click(db, user, req.lang, lemma, pos, req.surface, _hash_context(req.context), req.text_id)
    try:
        update_level_if_stale(db, user.id, req.lang)
        db.commit()
    except Exception:
        pass
    # Return derived metrics
    lex = _resolve_lexeme(db, req.lang, lemma, pos)
    prof = db.query(Profile).filter(Profile.user_id == user.id, Profile.lang == req.lang).first()
    ul = db.query(UserLexeme).filter(UserLexeme.user_id == user.id, UserLexeme.profile_id == (prof.id if prof else -1), UserLexeme.lexeme_id == lex.id).first()
    if not ul:
        return {"ok": True}
    a = ul.a_click or 0
    b = ul.b_nonclick or 0
    p_click = a / (a + b) if (a + b) > 0 else 0.0
    return {"ok": True, "lexeme_id": lex.id, "p_click": p_click, "n": a + b, "stability": ul.stability, "diversity": ul.distinct_texts}


class SrsWordsOut(BaseModel):
    lexeme_id: int
    lemma: str
    pos: Optional[str]
    p_click: float
    n: int
    stability: float
    diversity: int
    freq_rank: Optional[int] = None
    level_code: Optional[str] = None


@app.get("/srs/words", response_model=List[SrsWordsOut])
def get_srs_words(
    lang: str = Query(...),
    min_p: Optional[float] = None,
    max_p: Optional[float] = None,
    min_S: Optional[float] = None,
    max_S: Optional[float] = None,
    min_D: Optional[int] = None,
    max_D: Optional[int] = None,
    limit: int = 50,
    db: Session = Depends(get_db),
    user: User = Depends(_get_current_user),
):
    init_db()
    q = db.query(UserLexeme, Lexeme, LexemeInfo).join(Lexeme, UserLexeme.lexeme_id == Lexeme.id).outerjoin(LexemeInfo, LexemeInfo.lexeme_id == Lexeme.id).filter(
        UserLexeme.user_id == user.id,
        UserLexeme.profile_id == db.query(Profile.id).filter(Profile.user_id == user.id, Profile.lang == lang).scalar_subquery(),
    )
    # Push simple filters into SQL
    if min_S is not None:
        q = q.filter(UserLexeme.stability >= float(min_S))
    if max_S is not None:
        q = q.filter(UserLexeme.stability <= float(max_S))
    if min_D is not None:
        q = q.filter(UserLexeme.distinct_texts >= int(min_D))
    if max_D is not None:
        q = q.filter(UserLexeme.distinct_texts <= int(max_D))
    # p filters approximate: only apply when denominator > 0
    if min_p is not None or max_p is not None:
        denom = (UserLexeme.a_click + UserLexeme.b_nonclick)
        q = q.filter(denom > 0)
        if min_p is not None:
            q = q.filter((UserLexeme.a_click * 1.0) / (denom * 1.0) >= float(min_p))
        if max_p is not None:
            q = q.filter((UserLexeme.a_click * 1.0) / (denom * 1.0) <= float(max_p))
    rows = q.limit(1000).all()
    out: List[SrsWordsOut] = []
    for ul, lx, li in rows:
        a = ul.a_click or 0
        b = ul.b_nonclick or 0
        p = (a / (a + b)) if (a + b) > 0 else 0.0
        S = float(ul.stability or 0.0)
        D = int(ul.distinct_texts or 0)
        if min_p is not None and p < min_p: continue
        if max_p is not None and p > max_p: continue
        if min_S is not None and S < min_S: continue
        if max_S is not None and S > max_S: continue
        if min_D is not None and D < min_D: continue
        if max_D is not None and D > max_D: continue
        out.append(SrsWordsOut(lexeme_id=lx.id, lemma=lx.lemma, pos=lx.pos, p_click=p, n=(a+b), stability=S, diversity=D, freq_rank=(li.freq_rank if li else None), level_code=(li.level_code if li else None)))
        if len(out) >= limit:
            break
    return out


class SrsStatsOut(BaseModel):
    total: int
    by_p: Dict[str, int]
    by_S: Dict[str, int]
    by_D: Dict[str, int]


@app.get("/srs/stats", response_model=SrsStatsOut)
def get_srs_stats(lang: str, db: Session = Depends(get_db), user: User = Depends(_get_current_user)):
    init_db()
    pid = db.query(Profile.id).filter(Profile.user_id == user.id, Profile.lang == lang).scalar()
    if not pid:
        return SrsStatsOut(total=0, by_p={}, by_S={}, by_D={})
    rows = db.query(UserLexeme, Lexeme).join(Lexeme, UserLexeme.lexeme_id == Lexeme.id).filter(UserLexeme.user_id == user.id, UserLexeme.profile_id == pid).all()
    def bucketize(val: float, bounds: List[float]):
        for i, b in enumerate(bounds):
            if val < b: return f"<{b}"
        return f">={bounds[-1]}"
    by_p: Dict[str, int] = {}
    by_S: Dict[str, int] = {}
    by_D: Dict[str, int] = {}
    for ul, lx in rows:
        a = ul.a_click or 0
        b = ul.b_nonclick or 0
        p = (a / (a + b)) if (a + b) > 0 else 0.0
        S = float(ul.stability or 0.0)
        D = int(ul.distinct_texts or 0)
        pb = bucketize(p, [0.2, 0.4, 0.6, 0.8])
        Sb = bucketize(S, [0.33, 0.66, 0.85])
        Db = bucketize(min(1.0, 1 - pow(2.71828, -D / 8.0)), [0.25, 0.5, 0.75])
        by_p[pb] = by_p.get(pb, 0) + 1
        by_S[Sb] = by_S.get(Sb, 0) + 1
        by_D[Db] = by_D.get(Db, 0) + 1
    return SrsStatsOut(total=len(rows), by_p=by_p, by_S=by_S, by_D=by_D)


class LevelOut(BaseModel):
    level_value: float
    level_var: float
    last_update_at: Optional[str]
    last_activity_at: Optional[str]
    ess: float
    bins: Dict[str, List[float]]


@app.get("/srs/level", response_model=LevelOut)
def get_srs_level(lang: str, db: Session = Depends(get_db), user: User = Depends(_get_current_user)):
    init_db()
    # best-effort freshness
    try:
        update_level_if_stale(db, user.id, lang)
        db.commit()
    except Exception:
        pass
    prof = db.query(Profile).filter(Profile.user_id == user.id, Profile.lang == lang).first()
    if not prof:
        raise HTTPException(404, "profile not found")
    summ = get_level_summary(db, prof)
    return LevelOut(**summ)


# -------- Translation API --------
class TranslateIn(BaseModel):
    lang: str
    target_lang: Optional[str] = "en"
    unit: Literal["sentence", "paragraph", "text"]
    text: Optional[str] = None
    text_id: Optional[int] = None
    start: Optional[int] = None
    end: Optional[int] = None
    continue_with_reading: Optional[bool] = False
    provider: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None


def _sentence_spans(text: str, lang: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    n = len(text)
    i = 0
    enders = [".", "!", "?"]
    zh_enders = ["。", "！", "？"]
    if lang.startswith("zh"):
        # split on zh punctuation; include the punctuation in span
        start = 0
        while i < n:
            ch = text[i]
            if ch in zh_enders:
                spans.append((start, i + 1))
                start = i + 1
            i += 1
        if start < n:
            spans.append((start, n))
        return [(s, e) for (s, e) in spans if e > s and text[s:e].strip()]
    # Latin-like
    start = 0
    while i < n:
        ch = text[i]
        if ch in enders:
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            spans.append((start, j))
            start = j
            i = j
            continue
        i += 1
    if start < n:
        spans.append((start, n))
    return [(s, e) for (s, e) in spans if e > s and text[s:e].strip()]


def _paragraph_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    n = len(text)
    i = 0
    start = 0
    while i < n:
        if text[i] == "\n":
            # detect blank line
            j = i
            while j < n and text[j] == "\n":
                j += 1
            if j - i >= 2:
                if start < i:
                    spans.append((start, i))
                start = j
                i = j
                continue
        i += 1
    if start < n:
        spans.append((start, n))
    return [(s, e) for (s, e) in spans if e > s and text[s:e].strip()]


def _parse_xml_translations(xml_text: str) -> Tuple[str, List[str]]:
    """Parse XML output; returns (mode, items)
    mode: 'single' for <translation>, 'multi' for <translations>.<seg>
    items: list of strings (one item for single as [text]).
    """
    import xml.etree.ElementTree as ET
    try:
        xml_text = xml_text.strip()
        root = ET.fromstring(xml_text)
        tag = (root.tag or '').lower()
        if tag.endswith('translation') and tag != 'translations':
            return 'single', [ (root.text or '').strip() ]
        if tag.endswith('translations'):
            out: List[str] = []
            for seg in list(root):
                if (seg.tag or '').lower().endswith('seg'):
                    out.append((seg.text or '').strip())
            return 'multi', out
    except Exception:
        pass
    # fallback: treat as single chunk
    return 'single', [xml_text.strip()]


def _assemble_prev_messages(db: Session, user: User, text_id: Optional[int]) -> Optional[List[Dict[str, str]]]:
    if not text_id:
        return None
    msgs: List[Dict[str, str]] = []
    # Base conversation from reading generation
    gl = db.query(GenerationLog).filter(GenerationLog.text_id == text_id).order_by(GenerationLog.id.asc()).all()
    if gl:
        # take earliest prompt (closest to generation) and assistant text as a seed
        first = gl[0]
        if isinstance(first.prompt, dict):
            base = first.prompt.get("messages")
            if isinstance(base, list):
                for m in base[-4:]:
                    if isinstance(m, dict) and m.get("content"):
                        msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        rt = db.get(ReadingText, text_id)
        if rt and rt.content:
            msgs.append({"role": "assistant", "content": rt.content})
    # Append last few translation exchanges (user + assistant) if available
    tlogs = db.query(TranslationLog).filter(TranslationLog.text_id == text_id).order_by(TranslationLog.id.desc()).limit(3).all()
    for tl in reversed(tlogs):
        try:
            pm = tl.prompt.get("messages") if isinstance(tl.prompt, dict) else None
            if isinstance(pm, list) and pm:
                # take the last user content from that prompt
                last_user = None
                for m in reversed(pm):
                    if isinstance(m, dict) and m.get("role") == "user":
                        last_user = m.get("content", "")
                        break
                if last_user:
                    msgs.append({"role": "user", "content": last_user})
            if getattr(tl, 'response', None):
                msgs.append({"role": "assistant", "content": tl.response})
        except Exception:
            continue
    return msgs or None


@app.post("/translate")
def translate(payload: TranslateIn, db: Session = Depends(get_db), user: User = Depends(_get_current_user)):
    init_db()
    # Resolve source content
    raw: Optional[str] = payload.text
    base_offset = 0
    if raw is None and payload.text_id is not None:
        rt = db.get(ReadingText, payload.text_id)
        if not rt or rt.user_id != user.id:
            raise HTTPException(404, "reading text not found")
        raw = rt.content
        if payload.start is not None or payload.end is not None:
            s = max(0, int(payload.start or 0))
            e = min(len(raw), int(payload.end or len(raw)))
            if e <= s:
                raise HTTPException(400, "invalid span")
            base_offset = s
            raw = raw[s:e]
    if raw is None:
        raise HTTPException(400, "text or text_id required")

    # Build segments
    unit = payload.unit
    spans: List[Tuple[int, int]]
    if unit == "text":
        spans = [(0, len(raw))]
    elif unit == "sentence":
        spans = _sentence_spans(raw, payload.lang)
    elif unit == "paragraph":
        spans = _paragraph_spans(raw)
    else:
        raise HTTPException(400, "invalid unit")
    segments = [raw[s:e] for (s, e) in spans]

    # Previous conversation (optional)
    prev_msgs: Optional[List[Dict[str, str]]] = None
    if payload.continue_with_reading and payload.text_id:
        prev_msgs = _assemble_prev_messages(db, user, payload.text_id)

    spec = TranslationSpec(
        lang=payload.lang,
        target_lang=(payload.target_lang or "en"),
        unit=unit,
        content=(segments if len(segments) > 1 else segments[0]),
        continue_with_reading=bool(payload.continue_with_reading),
        script=None,
    )
    messages = build_translation_prompt(spec, prev_messages=prev_msgs)

    def _call_llm(msgs: List[Dict[str, str]]) -> str:
        return chat_complete(
            msgs,
            provider=payload.provider,
            model=payload.model,
            base_url=(payload.base_url or "http://localhost:1234/v1"),
            temperature=0.3,
        )

    translations: List[str]
    raw_response: str
    out = _call_llm(messages)
    raw_response = out
    mode, items_parsed = _parse_xml_translations(out)
    if isinstance(spec.content, list):
        if len(items_parsed) != len(segments):
            # retry per-segment in XML mode
            items_parsed = []
            for seg in segments:
                m = build_translation_prompt(TranslationSpec(lang=spec.lang, target_lang=spec.target_lang, unit="sentence", content=seg), prev_messages=prev_msgs)
                r = _call_llm(m)
                raw_response += "\n" + r
                _, one = _parse_xml_translations(r)
                items_parsed.append(one[0] if one else "")
    translations = [s.strip() for s in items_parsed]

    items = []
    for idx, (span, src, tr) in enumerate(zip(spans, segments, translations)):
        s0 = base_offset + span[0]
        e0 = base_offset + span[1]
        items.append({"start": s0, "end": e0, "source": src, "translation": tr})

    # Persist when translating a stored reading
    if payload.text_id is not None:
        from sqlalchemy.exc import IntegrityError  # local import
        for idx, (span, src, tr) in enumerate(zip(spans, segments, translations)):
            row = ReadingTextTranslation(
                user_id=user.id,
                text_id=payload.text_id,
                unit=unit,
                target_lang=(payload.target_lang or "en"),
                segment_index=(idx if unit != "text" else None),
                span_start=(base_offset + span[0]),
                span_end=(base_offset + span[1]),
                source_text=src,
                translated_text=tr,
                provider=(payload.provider or "openrouter"),
                model=payload.model,
            )
            try:
                db.add(row)
                db.flush()
            except IntegrityError:
                db.rollback()
        # minimal log
        try:
            db.add(TranslationLog(
                user_id=user.id,
                text_id=payload.text_id,
                unit=unit,
                target_lang=(payload.target_lang or "en"),
                provider=(payload.provider or "openrouter"),
                model=payload.model,
                prompt={"messages": messages},
                segments={"count": len(segments)},
                response=raw_response,
            ))
        except Exception:
            pass
        db.commit()

    return {
        "unit": unit,
        "target_lang": (payload.target_lang or "en"),
        "items": items,
        "provider": (payload.provider or "openrouter"),
        "model": payload.model,
    }


@app.get("/reading/{text_id}/translations")
def get_translations(text_id: int, unit: Literal["sentence", "paragraph", "text"], target_lang: str = "en", db: Session = Depends(get_db), user: User = Depends(_get_current_user)):
    init_db()
    rt = db.get(ReadingText, text_id)
    if not rt or rt.user_id != user.id:
        raise HTTPException(404, "reading text not found")
    rows = (
        db.query(ReadingTextTranslation)
        .filter(
            ReadingTextTranslation.text_id == text_id,
            ReadingTextTranslation.user_id == user.id,
            ReadingTextTranslation.unit == unit,
            ReadingTextTranslation.target_lang == target_lang,
        )
        .order_by(ReadingTextTranslation.segment_index.asc().nullsfirst(), ReadingTextTranslation.span_start.asc().nullslast())
        .all()
    )
    items = [
        {
            "start": r.span_start,
            "end": r.span_end,
            "source": r.source_text,
            "translation": r.translated_text,
        }
        for r in rows
    ]
    return {"unit": unit, "target_lang": target_lang, "items": items}


# ---- UI shell mounting (header/footer/templates) ----
try:
    # Optional UI libraries from libs/
    from arcadia_ui_style import ensure_templates  # type: ignore
    from arcadia_ui_core import router as ui_router, attach_ui, mount_ui_static  # type: ignore
    tdir = ensure_templates(Path(__file__).resolve().parent)
    templates = Jinja2Templates(directory=tdir)
    # Attach UI state and mount static assets
    attach_ui(
        app,
        templates,
        persist_header=True,
        brand_home_url="/",
        brand_name="Arcadia Lang",
        brand_tag="",
    )
    mount_ui_static(app)
    _templates_env = templates  # share env for project pages
    app.include_router(ui_router)
except Exception:
    # non-fatal during dev
    pass

# Words browser page (server-rendered HTML using shared header/footer)
@app.get("/words", response_class=HTMLResponse)
def words_page(request: Request, lang: Optional[str] = None, user: User = Depends(_get_current_user)):
    # Render plain template; client fetches data via /srs/words
    t = _templates()
    return t.TemplateResponse("words.html", {"request": request, "lang": lang or "es"})

# Login/Signup pages are provided by arcadia_ui_core router

# Placeholder pages for account menu
@app.get("/profile", response_class=HTMLResponse)
def profile_page(request: Request, user: User = Depends(_get_current_user)):
    t = _templates()
    return t.TemplateResponse("profile.html", {"request": request})


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, user: User = Depends(_get_current_user)):
    t = _templates()
    return t.TemplateResponse("settings.html", {"request": request})


@app.get("/stats", response_class=HTMLResponse)
def stats_page(request: Request, user: User = Depends(_get_current_user)):
    t = _templates()
    return t.TemplateResponse("stats.html", {"request": request})


@app.get("/logout")
def logout() -> RedirectResponse:
    resp = RedirectResponse(url="/", status_code=302)
    try:
        resp.delete_cookie("access_token", path="/")
    except Exception:
        pass
    return resp

# Static site (frontend build output). Mount after custom HTML routes so they take precedence.
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
