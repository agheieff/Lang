from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List, Callable
import os
import time
from collections import deque, defaultdict

from fastapi import FastAPI, HTTPException, Depends, Request, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Language engine
from Lang.parsing.registry import ENGINES
from Lang.parsing.morph_format import format_morph_label
from Lang.parsing.dicts.provider import DictionaryProviderChain, StarDictProvider
from Lang.parsing.dicts.cedict import CedictProvider
from .db import get_db, init_db
from .models import User, Profile, SubscriptionTier, ProfilePref, Lexeme, UserLexeme, WordEvent, UserLexemeContext, LexemeInfo, LexemeVariant
from sqlalchemy.orm import Session
import jwt
from datetime import datetime, timedelta
from argon2 import PasswordHasher
from Lang.tokenize.registry import TOKENIZERS
from Lang.tokenize.base import Token
from .llm import PromptSpec, build_reading_prompt, chat_complete, pick_words, estimate_level

# SRS parameters
_SRS_ALPHA_CLICK = 0.2
_SRS_BETA_EXPOSURE = 0.02
_DIVERSITY_K = 8.0
_SRS_BETA_NONLOOKUP = 2  # extra weight to b_nonclick for explicit non-lookup confirmations
_SRS_GAMMA_NONLOOKUP = 0.08  # stability boost factor for non-lookup


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


def _hash_context(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    try:
        import hashlib
        h = hashlib.sha256(text.encode('utf-8')).hexdigest()
        return h[:32]
    except Exception:
        return None


def _resolve_lexeme(db: Session, lang: str, lemma: str, pos: Optional[str]) -> Lexeme:
    # For Chinese, normalize to unified 'zh' with simplified lemma; capture Hans/Hant variants
    is_zh = lang.startswith("zh")
    canon_lang = "zh" if is_zh else lang
    canon_lemma = lemma
    hans_form = None
    hant_form = None
    if is_zh:
        try:
            from opencc import OpenCC  # type: ignore
            hans = OpenCC("t2s").convert(lemma)
            hant = OpenCC("s2t").convert(hans)
            canon_lemma = hans
            hans_form = hans
            hant_form = hant
        except Exception:
            # fallback: assume provided lemma is simplified
            hans_form = lemma
    lex = db.query(Lexeme).filter(Lexeme.lang == canon_lang, Lexeme.lemma == canon_lemma, Lexeme.pos == pos).first()
    if not lex:
        lex = Lexeme(lang=canon_lang, lemma=canon_lemma, pos=pos)
        db.add(lex)
        db.flush()
    if is_zh:
        # Always ensure Hans variant for simplified
        if hans_form:
            exists_hans = db.query(LexemeVariant).filter(LexemeVariant.lexeme_id == lex.id, LexemeVariant.script == "Hans", LexemeVariant.form == hans_form).first()
            if not exists_hans:
                db.add(LexemeVariant(lexeme_id=lex.id, script="Hans", form=hans_form))
        # Ensure Hant variant
        if hant_form:
            exists_hant = db.query(LexemeVariant).filter(LexemeVariant.lexeme_id == lex.id, LexemeVariant.script == "Hant", LexemeVariant.form == hant_form).first()
            if not exists_hant:
                db.add(LexemeVariant(lexeme_id=lex.id, script="Hant", form=hant_form))
        # Also record the original as variant with detected script from lang
        script = "Hant" if lang.endswith("Hant") else "Hans"
        exists_orig = db.query(LexemeVariant).filter(LexemeVariant.lexeme_id == lex.id, LexemeVariant.script == script, LexemeVariant.form == lemma).first()
        if not exists_orig:
            db.add(LexemeVariant(lexeme_id=lex.id, script=script, form=lemma))
    db.commit()
    db.refresh(lex)
    return lex


def _get_or_create_userlexeme(db: Session, user: User, profile: Profile, lexeme: Lexeme) -> UserLexeme:
    ul = db.query(UserLexeme).filter(UserLexeme.user_id == user.id, UserLexeme.profile_id == profile.id, UserLexeme.lexeme_id == lexeme.id).first()
    if ul:
        return ul
    ul = UserLexeme(user_id=user.id, profile_id=profile.id, lexeme_id=lexeme.id)
    db.add(ul)
    db.commit()
    db.refresh(ul)
    return ul


def _srs_click(db: Session, user: User, lang: str, lemma: str, pos: Optional[str], surface: Optional[str], context_hash: Optional[str]):
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
    ul.stability = float(max(0.0, min(1.0, (ul.stability or 0.0) * (1.0 - _SRS_ALPHA_CLICK))))
    # Event row
    ev = WordEvent(ts=datetime.utcnow(), user_id=user.id, profile_id=prof.id, lexeme_id=lex.id, event_type="click", count=1, surface=surface, context_hash=context_hash, source="manual", meta={})
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
    # For Chinese, if no translations on lemma, try the surface form too (handles un-normalized variants)
    if not translations and req.source_lang.startswith("zh") and req.surface != lemma:
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
            data = jwt.decode(auth.split(" ", 1)[1], _JWT_SECRET, algorithms=["HS256"])  # type: ignore
            uid = int(data.get("sub"))
            user = db.get(User, uid)
            if user:
                _srs_click(db, user, req.source_lang, lemma=lemma, pos=analysis.get("pos"), surface=req.surface, context_hash=None)
    except Exception:
        pass
    return resp
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


def _get_current_user(db: Session = Depends(get_db), authorization: Optional[str] = Header(default=None)) -> User:
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


class MeOut(BaseModel):
    id: int
    email: str
    subscription_tier: str


@app.get("/me", response_model=MeOut)
def get_me(user: User = Depends(_get_current_user)):
    return MeOut(id=user.id, email=user.email, subscription_tier=user.subscription_tier)


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


# -------- LLM reading generation --------
class GenRequest(BaseModel):
    lang: str
    length: Optional[int] = None  # UI length hint; unit is decided per-language
    include_words: Optional[List[str]] = None
    model: Optional[str] = None
    base_url: str = "http://localhost:1234/v1"


@app.post("/gen/reading")
def gen_reading(req: GenRequest, db: Session = Depends(get_db), user: User = Depends(_get_current_user)):
    init_db()
    script = "Hant" if req.lang.endswith("Hant") else ("Hans" if req.lang.startswith("zh") else None)
    # Auto-pick words if not provided
    words = req.include_words or pick_words(db, user, req.lang, count=12)
    level_hint = estimate_level(db, user, req.lang)
    # Per-language unit and default length
    unit = "chars" if req.lang.startswith("zh") else "words"
    approx_len = req.length if req.length is not None else (300 if unit == "chars" else 180)
    spec = PromptSpec(lang=req.lang, unit=unit, approx_len=approx_len, user_level_hint=level_hint, include_words=words, script=script)
    messages = build_reading_prompt(spec)
    try:
        text = chat_complete(req.base_url, req.model, messages)
    except Exception as e:
        raise HTTPException(502, f"LLM generation failed: {e}")
    return {"prompt": messages, "text": text, "level_hint": level_hint, "words": words}


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
    lemma: str
    pos: Optional[str] = None
    surface: Optional[str] = None
    context: Optional[str] = None


class ExposuresRequest(BaseModel):
    lang: str
    items: List[ExposureItem]


def _srs_exposure(db: Session, user: User, lang: str, lemma: str, pos: Optional[str], surface: Optional[str], context_hash: Optional[str]):
    prof = db.query(Profile).filter(Profile.user_id == user.id, Profile.lang == lang).first()
    if not prof:
        prof = Profile(user_id=user.id, lang=lang)
        db.add(prof)
        db.commit()
        db.refresh(prof)
    lex = _resolve_lexeme(db, lang, lemma, pos)
    ul = _get_or_create_userlexeme(db, user, prof, lex)
    # Beta non-click increment
    ul.b_nonclick = (ul.b_nonclick or 0) + 1
    ul.exposures = (ul.exposures or 0) + 1
    now = datetime.utcnow()
    if not ul.first_seen_at:
        ul.first_seen_at = now
    ul.last_seen_at = now
    # Stability increase
    ul.stability = float(max(0.0, min(1.0, (ul.stability or 0.0) + _SRS_BETA_EXPOSURE * (1.0 - (ul.stability or 0.0)))))
    # Diversity (distinct contexts)
    if context_hash:
        exists = db.query(UserLexemeContext).filter(UserLexemeContext.user_lexeme_id == ul.id, UserLexemeContext.context_hash == context_hash).first()
        if not exists:
            db.add(UserLexemeContext(user_lexeme_id=ul.id, context_hash=context_hash))
            ul.distinct_texts = (ul.distinct_texts or 0) + 1
    # Event
    db.add(WordEvent(ts=now, user_id=user.id, profile_id=prof.id, lexeme_id=lex.id, event_type="exposure", count=1, surface=surface, context_hash=context_hash, source="manual", meta={}))


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
        _srs_exposure(db, user, req.lang, lemma, pos, it.surface, _hash_context(it.context))
        count += 1
    db.commit()
    return {"ok": True, "count": count}


class NonLookupRequest(BaseModel):
    lang: str
    items: List[ExposureItem]


def _srs_nonlookup(db: Session, user: User, lang: str, lemma: str, pos: Optional[str], surface: Optional[str], context_hash: Optional[str]):
    # ensure profile and lexeme
    prof = db.query(Profile).filter(Profile.user_id == user.id, Profile.lang == lang).first()
    if not prof:
        prof = Profile(user_id=user.id, lang=lang)
        db.add(prof)
        db.commit()
        db.refresh(prof)
    lex = _resolve_lexeme(db, lang, lemma, pos)
    ul = _get_or_create_userlexeme(db, user, prof, lex)
    # Treat as an explicit non-click success: increase evidence against clicking and boost stability more strongly
    ul.b_nonclick = (ul.b_nonclick or 0) + _SRS_BETA_NONLOOKUP
    now = datetime.utcnow()
    ul.last_seen_at = now
    ul.stability = float(max(0.0, min(1.0, (ul.stability or 0.0) + _SRS_GAMMA_NONLOOKUP * (1.0 - (ul.stability or 0.0)))))
    db.add(WordEvent(ts=now, user_id=user.id, profile_id=prof.id, lexeme_id=lex.id, event_type="nonlookup", count=1, surface=surface, context_hash=context_hash, source="manual", meta={}))


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
        _srs_nonlookup(db, user, req.lang, lemma, pos, it.surface, _hash_context(it.context))
        count += 1
    db.commit()
    return {"ok": True, "count": count}


class ClickRequest(BaseModel):
    lang: str
    lemma: Optional[str] = None
    pos: Optional[str] = None
    surface: Optional[str] = None
    context: Optional[str] = None


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
    _srs_click(db, user, req.lang, lemma, pos, req.surface, _hash_context(req.context))
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


# Static site (frontend build output). We mount after API routes so they take priority.
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
