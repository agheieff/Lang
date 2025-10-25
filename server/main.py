from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List, Callable, Literal, Tuple
import os
import time
from collections import deque, defaultdict
import math

from fastapi import FastAPI, HTTPException, Depends, Request, Query, Header
from contextlib import asynccontextmanager
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
from .deps import get_current_account as _get_current_account, require_tier
from .api.profile import router as profile_router
from .api.wordlists import router as wordlists_router
from arcadia_auth import Account
from .models import Profile, SubscriptionTier, ProfilePref, Lexeme, UserLexeme, WordEvent, UserLexemeContext, LexemeInfo, LexemeVariant, ReadingText, GenerationLog, ReadingTextTranslation, TranslationLog, LLMModel, ReadingLookup, LLMRequestLog
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
    surface: str = Field(..., description="Surface form as clicked in text")
    context: Optional[str] = Field(None, description="Optional sentence context for disambiguation")
    text_id: Optional[int] = Field(None, description="Reading text id for caching")
    start: Optional[int] = Field(None, description="Start offset within text")
    end: Optional[int] = Field(None, description="End offset within text (exclusive)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database once at startup
    init_db()
    yield


app = FastAPI(lifespan=lifespan, title="Arcadia Lang", version="0.1.0")

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

# Mount API routers
app.include_router(profile_router)
app.include_router(wordlists_router, prefix="/api")

# Simple per-minute rate limit by IP or user tier for heavy endpoints
# TODO(deploy): tighten rate limits per tier; for local/dev, set effectively unlimited.
_RATE_LIMITS = {"free": 1_000_000_000, "premium": 1_000_000_000, "pro": 1_000_000_000}  # requests/minute; admin unlimited via bypass
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
                    account = db.get(Account, int(uid))
                    if account and account.subscription_tier in _RATE_LIMITS:
                        tier = account.subscription_tier
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


def _log_llm_request_safe(entry: dict) -> None:
    """Log LLM request/response using an isolated short-lived DB session."""
    from .db import SessionLocal as _SessionLocal
    from .models import LLMRequestLog as _LLMRequestLog
    s = _SessionLocal()
    try:
        s.add(_LLMRequestLog(**entry))
        s.commit()
    except Exception:
        try:
            s.rollback()
        except Exception:
            pass
    finally:
        try:
            s.close()
        except Exception:
            pass


def _clamp(v: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return lo


def _get_ci_target(db: Session, account: Account, lang: str) -> float:
    """Return the user's comprehensible input target (0..1) for the language.

    Reads ProfilePref.settings.ci_target when available; otherwise falls back to env default.
    """
    try:
        default = float(os.getenv("ARC_CI_DEFAULT", "0.95"))
    except Exception:
        default = 0.95
    ci_min = float(os.getenv("ARC_CI_MIN", "0.8"))
    ci_max = float(os.getenv("ARC_CI_MAX", "0.99"))
    prof = db.query(Profile).filter(Profile.account_id == account.id, Profile.lang == lang).first()
    if not prof:
        return _clamp(default, ci_min, ci_max)
    pref = db.query(ProfilePref).filter(ProfilePref.profile_id == prof.id).first()
    if not pref or not isinstance(pref.data, dict):
        return _clamp(default, ci_min, ci_max)
    ci = pref.data.get("ci_target")
    if ci is None:
        # also allow nested settings
        st = pref.data.get("settings") if isinstance(pref.data.get("settings"), dict) else None
        if st and isinstance(st.get("ci_target"), (int, float)):
            ci = st.get("ci_target")
    try:
        return _clamp(float(ci) if ci is not None else default, ci_min, ci_max)
    except Exception:
        return _clamp(default, ci_min, ci_max)


def _normalize_lemma_for_lang(lang: str, surface: str) -> str:
    if lang.startswith("zh"):
        try:
            # Prefer simplified as canonical
            return _OPENCC_T2S.convert(surface) if _OPENCC_T2S else surface
        except Exception:
            return surface
    # Latin-like: lowercase
    return surface.lower()


def _estimate_familiar_share(db: Session, account: Account, lang: str, text: str) -> float:
    """Best-effort estimate of familiar token share based on user lexicon and priors.

    Uses tokenizer for the language; maps tokens to lemmas; batches DB lookups for Lexeme, UserLexeme, and LexemeInfo.
    """
    try:
        tok = TOKENIZERS.get(lang, TOKENIZERS.get("default"))
    except Exception:
        tok = TOKENIZERS["default"]
    words = tok.tokenize(text)
    # Collect lemmas per token (best-effort)
    lemmas: list[str] = []
    lemma_set: set[str] = set()
    engine = _ENGINES.get(lang)
    for w in words:
        if not getattr(w, "is_word", False):
            continue
        s = w.text or ""
        if not s.strip():
            continue
        lemma = None
        try:
            if engine:
                a = engine.analyze_word(s, context=None)
                lemma = a.get("lemma") or None
        except Exception:
            lemma = None
        lemma = lemma or _normalize_lemma_for_lang(lang, s)
        lemmas.append(lemma)
        if len(lemma_set) < 2000:
            lemma_set.add(lemma)

    if not lemmas:
        return 1.0

    canon_lang = "zh" if lang.startswith("zh") else lang
    # Fetch lexemes for lemmas
    Lx = db.query(Lexeme).filter(Lexeme.lang == canon_lang, Lexeme.lemma.in_(list(lemma_set))).all()
    lemma_to_lx: dict[str, Lexeme] = {lx.lemma: lx for lx in Lx}
    lx_ids = [lx.id for lx in Lx]
    # Profile
    prof = db.query(Profile).filter(Profile.account_id == account.id, Profile.lang == lang).first()
    pid = prof.id if prof else None
    ul_by_id: dict[int, UserLexeme] = {}
    if pid and lx_ids:
        for ul in db.query(UserLexeme).filter(UserLexeme.account_id == account.id, UserLexeme.profile_id == pid, UserLexeme.lexeme_id.in_(lx_ids)).all():
            ul_by_id[ul.lexeme_id] = ul
    li_by_id: dict[int, LexemeInfo] = {}
    if lx_ids:
        for li in db.query(LexemeInfo).filter(LexemeInfo.lexeme_id.in_(lx_ids)).all():
            li_by_id[li.lexeme_id] = li

    # User level
    user_level = float(getattr(prof, "level_value", 0.0) or 0.0) if prof else 0.0
    alpha = float(os.getenv("ARC_CI_ALPHA", "0.6"))
    sigma = float(os.getenv("ARC_CI_SIGMA", "1.0"))
    w1 = float(os.getenv("ARC_CI_W1", "2.0"))
    w2 = float(os.getenv("ARC_CI_W2", "0.2"))
    w3 = float(os.getenv("ARC_CI_W3", "0.6"))

    def p_from_ul(ul: UserLexeme) -> float:
        a = float(getattr(ul, "a_click", 0) or 0)
        b = float(getattr(ul, "b_nonclick", 0) or 0)
        den = max(0.0, a + b)
        r = (a / den) if den > 0 else 0.0
        S = float(getattr(ul, "stability", 0.0) or 0.0)
        exp_count = float(getattr(ul, "exposures", 0) or 0)
        s = (w1 * r) + (w2 * math.log1p(exp_count)) + (w3 * S)
        try:
            return 1.0 / (1.0 + math.exp(-s))
        except Exception:
            return 0.9 if r > 0.6 else 0.5

    def p_from_prior(li: Optional[LexemeInfo], lemma: str) -> float:
        # Level proximity (HSK for zh when available)
        p_lvl = 0.5
        if canon_lang == "zh" and li and getattr(li, "level_code", None):
            try:
                code = (li.level_code or "").upper()
                if code.startswith("HSK"):
                    lvl = int(code[3:])
                    center = max(1.0, min(6.0, float(user_level) if user_level > 0 else 1.0))
                    d = abs(float(lvl) - center)
                    p_lvl = math.exp(-0.5 * (d / max(0.1, sigma)) ** 2)
            except Exception:
                p_lvl = 0.5
        # Frequency prior when available
        p_freq = 0.5
        if li and getattr(li, "freq_rank", None):
            try:
                p_freq = 1.0 / (1.0 + (float(li.freq_rank) / 5000.0))
            except Exception:
                p_freq = 0.5
        p = (alpha * p_lvl) + ((1.0 - alpha) * p_freq)
        return _clamp(p, 0.05, 0.99)

    # Precompute per-lemma probabilities
    p_by_lemma: dict[str, float] = {}
    for lem in lemma_set:
        lx = lemma_to_lx.get(lem)
        if lx and lx.id in ul_by_id:
            p_by_lemma[lem] = p_from_ul(ul_by_id[lx.id])
        else:
            li = li_by_id.get(lx.id) if lx else None
            p_by_lemma[lem] = p_from_prior(li, lem)

    # Aggregate across occurrences
    vals = [p_by_lemma.get(l, 0.2) for l in lemmas]
    return sum(vals) / float(len(vals) or 1)


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


def _get_or_create_userlexeme(db: Session, account: Account, profile: Profile, lexeme: Lexeme) -> UserLexeme:
    ul = db.query(UserLexeme).filter(UserLexeme.account_id == account.id, UserLexeme.profile_id == profile.id, UserLexeme.lexeme_id == lexeme.id).first()
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
    ul = UserLexeme(account_id=account.id, profile_id=profile.id, lexeme_id=lexeme.id, importance=imp)
    db.add(ul)
    db.flush()
    return ul


def _srs_click(db: Session, account: Account, lang: str, lemma: str, pos: Optional[str], surface: Optional[str], context_hash: Optional[str], text_id: Optional[int] = None):
    # Pick or create profile for the lang
    prof = db.query(Profile).filter(Profile.account_id == account.id, Profile.lang == lang).first()
    if not prof:
        prof = Profile(account_id=account.id, lang=lang)
        db.add(prof)
        db.flush()
    lex = _resolve_lexeme(db, lang, lemma, pos)
    ul = _get_or_create_userlexeme(db, account, prof, lex)
    # Update Beta and stability
    ul.a_click = (ul.a_click or 0) + 1
    ul.clicks = (ul.clicks or 0) + 1
    ul.last_clicked_at = datetime.utcnow()
    now = datetime.utcnow()
    _decay_posterior(ul, now, _HL_CLICK_D)
    ul.alpha = float((ul.alpha or 1.0) + _W_CLICK)
    _schedule_next(ul, 0, now)
    # Event row
    ev = WordEvent(ts=now, account_id=account.id, profile_id=prof.id, lexeme_id=lex.id, event_type="click", count=1, surface=surface, context_hash=context_hash, source="manual", meta={}, text_id=text_id)
    db.add(ev)
    # Commit happens in caller endpoint


@app.post("/api/lookup")
def lookup(req: LookupRequest, request: Request, db: Session = Depends(get_db)) -> Dict[str, Any]:
    engine = _ENGINES.get(req.source_lang)
    if not engine:
        raise HTTPException(status_code=400, detail="language not supported yet")

    # Get user and profile to determine target language
    user = None
    target_lang = "en"  # default fallback
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        try:
            data = decode_token(auth.split(" ", 1)[1], _JWT_SECRET, ["HS256"])  # type: ignore
            uid = int((data or {}).get("sub")) if data and data.get("sub") is not None else None
            if uid is not None:
                account = db.get(Account, uid)
                if account:
                    profile = db.query(Profile).filter(Profile.account_id == account.id, Profile.lang == req.source_lang).first()
                    if profile and profile.target_lang:
                        target_lang = profile.target_lang
        except Exception:
            pass

    # Analyze word (lemma, pos, morph)
    analysis = engine.analyze_word(req.surface, context=req.context)
    lemma = analysis.get("lemma") or req.surface
    morph = analysis.get("morph") or {}
    label = format_morph_label(analysis.get("pos"), morph)

    translations: Dict[str, Any] | list | None = None
    mode = "analysis"

    # If authorized and text_id/start/end provided, try cache
    uid: Optional[int] = None
    rt: Optional[ReadingText] = None
    try:
        auth = request.headers.get("authorization") or request.headers.get("Authorization")
        if auth and auth.lower().startswith("bearer "):
            data = decode_token(auth.split(" ", 1)[1], _JWT_SECRET, ["HS256"])  # type: ignore
            uid = int((data or {}).get("sub")) if data and data.get("sub") is not None else None
    except Exception:
        uid = None

    context_hash = _hash_context(req.context)
    # Best-effort span validation
    if req.text_id is not None and req.start is not None and req.end is not None and user is not None:
        try:
            rt = db.get(ReadingText, int(req.text_id))
            if rt and rt.user_id == user.id:
                s = max(0, int(req.start))
                e = int(req.end)
                if e > s:
                    if rt.content:
                        e = min(e, len(rt.content))
                    # Try cache read
                    row = (
                        db.query(ReadingLookup)
                        .filter(
                            ReadingLookup.user_id == user.id,
                            ReadingLookup.text_id == rt.id,
                            ReadingLookup.target_lang == target_lang,
                            ReadingLookup.span_start == s,
                            ReadingLookup.span_end == e,
                        )
                        .first()
                    )
                    if row and row.translations:
                        translations = row.translations
                        mode = "translation"
        except Exception:
            pass

    # Compute translations if not from cache
    if translations is None:
        tr = get_dict_chain().translations(req.source_lang, target_lang, lemma)
        if not tr and req.surface != lemma:
            more = get_dict_chain().translations(req.source_lang, target_lang, req.surface)
            if more:
                tr = more
        translations = tr or []
        mode = "translation" if translations else "analysis"
        # Persist lookup if we have ownership context
        if user is not None and rt is not None and req.start is not None and req.end is not None:
            try:
                s = max(0, int(req.start))
                e = int(req.end)
                if rt.content:
                    e = min(e, len(rt.content))
                from sqlalchemy.dialects.sqlite import insert
                from .models import ReadingLookup as RL
                stmt = insert(RL).values(
                    user_id=user.id,
                    text_id=rt.id,
                    lang=req.source_lang,
                    target_lang=target_lang,
                    surface=req.surface,
                    lemma=lemma,
                    pos=analysis.get("pos"),
                    span_start=s,
                    span_end=e,
                    context_hash=context_hash,
                    translations=(translations if isinstance(translations, (dict, list)) else {}),
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=[RL.user_id, RL.text_id, RL.target_lang, RL.span_start, RL.span_end],
                    set_={
                        "surface": stmt.excluded.surface,
                        "lemma": stmt.excluded.lemma,
                        "pos": stmt.excluded.pos,
                        "context_hash": stmt.excluded.context_hash,
                        "translations": stmt.excluded.translations,
                    },
                )
                db.execute(stmt)
            except Exception:
                pass

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
        if user is not None:
            _srs_click(db, user, req.source_lang, lemma=lemma, pos=analysis.get("pos"), surface=req.surface, context_hash=context_hash, text_id=(rt.id if rt else None))
            db.commit()
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
    account = db.get(Account, user_id)
    if not account:
        raise HTTPException(401, "Account not found")
    return account
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


def require_tier(allowed: set[str]) -> Callable[[Account], Account]:
    def dep(account: Account = Depends(_get_current_account)) -> Account:
        if account.subscription_tier not in allowed:
            raise HTTPException(403, "Insufficient subscription tier")
        return account
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
def upsert_profile(req: ProfileRequest, db: Session = Depends(get_db), account: Account = Depends(_get_current_account)):
    _ensure_default_tiers(db)
    if not _is_supported_lang(req.lang):
        raise HTTPException(400, "Unsupported language code")
    prof = db.query(Profile).filter(Profile.account_id == account.id, Profile.lang == req.lang).first()
    if not prof:
        prof = Profile(account_id=account.id, lang=req.lang)
        db.add(prof)
        db.flush()
    if req.subscription_tier:
        # Validate against DB-backed tiers
        tier = db.query(SubscriptionTier).filter(SubscriptionTier.name == req.subscription_tier).first()
        if not tier:
            raise HTTPException(400, "Invalid subscription tier")
        account.subscription_tier = req.subscription_tier
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
    return {"ok": True, "account_id": account.id, "lang": prof.lang, "subscription_tier": account.subscription_tier, "level_value": prof.level_value, "level_var": prof.level_var, "level_code": prof.level_code}


class ProfileOut(BaseModel):
    lang: str
    created_at: datetime
    settings: Optional[Dict[str, Any]] = None
    level_value: float
    level_var: float
    level_code: Optional[str] = None
    preferred_script: Optional[str] = None


@app.get("/me/profiles", response_model=List[ProfileOut])
def list_profiles(db: Session = Depends(get_db), account: Account = Depends(_get_current_account)):
    profiles = db.query(Profile).filter(Profile.account_id == account.id).all()
    out: List[ProfileOut] = []
    for p in profiles:
        pref = db.query(ProfilePref).filter(ProfilePref.profile_id == p.id).first()
        out.append(ProfileOut(lang=p.lang, created_at=p.created_at, settings=(pref.data if pref else None), level_value=p.level_value, level_var=p.level_var, level_code=p.level_code, preferred_script=p.preferred_script))
    return out


@app.delete("/me/profile")
def delete_profile(lang: str, db: Session = Depends(get_db), user: User = Depends(_get_current_user)):
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
    _ensure_default_tiers(db)
    tiers = db.query(SubscriptionTier).order_by(SubscriptionTier.id.asc()).all()
    return [TierOut(name=t.name, description=t.description) for t in tiers]


class TierSetRequest(BaseModel):
    name: str


@app.get("/me/tier", response_model=TierOut)
def get_my_tier(db: Session = Depends(get_db), account: Account = Depends(_get_current_account)):
    _ensure_default_tiers(db)
    t = db.query(SubscriptionTier).filter(SubscriptionTier.name == account.subscription_tier).first()
    if not t:
        t = db.query(SubscriptionTier).first()
    return TierOut(name=t.name, description=t.description if t else None)


@app.post("/me/tier", response_model=TierOut)
def set_my_tier(req: TierSetRequest, db: Session = Depends(get_db), account: Account = Depends(_get_current_account)):
    _ensure_default_tiers(db)
    t = db.query(SubscriptionTier).filter(SubscriptionTier.name == req.name).first()
    if not t:
        raise HTTPException(400, "Invalid subscription tier")
    account.subscription_tier = t.name
    db.commit()
    return TierOut(name=t.name, description=t.description)


class MeOut(BaseModel):
    id: int
    email: str
    subscription_tier: str


@app.get("/me", response_model=MeOut)
def get_me(account: Account = Depends(_get_current_account)):
    return MeOut(id=account.id, email=account.email, subscription_tier=account.subscription_tier)


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


def _get_or_create_profile(db: Session, account: Account, lang: str) -> Profile:
    prof = db.query(Profile).filter(Profile.account_id == account.id, Profile.lang == lang).first()
    if prof:
        return prof
    prof = Profile(account_id=account.id, lang=lang)
    db.add(prof)
    db.flush()
    return prof


def _get_pref_row(db: Session, profile_id: int) -> ProfilePref:
    pref = db.query(ProfilePref).filter(ProfilePref.profile_id == profile_id).first()
    if not pref:
        pref = ProfilePref(profile_id=profile_id, data={})
        db.add(pref)
        db.flush()
    return pref


@app.get("/theme")
def get_theme(lang: str, db: Session = Depends(get_db), account: Account = Depends(_get_current_account)):
    # Read-only: do not create profile or prefs on GET
    prof = db.query(Profile).filter(Profile.account_id == account.id, Profile.lang == lang).first()
    if not prof:
        return {"name": None, "vars": {}}
    pref = db.query(ProfilePref).filter(ProfilePref.profile_id == prof.id).first()
    data = dict((pref.data or {}) if pref else {})
    theme = data.get("theme") or {}
    return {"name": theme.get("name"), "vars": theme.get("vars") or {}}


@app.post("/theme")
def set_theme(payload: ThemeIn, lang: str, db: Session = Depends(get_db), account: Account = Depends(_get_current_account)):
    prof = _get_or_create_profile(db, account, lang)
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
def get_ui_prefs(lang: str, db: Session = Depends(get_db), account: Account = Depends(_get_current_account)):
    # Read-only: do not create profile or prefs on GET
    prof = db.query(Profile).filter(Profile.account_id == account.id, Profile.lang == lang).first()
    if not prof:
        return {}
    pref = db.query(ProfilePref).filter(ProfilePref.profile_id == prof.id).first()
    data = dict((pref.data or {}) if pref else {})
    return data.get("ui_prefs") or {}


@app.post("/prefs")
def set_ui_prefs(payload: UIPrefsIn, lang: str, db: Session = Depends(get_db), account: Account = Depends(_get_current_account)):
    prof = _get_or_create_profile(db, account, lang)
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
def gen_reading(req: GenRequest, db: Session = Depends(get_db), account: Account = Depends(_get_current_account)):
    # Preferred script: pick from user's profile when available
    script = None
    if req.lang.startswith("zh"):
        prof = db.query(Profile).filter(Profile.account_id == account.id, Profile.lang == req.lang).first()
        if prof and getattr(prof, "preferred_script", None) in ("Hans", "Hant"):
            script = prof.preferred_script
        else:
            script = "Hans"
    # Comprehensible Input target (0..1)
    ci_target = _get_ci_target(db, account, req.lang)
    # Derive new word ratio from target (more familiar → fewer new words)
    base_new_ratio = max(0.02, min(0.6, 1.0 - ci_target + 0.05))
    # Auto-pick words if not provided
    words = req.include_words or pick_words(db, account, req.lang, count=12, new_ratio=base_new_ratio)
    level_hint = compose_level_hint(db, account, req.lang)
    # Per-language unit and default length
    unit = "chars" if req.lang.startswith("zh") else "words"
    approx_len = req.length if req.length is not None else (300 if unit == "chars" else 180)
    spec = PromptSpec(lang=req.lang, unit=unit, approx_len=approx_len, user_level_hint=level_hint, include_words=words, script=script, ci_target=ci_target)
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
        # Persist request/error for debugging (isolated session)
        _log_llm_request_safe({
            "account_id": account.id,
            "text_id": None,
            "kind": "reading",
            "provider": (req.provider or None),
            "model": (req.model or None),
            "base_url": (req.base_url or None),
            "status": "error",
            "request": {"messages": messages},
            "response": None,
            "error": str(e),
        })
        # Graceful error: return a sanitized message without stack traces
        import logging
        logging.getLogger("uvicorn.error").warning("LLM generation failed: %s", e)
        raise HTTPException(status_code=503, detail="No LLM backend available")
    # If empty, retry once with alternate provider, else return 503 without persisting
    if not text or text.strip() == "":
        alt_provider = "lmstudio" if (req.provider or "openrouter").lower() == "openrouter" else "openrouter"
        try:
            retry = chat_complete(
                messages,
                provider=alt_provider,
                model=req.model,
                base_url=req.base_url,
            )
        except Exception:
            retry = ""
        if not retry or retry.strip() == "":
            # Log empty output failure (isolated session)
            _log_llm_request_safe({
                "account_id": account.id,
                "text_id": None,
                "kind": "reading",
                "provider": (req.provider or None),
                "model": (req.model or None),
                "base_url": (req.base_url or None),
                "status": "error",
                "request": {"messages": messages, "note": "empty output then retry failed"},
                "response": None,
                "error": "empty output on both providers",
            })
            raise HTTPException(status_code=503, detail="No LLM backend available")
        text = retry
    # Best-effort CI evaluation and single retry if too unfamiliar
    try:
        share = _estimate_familiar_share(db, account, req.lang, text)
    except Exception:
        share = None  # ignore estimator errors
    if isinstance(share, float) and share < (ci_target - 0.03):
        # Retry once with stronger bias towards known words
        try:
            words2 = req.include_words or pick_words(db, account, req.lang, count=12, new_ratio=max(0.02, base_new_ratio * 0.5))
            spec2 = PromptSpec(lang=req.lang, unit=unit, approx_len=approx_len, user_level_hint=level_hint, include_words=words2, script=script, ci_target=ci_target)
            messages2 = build_reading_prompt(spec2)
            retry_text = chat_complete(messages2, provider=req.provider, model=req.model, base_url=req.base_url)
            if retry_text and retry_text.strip():
                text = retry_text
                messages = messages2
                words = words2
        except Exception:
            pass

    # Save text and generation log (unlimited tier check placeholder)
    source = "llm"
    rt = ReadingText(account_id=account.id, lang=req.lang, content=text, source=source)
    db.add(rt)
    db.flush()
    pid = db.query(Profile.id).filter(Profile.account_id == account.id, Profile.lang == req.lang).scalar()
    gl = GenerationLog(account_id=account.id, profile_id=pid, text_id=rt.id, model=req.model, base_url=req.base_url, prompt={"messages": messages, "provider": (req.provider or None)}, words={"include": words}, level_hint=level_hint, approx_len=approx_len, unit=unit)
    db.add(gl)
    try:
        _log_llm_request_safe({
            "account_id": account.id,
            "text_id": rt.id,
            "kind": "reading",
            "provider": (req.provider or None),
            "model": (req.model or None),
            "base_url": (req.base_url or None),
            "status": "ok",
            "request": {"messages": messages},
            "response": text,
            "error": None,
        })
    except Exception:
        pass
    db.commit()
    return {"prompt": messages, "text": text, "level_hint": level_hint, "words": words, "text_id": rt.id}


# -------- Urgent words (selection only) --------
@app.get("/srs/urgent")
def srs_urgent(lang: str, total: int = 12, new_ratio: float = 0.3, db: Session = Depends(get_db), account: Account = Depends(_get_current_account)):
    items = urgent_words_detailed(db, account, lang, total=total, new_ratio=new_ratio)
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
def upsert_lexeme_info(req: LexemeInfoUpsertRequest, db: Session = Depends(get_db), _admin: Account = Depends(require_tier({"admin"}))):
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


def _srs_exposure(db: Session, account: Account, lang: str, lemma: str, pos: Optional[str], surface: Optional[str], context_hash: Optional[str], text_id: Optional[int] = None):
    prof = db.query(Profile).filter(Profile.account_id == account.id, Profile.lang == lang).first()
    if not prof:
        prof = Profile(account_id=account.id, lang=lang)
        db.add(prof)
        db.flush()
    lex = _resolve_lexeme(db, lang, lemma, pos)
    ul = _get_or_create_userlexeme(db, account, prof, lex)
    now = datetime.utcnow()
    # determine exposure weight with gating
    # 1) session collapse: skip additional weight if last exposure was recent
    recent_ev = db.query(WordEvent).filter(
        WordEvent.account_id == account.id,
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
    db.add(WordEvent(ts=now, account_id=account.id, profile_id=prof.id, lexeme_id=lex.id, event_type="exposure", count=1, surface=surface, context_hash=context_hash, source="manual", meta={}, text_id=text_id))

    # Synthetic nonlookup promotion (conservative)
    if _SYN_NL_ENABLE:
        try:
            if (ul.clicks or 0) == 0 and int(ul.distinct_texts or 0) >= _SYN_NL_MIN_DISTINCT and ul.first_seen_at:
                days_seen = (now - ul.first_seen_at).total_seconds() / 86400.0
                if days_seen >= _SYN_NL_MIN_DAYS:
                    # cooldown: no recent nonlookup
                    recent_nl = db.query(WordEvent).filter(
                        WordEvent.account_id == account.id,
                        WordEvent.profile_id == prof.id,
                        WordEvent.lexeme_id == lex.id,
                        WordEvent.event_type == "nonlookup",
                        WordEvent.ts >= (now - timedelta(days=_SYN_NL_COOLDOWN_DAYS))
                    ).first()
                    if not recent_nl:
                        _srs_nonlookup(db, account, lang, lemma, pos, surface, context_hash, text_id)
        except Exception:
            pass


@app.post("/srs/event/exposures")
def srs_exposures(req: ExposuresRequest, db: Session = Depends(get_db), account: Account = Depends(_get_current_account)):
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
        _srs_exposure(db, account, req.lang, lemma, pos, it.surface, _hash_context(it.context), req.text_id)
        count += 1
    # Update bulk level estimate if stale (best-effort)
    try:
        update_level_if_stale(db, account.id, req.lang)
    except Exception:
        pass
    db.commit()
    return {"ok": True, "count": count}


class NonLookupRequest(BaseModel):
    lang: str
    items: List[ExposureItem]
    text_id: Optional[int] = None


def _srs_nonlookup(db: Session, account: Account, lang: str, lemma: str, pos: Optional[str], surface: Optional[str], context_hash: Optional[str], text_id: Optional[int] = None):
    # ensure profile and lexeme
    prof = db.query(Profile).filter(Profile.account_id == account.id, Profile.lang == lang).first()
    if not prof:
        prof = Profile(account_id=account.id, lang=lang)
        db.add(prof)
        db.flush()
    lex = _resolve_lexeme(db, lang, lemma, pos)
    ul = _get_or_create_userlexeme(db, account, prof, lex)
    # explicit non-click success: add strong nonlookup weight and schedule
    now = datetime.utcnow()
    _decay_posterior(ul, now, _HL_NONLOOK_D)
    ul.b_nonclick = (ul.b_nonclick or 0) + _SRS_BETA_NONLOOKUP
    ul.beta = float((ul.beta or 9.0) + _W_NONLOOK)
    ul.last_seen_at = now
    _schedule_next(ul, 3, now)
    db.add(WordEvent(ts=now, account_id=account.id, profile_id=prof.id, lexeme_id=lex.id, event_type="nonlookup", count=1, surface=surface, context_hash=context_hash, source="manual", meta={}, text_id=text_id))


@app.post("/srs/event/nonlookup")
def srs_nonlookup(req: NonLookupRequest, db: Session = Depends(get_db), account: Account = Depends(_get_current_account)):
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
        _srs_nonlookup(db, account, req.lang, lemma, pos, it.surface, _hash_context(it.context), req.text_id)
        count += 1
    try:
        update_level_if_stale(db, account.id, req.lang)
    except Exception:
        pass
    db.commit()
    return {"ok": True, "count": count}


class ClickRequest(BaseModel):
    lang: str
    lemma: Optional[str] = None
    pos: Optional[str] = None
    surface: Optional[str] = None
    context: Optional[str] = None
    text_id: Optional[int] = None


@app.post("/srs/event/click")
def srs_click(req: ClickRequest, db: Session = Depends(get_db), account: Account = Depends(_get_current_account)):
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
    _srs_click(db, account, req.lang, lemma, pos, req.surface, _hash_context(req.context), req.text_id)
    try:
        update_level_if_stale(db, account.id, req.lang)
        db.commit()
    except Exception:
        pass
    # Return derived metrics
    lex = _resolve_lexeme(db, req.lang, lemma, pos)
    prof = db.query(Profile).filter(Profile.account_id == account.id, Profile.lang == req.lang).first()
    ul = db.query(UserLexeme).filter(UserLexeme.account_id == account.id, UserLexeme.profile_id == (prof.id if prof else -1), UserLexeme.lexeme_id == lex.id).first()
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
    account: Account = Depends(_get_current_account),
):
    q = db.query(UserLexeme, Lexeme, LexemeInfo).join(Lexeme, UserLexeme.lexeme_id == Lexeme.id).outerjoin(LexemeInfo, LexemeInfo.lexeme_id == Lexeme.id).filter(
        UserLexeme.account_id == account.id,
        UserLexeme.profile_id == db.query(Profile.id).filter(Profile.account_id == account.id, Profile.lang == lang).scalar_subquery(),
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
def get_srs_stats(lang: str, db: Session = Depends(get_db), account: Account = Depends(_get_current_account)):
    pid = db.query(Profile.id).filter(Profile.account_id == account.id, Profile.lang == lang).scalar()
    if not pid:
        return SrsStatsOut(total=0, by_p={}, by_S={}, by_D={})
    rows = db.query(UserLexeme, Lexeme).join(Lexeme, UserLexeme.lexeme_id == Lexeme.id).filter(UserLexeme.account_id == account.id, UserLexeme.profile_id == pid).all()
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
def get_srs_level(lang: str, db: Session = Depends(get_db), account: Account = Depends(_get_current_account)):
    # best-effort freshness
    try:
        update_level_if_stale(db, account.id, lang)
        db.commit()
    except Exception:
        pass
    prof = db.query(Profile).filter(Profile.account_id == account.id, Profile.lang == lang).first()
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
    """Parse XML-like output from LLMs.
    Returns a pair (mode, items):
    - mode: 'single' when a single <translation> element is parsed; 'multi' when a
      <translations> container with segments is parsed.
    - items: list of strings (for 'single' it's a singleton list with the text).

    Robustness improvements:
    - Strip surrounding markdown code fences (``` or ```xml) and whitespace
      before XML parsing.
    - Accept root wrappers; descend to first <translation> or <translations>
      node if the root differs.
    - For multi, accept child tags in {'seg','s','item','line'}.
    """
    import re
    import xml.etree.ElementTree as ET

    def _strip_fences(s: str) -> str:
        t = s.strip()
        # Regex: optional language marker; capture inner block
        m = re.match(r"^```[ \t]*([a-zA-Z0-9_-]+)?\s*\n(?P<body>[\s\S]*?)\n?```\s*$", t)
        return (m.group("body") if m else t).strip()

    def _local(tag: Optional[str]) -> str:
        if not tag:
            return ""
        # Strip XML namespace if present: {ns}local
        return tag.split('}', 1)[-1].lower()

    cleaned = _strip_fences(xml_text)

    try:
        root = ET.fromstring(cleaned)

        wanted = {"translation", "translations"}

        # Find the operative node: either root or a descendant
        def _find_node(r: ET.Element) -> tuple[ET.Element, str] | tuple[None, None]:
            rname = _local(r.tag)
            if rname in wanted:
                return r, rname
            # If single child and that child matches, descend
            kids = [c for c in list(r) if isinstance(getattr(c, 'tag', None), str)]
            if len(kids) == 1:
                k = kids[0]
                kname = _local(k.tag)
                if kname in wanted:
                    return k, kname
            # Otherwise, find first descendant
            for el in r.iter():
                if el is r:
                    continue
                name = _local(getattr(el, 'tag', None))
                if name in wanted:
                    return el, name
            return None, None

        node, name = _find_node(root)
        if node is None or name is None:
            # No recognizable tags: fallback to single with trimmed original
            return "single", [cleaned.strip()]

        if name == "translation":
            return "single", [((node.text or "").strip())]

        # name == 'translations'
        out: List[str] = []
        accept = {"seg", "s", "item", "line"}
        for child in list(node):
            if _local(child.tag) in accept:
                out.append((child.text or "").strip())
        return "multi", out
    except Exception:
        # Not XML — treat the cleaned text as a single translation
        return "single", [cleaned.strip()]


def _assemble_prev_messages(db: Session, account: Account, text_id: Optional[int]) -> Optional[List[Dict[str, str]]]:
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
def translate(payload: TranslateIn, db: Session = Depends(get_db), account: Account = Depends(_get_current_account)):
    # Get target language from profile, fallback to payload or default
    profile = db.query(Profile).filter(Profile.account_id == account.id, Profile.lang == payload.lang).first()
    target_lang = payload.target_lang or (profile.target_lang if profile else "en")

    # Resolve source content
    raw: Optional[str] = payload.text
    base_offset = 0
    if raw is None and payload.text_id is not None:
        rt = db.get(ReadingText, payload.text_id)
        if not rt or rt.account_id != account.id:
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
        prev_msgs = _assemble_prev_messages(db, account, payload.text_id)

    spec = TranslationSpec(
        lang=payload.lang,
        target_lang=target_lang,
        unit=unit,
        content=(segments if len(segments) > 1 else segments[0]),
        continue_with_reading=bool(payload.continue_with_reading),
        script=None,
    )
    messages = build_translation_prompt(spec, prev_messages=prev_msgs)

    def _call_llm(msgs: List[Dict[str, str]]) -> str:
        try:
            return chat_complete(
                msgs,
                provider=payload.provider,
                model=payload.model,
                base_url=(payload.base_url or "http://localhost:1234/v1"),
                temperature=0.3,
            )
        except Exception as e:
            _log_llm_request_safe({
                "account_id": account.id,
                "text_id": (payload.text_id or None),
                "kind": "translation",
                "provider": (payload.provider or None),
                "model": (payload.model or None),
                "base_url": (payload.base_url or None),
                "status": "error",
                "request": {"messages": msgs},
                "response": None,
                "error": str(e),
            })
            raise HTTPException(status_code=503, detail="No LLM backend available")

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
                m = build_translation_prompt(TranslationSpec(lang=spec.lang, target_lang=target_lang, unit="sentence", content=seg), prev_messages=prev_msgs)
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

    # Log request/response for debugging (isolated session)
    try:
        _log_llm_request_safe({
            "account_id": account.id,
            "text_id": (payload.text_id or None),
            "kind": "translation",
            "provider": (payload.provider or None),
            "model": (payload.model or None),
            "base_url": (payload.base_url or None),
            "status": "ok",
            "request": {"messages": messages},
            "response": raw_response,
            "error": None,
        })
    except Exception:
        pass

    # Persist when translating a stored reading
    if payload.text_id is not None:
        from sqlalchemy.dialects.sqlite import insert
        from .models import ReadingTextTranslation as RTT
        for idx, (span, src, tr) in enumerate(zip(spans, segments, translations)):
            seg_idx = (idx if unit != "text" else None)
            s = base_offset + span[0]
            e = base_offset + span[1]
            stmt = insert(RTT).values(
                account_id=account.id,
                text_id=payload.text_id,
                unit=unit,
                target_lang=target_lang,
                segment_index=seg_idx,
                span_start=s,
                span_end=e,
                source_text=src,
                translated_text=tr,
                provider=(payload.provider or "openrouter"),
                model=payload.model,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=[RTT.account_id, RTT.text_id, RTT.target_lang, RTT.unit, RTT.segment_index, RTT.span_start, RTT.span_end],
                set_={
                    "translated_text": stmt.excluded.translated_text,
                    "source_text": stmt.excluded.source_text,
                    "provider": stmt.excluded.provider,
                    "model": stmt.excluded.model,
                },
            )
            db.execute(stmt)
        # minimal log
        try:
            db.add(TranslationLog(
                account_id=account.id,
                text_id=payload.text_id,
                unit=unit,
                target_lang=target_lang,
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
        "target_lang": target_lang,
        "items": items,
        "provider": (payload.provider or "openrouter"),
        "model": payload.model,
    }


@app.get("/reading/{text_id}/translations")
def get_translations(text_id: int, unit: Literal["sentence", "paragraph", "text"], target_lang: Optional[str] = None, db: Session = Depends(get_db), account: Account = Depends(_get_current_account)):
    rt = db.get(ReadingText, text_id)
    if not rt or rt.account_id != account.id:
        raise HTTPException(404, "reading text not found")

    # Get target language from profile if not provided
    if target_lang is None:
        profile = db.query(Profile).filter(Profile.account_id == account.id, Profile.lang == rt.lang).first()
        target_lang = profile.target_lang if profile else "en"

    rows = (
        db.query(ReadingTextTranslation)
        .filter(
            ReadingTextTranslation.text_id == text_id,
            ReadingTextTranslation.account_id == account.id,
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


class MarkReadIn(BaseModel):
    read: bool


@app.get("/reading/{text_id}/lookups")
def get_reading_lookups(text_id: int, target_lang: Optional[str] = None, db: Session = Depends(get_db), account: Account = Depends(_get_current_account)):
    rt = db.get(ReadingText, text_id)
    if not rt or rt.account_id != account.id:
        raise HTTPException(404, "reading text not found")

    # Get target language from profile if not provided
    if target_lang is None:
        profile = db.query(Profile).filter(Profile.account_id == account.id, Profile.lang == rt.lang).first()
        target_lang = profile.target_lang if profile else "en"

    rows = (
        db.query(ReadingLookup)
        .filter(
            ReadingLookup.account_id == account.id,
            ReadingLookup.text_id == text_id,
            ReadingLookup.target_lang == target_lang,
        )
        .order_by(ReadingLookup.span_start.asc())
        .all()
    )
    return [
        {
            "start": r.span_start,
            "end": r.span_end,
            "surface": r.surface,
            "lemma": r.lemma,
            "pos": r.pos,
            "translations": r.translations,
        }
        for r in rows
    ]


@app.post("/reading/{text_id}/mark_read")
def mark_reading(text_id: int, payload: MarkReadIn, db: Session = Depends(get_db), account: Account = Depends(_get_current_account)):
    rt = db.get(ReadingText, text_id)
    if not rt or rt.account_id != account.id:
        raise HTTPException(404, "reading text not found")
    now = datetime.utcnow()
    if payload.read:
        rt.is_read = True
        rt.read_at = now
    else:
        rt.is_read = False
        rt.read_at = None
    db.commit()
    return {"ok": True, "is_read": rt.is_read, "read_at": (rt.read_at.isoformat() if rt.read_at else None)}


@app.get("/reading/{text_id}/meta")
def get_reading_meta(text_id: int, db: Session = Depends(get_db), account: Account = Depends(_get_current_account)):
    rt = db.get(ReadingText, text_id)
    if not rt or rt.account_id != account.id:
        raise HTTPException(404, "reading text not found")
    return {
        "text_id": rt.id,
        "lang": rt.lang,
        "is_read": bool(getattr(rt, "is_read", False)),
        "read_at": (rt.read_at.isoformat() if getattr(rt, "read_at", None) else None),
        "created_at": (rt.created_at.isoformat() if getattr(rt, "created_at", None) else None),
    }


# ---- UI shell mounting (header/footer/templates) ----
try:
    # Optional UI libraries from libs/
    from arcadia_ui_style import ensure_templates  # type: ignore
    from arcadia_ui_core import router as ui_router, attach_ui, mount_ui_static, ContextMenuRegistry, MenuItem  # type: ignore
    tdir = ensure_templates(Path(__file__).resolve().parent)
    templates = Jinja2Templates(directory=tdir)

    # Create context menus for word interactions
    context_menu_registry = ContextMenuRegistry()

    # Context menu for word items
    def word_context_menu(req: ContextMenuRequest) -> List[Dict[str, Any]]:
        lang = req.dataset.get('lang', 'zh')
        lemma = req.dataset.get('lemma', '')
        pos = req.dataset.get('pos', '')

        return [
            MenuItem(
                label="Mark as known",
                hx={
                    "post": f"/srs/event/nonlookup",
                    "vals": f"{{'lang': '{lang}', 'items': [{{'lemma': '{lemma}', 'pos': '{pos}'}}]}}"
                }
            ),
            MenuItem(
                label="Look up word",
                href=f"/api/lookup?lang={lang}&lemma={lemma}"
            ),
            MenuItem(divider=True),
            MenuItem(
                label="Add to wordlist",
                hx={
                    "post": f"/api/wordlists/add",
                    "vals": f"{{'lang': '{lang}', 'lemma': '{lemma}'}}"
                }
            ),
        ]

    context_menu_registry.add("word", word_context_menu)

    # Navigation items for header
    nav_items = [
        {"label": "Words", "href": "/words", "active": True},
        {"label": "Stats", "href": "/stats"},
        {"label": "Settings", "href": "/settings"},
    ]

    # Attach UI state and mount static assets with enhanced features
    attach_ui(
        app,
        templates,
        persist_header=True,
        brand_home_url="/",
        brand_name="Arcadia Lang",
        brand_tag="",
        nav_items=nav_items,
        context_menus=context_menu_registry,
    )
    mount_ui_static(app)
    _templates_env = templates  # share env for project pages
    app.include_router(ui_router)
except Exception:
    # non-fatal during dev
    pass

# Words browser page (server-rendered HTML using shared header/footer)
@app.get("/words", response_class=HTMLResponse)
def words_page(request: Request, lang: Optional[str] = None, account: Account = Depends(_get_current_account)):
    # Use render_page utility for consistent UI layout
    from arcadia_ui_core import render_page
    t = _templates()
    return render_page(
        request,
        templates,
        content_template="words.html",
        title="My Words",
        context={"lang": lang or "es"}
    )

# Login/Signup pages are provided by arcadia_ui_core router

# Placeholder pages for account menu
@app.get("/profile", response_class=HTMLResponse)
def profile_page(request: Request, account: Account = Depends(_get_current_account)):
    from arcadia_ui_core import render_page
    t = _templates()
    return render_page(
        request,
        templates,
        content_template="profile.html",
        title="Profile"
    )


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, account: Account = Depends(_get_current_account)):
    from arcadia_ui_core import render_page
    t = _templates()
    return render_page(
        request,
        templates,
        content_template="settings.html",
        title="Settings"
    )


@app.get("/stats", response_class=HTMLResponse)
def stats_page(request: Request, account: Account = Depends(_get_current_account)):
    from arcadia_ui_core import render_page
    t = _templates()
    return render_page(
        request,
        templates,
        content_template="stats.html",
        title="Statistics"
    )


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
