from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List, Callable, Literal, Tuple
import os
import time
from collections import deque, defaultdict
import math

from fastapi import FastAPI, HTTPException, Depends, Request, Query
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from fastapi.responses import RedirectResponse

# Language engine
from langs.parsing import ENGINES, format_morph_label
from langs.dicts import DictionaryProviderChain, StarDictProvider, CedictProvider
from .db import get_db, init_db, SessionLocal, DB_PATH
from .deps import get_current_account as _get_current_account, require_tier
from .api.profile import router as profile_router
from .api.wordlists import router as wordlists_router
from .api.tiers import router as tiers_router
from .config import MSP_ENABLE
if MSP_ENABLE:
    try:
        from .api.mstream import router as msp_router
    except Exception:
        msp_router = None  # type: ignore
from arcadia_auth import Account
from .models import Profile, SubscriptionTier, ProfilePref, Lexeme, UserLexeme, WordEvent, UserLexemeContext, LexemeInfo, LexemeVariant, ReadingText, GenerationLog, ReadingTextTranslation, TranslationLog, ReadingLookup
from .level import update_level_if_stale, update_level_for_profile
from .level import get_level_summary
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from langs.tokenize import TOKENIZERS, Token
from .llm import PromptSpec, build_reading_prompt, chat_complete, build_translation_prompt, TranslationSpec
from logic.words import pick_words, compose_level_hint, urgent_words_detailed
from logic.srs import (
    decay_posterior as _decay_posterior,
    retention_now as _retention_now,
    SRSParams,
    schedule_next,
)
from profiles.service import get_or_create as _get_or_create_profile
from arcadia_auth import decode_token, create_auth_router, AuthSettings, mount_cookie_agent_middleware  # type: ignore
from fastapi.templating import Jinja2Templates

# SRS parameters
_SRS_ALPHA_CLICK = 0.2
_SRS_BETA_EXPOSURE = 0.02
_DIVERSITY_K = 8.0
_SRS_BETA_NONLOOKUP = 2
_SRS_GAMMA_NONLOOKUP = 0.08

# SRS config (env-overridable)
# Centralized SRS/config values
from .config import (
    _W_CLICK,
    _W_NONLOOK,
    _W_EXPOSURE,
    _HL_CLICK_D,
    _HL_NONLOOK_D,
    _HL_EXPOSURE_D,
    _FSRS_TARGET_R,
    _FSRS_FAIL_F,
    _G_PASS_WEAK,
    _G_PASS_NORM,
    _G_PASS_STRONG,
)

# Exposure gating
from .config import (
    _SESSION_MIN,
    _EXPOSURE_WEAK_W,
    _DISTINCT_PROMOTE,
    _FREQ_LOW_THRESH,
    _DIFF_HIGH,
)

# Synthetic nonlookup promotion
from .config import (
    _SYN_NL_ENABLE,
    _SYN_NL_MIN_DISTINCT,
    _SYN_NL_MIN_DAYS,
    _SYN_NL_COOLDOWN_DAYS,
)


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
    from arcadia_auth import create_sqlite_repo  # type: ignore
    # IMPORTANT: use the same SQLite file as the app DB so Account lookups work
    _auth_settings = AuthSettings(secret_key=_JWT_SECRET)
    app.include_router(create_auth_router(create_sqlite_repo(f"sqlite:///{DB_PATH}"), _auth_settings))
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
app.include_router(tiers_router)
if MSP_ENABLE and msp_router is not None:
    app.include_router(msp_router, prefix="/msp")

# Simple per-minute rate limit by IP or user tier for heavy endpoints
# TODO(deploy): tighten rate limits per tier; for local/dev, set effectively unlimited.
from .config import RATE_LIMITS as _RATE_LIMITS, RATE_WINDOW_SEC as _RATE_WINDOW_SEC
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
    window = float(_RATE_WINDOW_SEC)
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


from logic.logs import log_llm_request as _log_llm_request_safe


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


## moved to logic.srs to separate math from web concerns


_SRS_PARAMS = SRSParams(
    target_retention=_FSRS_TARGET_R,
    fail_factor=_FSRS_FAIL_F,
    g_pass_weak=_G_PASS_WEAK,
    g_pass_norm=_G_PASS_NORM,
    g_pass_strong=_G_PASS_STRONG,
)


def _schedule_next(ul: UserLexeme, quality: int, now: datetime) -> None:
    schedule_next(ul, quality, now, _SRS_PARAMS)


from logic.lexemes import resolve_lexeme as _resolve_lexeme, get_or_create_userlexeme as _get_or_create_userlexeme


def _srs_click(db: Session, account: Account, lang: str, lemma: str, pos: Optional[str], surface: Optional[str], context_hash: Optional[str], text_id: Optional[int] = None):
    # Pick or create profile for the lang
    prof = _get_or_create_profile(db, account.id, lang)
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
                    user = account
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
            if rt and rt.account_id == user.id:
                s = max(0, int(req.start))
                e = int(req.end)
                if e > s:
                    if rt.content:
                        e = min(e, len(rt.content))
                    # Try cache read
                    row = (
                        db.query(ReadingLookup)
                        .filter(
                            ReadingLookup.account_id == user.id,
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
                    account_id=user.id,
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
                    index_elements=[RL.account_id, RL.text_id, RL.target_lang, RL.span_start, RL.span_end],
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
## Authentication helpers live under server.deps
## Tier setup and endpoints live under server/api/tiers.py


# Authorization helpers are provided by server.deps.require_tier
 


## /me endpoints are implemented in server/api/profile.py


# ---- UI Theme and Prefs (minimal) ----
## Theme and UI prefs endpoints live under server/api/profile.py


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
    # Prefer user profile length when available
    prof = db.query(Profile).filter(Profile.account_id == account.id, Profile.lang == req.lang).first()
    prof_len = None
    try:
        if prof and isinstance(prof.text_length, int) and prof.text_length and prof.text_length > 0:
            prof_len = int(prof.text_length)
    except Exception:
        prof_len = None
    approx_len = req.length if req.length is not None else (prof_len if prof_len is not None else (300 if unit == "chars" else 180))
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
    prof = _get_or_create_profile(db, account.id, lang)
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
    prof = _get_or_create_profile(db, account.id, lang)
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


def _strip_fences_json(s: str) -> str:
    import re
    t = s.strip()
    m = re.match(r"^```[ \t]*([a-zA-Z0-9_-]+)?\s*\n(?P<body>[\s\S]*?)\n?```\s*$", t)
    return (m.group("body") if m else t).strip()


def _parse_json_translations(text: str) -> List[str]:
    import json
    cleaned = _strip_fences_json(text)
    try:
        data = json.loads(cleaned)
    except Exception:
        # Some models return a bare JSON array; try to extract if present inside text
        try:
            # naive fallback: find first '[' ... ']' and parse
            start = cleaned.find('[')
            end = cleaned.rfind(']')
            if start != -1 and end != -1 and end > start:
                data = json.loads(cleaned[start:end+1])
            else:
                return [cleaned.strip()]
        except Exception:
            return [cleaned.strip()]
    if isinstance(data, list):
        return [str(x).strip() for x in data]
    if isinstance(data, dict):
        vals = data.get("translations")
        if isinstance(vals, list):
            return [str(x).strip() for x in vals]
        # Fallback keys
        if isinstance(data.get("translation"), str):
            return [data.get("translation").strip()]
    # Last resort: stringify
    return [str(data).strip()]


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
    items_parsed = _parse_json_translations(out)
    if isinstance(spec.content, list) and len(items_parsed) != len(segments):
        # retry per-segment with JSON mode
        items_parsed = []
        for seg in segments:
            m = build_translation_prompt(TranslationSpec(lang=spec.lang, target_lang=target_lang, unit="sentence", content=seg), prev_messages=prev_msgs)
            r = _call_llm(m)
            raw_response += "\n" + r
            ones = _parse_json_translations(r)
            items_parsed.append(ones[0] if ones else "")
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
