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
from .db import get_db, init_db, DB_PATH
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
from .services.srs_service import (
    srs_click as _svc_srs_click,
    srs_exposure as _svc_srs_exposure,
    srs_nonlookup as _svc_srs_nonlookup,
)
from .services.llm_service import generate_reading as _svc_generate_reading
from .services.translation_service import (
    sentence_spans as _svc_sentence_spans,
    paragraph_spans as _svc_paragraph_spans,
    assemble_prev_messages as _svc_assemble_prev_messages,
    translate_text as _svc_translate_text,
)
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from langs.tokenize import TOKENIZERS, Token
from .llm import (
    PromptSpec,
    build_reading_prompt,
    chat_complete,
    build_translation_prompt,
    TranslationSpec,
    pick_words,
    compose_level_hint,
    urgent_words_detailed,
)
# No direct SRS imports; endpoints delegate to services
# from .repos.profiles import get_or_create_profile as _get_or_create_profile  # not needed after services split
try:
    from logic.mstream.saver import save_word_gloss  # type: ignore
except Exception:
    def save_word_gloss(*args, **kwargs):  # type: ignore
        return None
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

# Simple per-minute rate limit by IP or user (no DB lookups)
# Tier is assumed 'free' until we embed it in JWTs.
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
_OPENCC_T2S = None  # optional opencc converter placeholder


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


try:
    from logic.logs import log_llm_request as _log_llm_request_safe  # type: ignore
except Exception:
    def _log_llm_request_safe(*args, **kwargs):  # type: ignore
        return None


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


# SRS scheduling lives in services.srs_service


from .services.lexeme_service import resolve_lexeme as _resolve_lexeme, get_or_create_userlexeme as _get_or_create_userlexeme


def _srs_click(db: Session, account: Account, lang: str, lemma: str, pos: Optional[str], surface: Optional[str], context_hash: Optional[str], text_id: Optional[int] = None):
    _svc_srs_click(
        db,
        account_id=account.id,
        lang=lang,
        lemma=lemma,
        pos=pos,
        surface=surface,
        context_hash=context_hash,
        text_id=text_id,
    )


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
                # No global lookup; just use uid and local profile
                class _U: pass
                _tmp = _U(); _tmp.id = uid  # minimal identity
                user = _tmp
                profile = db.query(Profile).filter(Profile.account_id == uid, Profile.lang == req.source_lang).first()
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
    try:
        return _svc_generate_reading(
            db,
            account_id=account.id,
            lang=req.lang,
            length=req.length,
            include_words=req.include_words,
            model=req.model,
            provider=req.provider,
            base_url=req.base_url,
        )
    except Exception:
        raise HTTPException(status_code=503, detail="No LLM backend available")


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
    _svc_srs_exposure(
        db,
        account_id=account.id,
        lang=lang,
        lemma=lemma,
        pos=pos,
        surface=surface,
        context_hash=context_hash,
        text_id=text_id,
    )


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
    _svc_srs_nonlookup(
        db,
        account_id=account.id,
        lang=lang,
        lemma=lemma,
        pos=pos,
        surface=surface,
        context_hash=context_hash,
        text_id=text_id,
    )


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
    return _svc_sentence_spans(text, lang)


def _paragraph_spans(text: str) -> List[Tuple[int, int]]:
    return _svc_paragraph_spans(text)


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
    return _svc_assemble_prev_messages(db, account.id, text_id)


@app.post("/translate")
def translate(payload: TranslateIn, db: Session = Depends(get_db), account: Account = Depends(_get_current_account)):
    profile = db.query(Profile).filter(Profile.account_id == account.id, Profile.lang == payload.lang).first()
    target_lang = payload.target_lang or (profile.target_lang if profile else "en")
    try:
        return _svc_translate_text(
            db,
            account_id=account.id,
            lang=payload.lang,
            target_lang=target_lang,
            unit=payload.unit,
            text=payload.text,
            text_id=payload.text_id,
            start=payload.start,
            end=payload.end,
            continue_with_reading=bool(payload.continue_with_reading),
            provider=payload.provider,
            model=payload.model,
            base_url=payload.base_url,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(503, str(e))


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


@app.get("/reading/{text_id}/words")
def get_reading_words(text_id: int, db: Session = Depends(get_db), account: Account = Depends(_get_current_account)):
    """Get cached word glosses for a reading text (optimization for faster lookups)."""
    rt = db.get(ReadingText, text_id)
    if not rt or rt.account_id != account.id:
        raise HTTPException(404, "reading text not found")
    
    rows = (
        db.query(ReadingWordGloss)
        .filter(
            ReadingWordGloss.account_id == account.id,
            ReadingWordGloss.text_id == text_id,
        )
        .order_by(ReadingWordGloss.span_start, ReadingWordGloss.span_end)
        .all()
    )
    
    return {
        "text_id": text_id,
        "words": [
            {
                "surface": w.surface,
                "lemma": w.lemma,
                "pos": w.pos,
                "pinyin": w.pinyin,
                "translation": w.translation,
                "lemma_translation": w.lemma_translation,
                "grammar": w.grammar,
                "span_start": w.span_start,
                "span_end": w.span_end,
            }
            for w in rows
        ],
    }


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

# Login/Signup pages
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    from arcadia_ui_core import render_page
    t = _templates()
    return render_page(
        request,
        t,
        content_template="login.html",
        title="Log in",
    )


@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request):
    from arcadia_ui_core import render_page
    t = _templates()
    return render_page(
        request,
        t,
        content_template="signup.html",
        title="Sign up",
    )

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
