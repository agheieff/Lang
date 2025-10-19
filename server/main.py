from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

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
from .models import User, Profile
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


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


 # Shared dictionary provider chain (lazy-init on first call)
_DICT_CHAIN: Optional[DictionaryProviderChain] = None
_PH = PasswordHasher()
_JWT_SECRET = "dev-secret-change"  # replace in production
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
    token_type: str = "bearer"


def _create_token(user_id: int) -> str:
    payload = {"sub": str(user_id), "exp": datetime.utcnow() + timedelta(days=7)}
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


@app.post("/auth/register", response_model=TokenResponse)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    init_db()
    existing = db.query(User).filter(User.email == req.email).first()
    if existing:
        raise HTTPException(400, "Email already registered")
    pwd = _PH.hash(req.password)
    user = User(email=req.email, password_hash=pwd)
    db.add(user)
    db.commit()
    db.refresh(user)
    return TokenResponse(access_token=_create_token(user.id))


@app.post("/auth/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    init_db()
    user = db.query(User).filter(User.email == req.email).first()
    if not user:
        raise HTTPException(401, "Invalid credentials")
    try:
        _PH.verify(user.password_hash, req.password)
    except Exception:
        raise HTTPException(401, "Invalid credentials")
    return TokenResponse(access_token=_create_token(user.id))


class ProfileRequest(BaseModel):
    lang: str
    subscription_tier: Optional[str] = None  # free|pro|enterprise


@app.post("/me/profile")
def upsert_profile(req: ProfileRequest, db: Session = Depends(get_db), user: User = Depends(_get_current_user)):
    init_db()
    prof = db.query(Profile).filter(Profile.user_id == user.id, Profile.lang == req.lang).first()
    if not prof:
        prof = Profile(user_id=user.id, lang=req.lang)
        db.add(prof)
    if req.subscription_tier:
        user.subscription_tier = req.subscription_tier
    db.commit()
    return {"ok": True, "user_id": user.id, "lang": prof.lang, "subscription_tier": user.subscription_tier}


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
