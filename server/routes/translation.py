from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from server.auth import Account  # type: ignore

from ..account_db import get_db
from ..deps import get_current_account as _get_current_account
from ..models import Profile
from ..services.translation_service import (
    assemble_prev_messages as _svc_assemble_prev_messages,
    paragraph_spans as _svc_paragraph_spans,
    sentence_spans as _svc_sentence_spans,
    translate_text as _svc_translate_text,
)


router = APIRouter(tags=["translation"])


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


def _assemble_prev_messages(db: Session, account: Account, text_id: Optional[int]):
    return _svc_assemble_prev_messages(db, account.id, text_id)


@router.post("/translate")
def translate(
    payload: TranslateIn,
    db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    profile = (
        db.query(Profile)
        .filter(Profile.account_id == account.id, Profile.lang == payload.lang)
        .first()
    )
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

