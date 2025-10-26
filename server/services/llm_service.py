from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime

from sqlalchemy.orm import Session

from ..models import (
    Profile,
    ReadingText,
    GenerationLog,
)
from ..services.level_service import get_ci_target
from ..llm import PromptSpec, build_reading_prompt, chat_complete
from ..llm import pick_words as _pick_words, compose_level_hint as _compose_level_hint
from ..config import MSP_ENABLE
try:
    from logic.mstream.saver import save_word_gloss  # type: ignore
except Exception:
    def save_word_gloss(*args, **kwargs):  # type: ignore
        return None
try:
    from logic.logs import log_llm_request as _log_llm_request_safe  # type: ignore
except Exception:
    def _log_llm_request_safe(*args, **kwargs):  # type: ignore
        return None


def generate_reading(
    db: Session,
    *,
    account_id: int,
    lang: str,
    length: Optional[int],
    include_words: Optional[List[str]],
    model: Optional[str],
    provider: Optional[str],
    base_url: str,
) -> Dict[str, Any]:
    # Preferred script
    script = None
    if lang.startswith("zh"):
        prof = db.query(Profile).filter(Profile.account_id == account_id, Profile.lang == lang).first()
        if prof and getattr(prof, "preferred_script", None) in ("Hans", "Hant"):
            script = prof.preferred_script
        else:
            script = "Hans"

    # CI target and level hint
    ci_target = get_ci_target(db, account_id, lang)
    base_new_ratio = max(0.02, min(0.6, 1.0 - ci_target + 0.05))

    # Wrap account id as a minimal identity for llm helpers
    class _U: pass
    user = _U(); user.id = account_id

    words = include_words or _pick_words(db, user, lang, count=12, new_ratio=base_new_ratio)
    level_hint = _compose_level_hint(db, user, lang)

    unit = "chars" if lang.startswith("zh") else "words"
    prof = db.query(Profile).filter(Profile.account_id == account_id, Profile.lang == lang).first()
    prof_len = None
    try:
        if prof and isinstance(prof.text_length, int) and prof.text_length and prof.text_length > 0:
            prof_len = int(prof.text_length)
    except Exception:
        prof_len = None
    approx_len = length if length is not None else (prof_len if prof_len is not None else (300 if unit == "chars" else 180))
    try:
        approx_len = max(50, min(2000, int(approx_len)))
    except Exception:
        approx_len = 300 if unit == "chars" else 180

    spec = PromptSpec(
        lang=lang,
        unit=unit,
        approx_len=approx_len,
        user_level_hint=level_hint,
        include_words=words,
        script=script,
        ci_target=ci_target,
    )
    messages = build_reading_prompt(spec)

    try:
        text = chat_complete(messages, provider=provider, model=model, base_url=base_url)
    except Exception as e:
        _log_llm_request_safe({
            "account_id": account_id,
            "text_id": None,
            "kind": "reading",
            "provider": (provider or None),
            "model": (model or None),
            "base_url": (base_url or None),
            "status": "error",
            "request": {"messages": messages},
            "response": None,
            "error": str(e),
        })
        raise

    if not text or text.strip() == "":
        raise RuntimeError("empty llm output")

    # Persist reading + generation log
    rt = ReadingText(account_id=account_id, lang=lang, content=text)
    db.add(rt)
    db.flush()
    gl = GenerationLog(
        account_id=account_id,
        profile_id=(prof.id if prof else None),
        text_id=rt.id,
        model=model,
        base_url=base_url,
        prompt={"messages": messages},
        words={"include": words},
        level_hint=level_hint or None,
        approx_len=approx_len,
        unit=unit,
        created_at=datetime.utcnow(),
    )
    db.add(gl)
    try:
        _log_llm_request_safe({
            "account_id": account_id,
            "text_id": rt.id,
            "kind": "reading",
            "provider": (provider or None),
            "model": (model or None),
            "base_url": (base_url or None),
            "status": "ok",
            "request": {"messages": messages},
            "response": text,
            "error": None,
        })
    except Exception:
        pass

    # Save word gloss placeholders for local caches (best-effort)
    try:
        if words and rt.id:
            for i, w in enumerate(words):
                try:
                    save_word_gloss(
                        db,
                        account_id=account_id,
                        text_id=rt.id,
                        lang=lang,
                        surface=w,
                        span_start=-1 * (i + 1),
                        span_end=-1 * (i + 1),
                        lemma=w,
                        pos=None,
                        pinyin=None,
                        translation=None,
                        lemma_translation=None,
                        grammar=dict(),
                    )
                except Exception:
                    continue
    except Exception:
        pass

    db.commit()
    return {"prompt": messages, "text": text, "level_hint": level_hint, "words": words, "text_id": rt.id}
