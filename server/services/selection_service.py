from __future__ import annotations

from typing import Optional
from sqlalchemy.orm import Session

from ..models import Profile, ReadingText


class SelectionService:
    def pick_current_or_new(self, db: Session, account_id: int, lang: str) -> Optional[ReadingText]:
        prof = db.query(Profile).filter(Profile.account_id == account_id, Profile.lang == lang).first()
        if not prof:
            return None
        if getattr(prof, "current_text_id", None):
            rt = db.get(ReadingText, int(prof.current_text_id))
            if rt and rt.account_id == account_id:
                return rt
        rt = (
            db.query(ReadingText)
            .filter(ReadingText.account_id == account_id, ReadingText.lang == lang, ReadingText.opened_at.is_(None))
            .order_by(ReadingText.created_at.desc())
            .first()
        )
        if rt:
            try:
                prof.current_text_id = rt.id
                db.commit()
            except Exception:
                db.rollback()
            return rt
        return None
