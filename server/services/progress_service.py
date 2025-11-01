from __future__ import annotations

from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session

from ..models import Profile, ReadingText


class ProgressService:
    def record_session(self, db: Session, account_id: int, payload) -> None:
        # Placeholder for telemetry ingestion
        return None

    def complete_and_mark_read(self, db: Session, account_id: int, prior_text_id: Optional[int]) -> None:
        if not prior_text_id:
            return
        try:
            rt = db.get(ReadingText, int(prior_text_id))
            if rt and rt.account_id == int(account_id):
                rt.is_read = True
                rt.read_at = datetime.utcnow()
                db.commit()
        except Exception:
            try:
                db.rollback()
            except Exception:
                pass
