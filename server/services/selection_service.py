from __future__ import annotations

from typing import Optional
from sqlalchemy.orm import Session

from ..models import Profile, ReadingText
from .pool_selection_service import get_pool_selection_service


class SelectionService:
    def pick_current_or_new(self, db: Session, account_id: int, lang: str) -> Optional[ReadingText]:
        """Get the current text or pick a new one.
        
        Logic:
        1. If profile.current_text_id exists, return that text (show what user was last reading)
        2. If no current_text_id, use pool selection to pick best match
        3. Fallback to any unopened text if pool is empty
        """
        from datetime import datetime
        
        prof = db.query(Profile).filter(Profile.account_id == account_id, Profile.lang == lang).first()
        if not prof:
            return None
        
        # Case 1: User has a current_text_id - return that text (even if already opened/read)
        if getattr(prof, "current_text_id", None):
            rt = db.get(ReadingText, int(prof.current_text_id))
            if rt and rt.account_id == account_id:
                return rt
        
        # Case 2: Try pool-based selection first
        pool_service = get_pool_selection_service()
        rt = pool_service.select_from_pool(db, account_id, lang, prof)
        
        # Case 3: Fallback to any unopened text if pool selection failed
        if not rt:
            rt = (
                db.query(ReadingText)
                .filter(
                    ReadingText.account_id == account_id, 
                    ReadingText.lang == lang, 
                    ReadingText.opened_at.is_(None),
                    ReadingText.content.is_not(None),
                    ReadingText.content != ""
                )
                .order_by(ReadingText.created_at.asc())
                .first()
            )
        
        if rt:
            try:
                # Mark as no longer pooled
                rt.pooled = False
                # Set current_text_id
                prof.current_text_id = rt.id
                # Mark as opened immediately
                rt.opened_at = datetime.utcnow()
                db.commit()
                print(f"[SELECTION] Set current_text_id={rt.id} (ci={getattr(rt, 'ci_target', None)}, topic={getattr(rt, 'topic', None)}) for account_id={account_id}")
                return rt
            except Exception as e:
                print(f"[SELECTION] Failed to set current_text_id and mark opened: {e}")
                db.rollback()
        
        # Case 4: No ready texts available - return None (will trigger generation)
        return None
