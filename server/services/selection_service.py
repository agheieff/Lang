from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_

from ..models import Profile, ReadingText, ProfileTextRead


class SelectionService:
    def pick_current_or_new(
        self,
        account_db: Session,  # Per-account DB for Profile, ProfileTextRead
        global_db: Session,   # Global DB for ReadingText
        account_id: int,
        lang: str,
    ) -> Optional[ReadingText]:
        """Get the current text or pick a new one from global pool.
        
        Logic:
        1. If profile.current_text_id exists, return that text
        2. Otherwise pick next ready text from global pool (matching lang/target_lang)
        3. Skip texts the user has already read (unless reread is allowed)
        """
        prof = account_db.query(Profile).filter(
            Profile.account_id == account_id,
            Profile.lang == lang
        ).first()
        if not prof:
            return None
        
        target_lang = prof.target_lang
        
        # Case 1: User has a current_text_id - return that text
        if getattr(prof, "current_text_id", None):
            rt = global_db.get(ReadingText, int(prof.current_text_id))
            if rt and rt.lang == lang and rt.target_lang == target_lang:
                return rt
        
        # Get IDs of texts this profile has read
        read_text_ids = set(
            r.text_id for r in account_db.query(ProfileTextRead.text_id)
            .filter(ProfileTextRead.profile_id == prof.id)
            .all()
        )
        
        # Case 2: Find a ready text from global pool
        # Ready = has content + words_complete + sentences_complete
        query = (
            global_db.query(ReadingText)
            .filter(
                ReadingText.lang == lang,
                ReadingText.target_lang == target_lang,
                ReadingText.content.isnot(None),
                ReadingText.content != "",
                ReadingText.words_complete == True,
                ReadingText.sentences_complete == True,
            )
        )
        
        # Filter out already-read texts (unless reread is allowed)
        if read_text_ids:
            query = query.filter(~ReadingText.id.in_(read_text_ids))
        
        rt = query.order_by(ReadingText.created_at.asc()).first()
        
        if rt:
            try:
                # Set current_text_id on profile
                prof.current_text_id = rt.id
                account_db.commit()
                print(f"[SELECTION] Set current_text_id={rt.id} for profile_id={prof.id}")
                return rt
            except Exception as e:
                print(f"[SELECTION] Failed to set current_text_id: {e}")
                account_db.rollback()
        
        # Case 3: No ready texts available - return None (will trigger generation)
        return None
    
    def get_next_text(
        self,
        account_db: Session,
        global_db: Session,
        profile: Profile,
    ) -> Optional[ReadingText]:
        """Get the next unread text for a profile.
        
        Used when user clicks "Next" and we need to find another text.
        """
        read_text_ids = set(
            r.text_id for r in account_db.query(ProfileTextRead.text_id)
            .filter(ProfileTextRead.profile_id == profile.id)
            .all()
        )
        
        query = (
            global_db.query(ReadingText)
            .filter(
                ReadingText.lang == profile.lang,
                ReadingText.target_lang == profile.target_lang,
                ReadingText.content.isnot(None),
                ReadingText.content != "",
                ReadingText.words_complete == True,
                ReadingText.sentences_complete == True,
            )
        )
        
        if read_text_ids:
            query = query.filter(~ReadingText.id.in_(read_text_ids))
        
        return query.order_by(ReadingText.created_at.asc()).first()
    
    def mark_text_read(
        self,
        account_db: Session,
        profile: Profile,
        text_id: int,
    ) -> None:
        """Mark a text as read by the profile."""
        now = datetime.now(timezone.utc)
        
        existing = account_db.query(ProfileTextRead).filter(
            ProfileTextRead.profile_id == profile.id,
            ProfileTextRead.text_id == text_id
        ).first()
        
        if existing:
            existing.read_count += 1
            existing.last_read_at = now
        else:
            account_db.add(ProfileTextRead(
                profile_id=profile.id,
                text_id=text_id,
                read_count=1,
                first_read_at=now,
                last_read_at=now,
            ))
        
        account_db.flush()
