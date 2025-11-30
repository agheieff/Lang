"""
Session processing service.
Handles processing of user interaction data from local storage when user moves to next text.
Separated from generation pipeline to maintain clear separation of concerns.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any

from sqlalchemy.orm import Session

from ..models import (
    ReadingLookup,
    ReadingWordGloss,
    Profile,
    Lexeme,
    # UserLexeme removed
)
from ..services.srs_service import SrsService
from ..schemas.session import TextSessionState


class SessionProcessingService:
    """
    Processes user session data from local storage.
    
    This service handles:
    - Processing word click/lookup events
    - Updating SRS (spaced repetition system) data
    - Recording user progress
    - Updating user proficiency levels
    
    It does NOT handle:
    - Text generation (handled by TextGenerationService)
    - Translation generation (handled by TranslationService)
    - Notifications (handled by NotificationService)
    """
    
    def __init__(self):
        self.srs_service = SrsService()
    
    def process_session_data(self, 
                            db: Session, 
                            account_id: int, 
                            text_id: int,
                            session_data: Dict[str, Any]) -> bool:
        """
        Process session data from a text reading session.
        
        Args:
            db: Database session
            account_id: User account ID
            text_id: Text ID that was just completed
            session_data: Session data from local storage containing:
                - lookups: Word lookup events
                - interactions: User interactions with words
                - time_spent: Time spent on the text
                - other metrics
            
        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            print(f"[SESSION_PROCESSING] Processing session data for account_id={account_id} text_id={text_id}")
            
            # Process word lookups
            lookups_processed = self._process_word_lookups(db, account_id, text_id, session_data.get("lookups", []))
            
            # Update SRS data
            srs_updated = self._update_srs_data(db, account_id, session_data)
            
            # Record user progress
            progress_updated = self._record_progress(db, account_id, text_id, session_data)
            
            # Update user level if needed
            self._update_user_level(db, account_id)
            
            # Adapt CI preference based on lookup rate
            self._adapt_ci_preference(db, account_id, text_id, session_data)
            
            print(f"[SESSION_PROCESSING] Session processing completed: lookups={lookups_processed} srs={srs_updated} progress={progress_updated}")
            return True
            
        except Exception as e:
            print(f"[SESSION_PROCESSING] Error processing session data: {e}")
            try:
                db.rollback()
            except Exception:
                pass
            return False
    
    def _process_word_lookups(self, 
                            db: Session, 
                            account_id: int, 
                            text_id: int, 
                            lookups: List[Dict]) -> int:
        """Process word lookup events from session data."""
        # Skip if no valid text_id
        if not text_id or text_id <= 0:
            return 0
        
        # Get text to determine lang
        from ..models import ReadingText
        text = db.query(ReadingText).filter(
            ReadingText.id == text_id,
            ReadingText.account_id == account_id
        ).first()
        if not text:
            return 0
        
        lang = text.lang
        
        # Get target_lang from profile
        profile = db.query(Profile).filter(
            Profile.account_id == account_id,
            Profile.lang == lang
        ).first()
        target_lang = profile.target_lang if profile else "en"
        
        processed_count = 0
        
        for lookup in lookups:
            try:
                # Extract lookup data
                word_surface = lookup.get("word")
                span_start = lookup.get("span_start")
                span_end = lookup.get("span_end")
                pos = lookup.get("pos")
                lemma = lookup.get("lemma")
                translations = lookup.get("translations", [])
                timestamp = lookup.get("timestamp")
                
                if not word_surface or span_start is None or span_end is None:
                    continue
                
                # Create lookup record
                lookup_record = ReadingLookup(
                    account_id=account_id,
                    text_id=text_id,
                    lang=lang,
                    target_lang=target_lang,
                    span_start=span_start,
                    span_end=span_end,
                    surface=word_surface,
                    lemma=lemma,
                    pos=pos,
                    translations=translations,
                    created_at=datetime.fromisoformat(timestamp) if timestamp else datetime.utcnow()
                )
                
                db.add(lookup_record)
                processed_count += 1
                
            except Exception as e:
                print(f"[SESSION_PROCESSING] Error processing lookup: {e}")
                continue
        
        return processed_count
    
    def _update_srs_data(self, db: Session, account_id: int, session_data: Dict) -> bool:
        """Update SRS data based on session interactions."""
        from ..models import Profile, Lexeme
        from .srs_service import srs_click, srs_exposure, _ensure_profile, _resolve_lexeme
        
        try:
            interactions = session_data.get("interactions", [])
            lang = session_data.get("lang", "zh")
            
            # Pre-fetch profile once
            profile = _ensure_profile(db, account_id, lang)
            
            # Optimization: Bulk resolve lexemes for this batch if possible
            # For now, we'll use a local cache to avoid resolving the same lexeme multiple times in one session
            lexeme_cache = {} 
            
            # Track session start for collapse logic
            # Assuming the first interaction timestamp or now roughly
            session_start_time = datetime.utcnow()
            if interactions:
                 first_ts = interactions[0].get("timestamp")
                 if first_ts:
                     try:
                         session_start_time = datetime.fromisoformat(first_ts)
                     except Exception:
                         pass

            for interaction in interactions:
                word_surface = interaction.get("word")
                event_type = interaction.get("event_type")  # "lookup", "click", "exposure"
                timestamp = interaction.get("timestamp")
                
                # These fields are needed for SRS
                lemma = interaction.get("lemma") or word_surface # Fallback
                pos = interaction.get("pos")
                context_hash = interaction.get("context_hash") # If available
                
                if not word_surface or not event_type:
                    continue
                
                # Resolve lexeme (cached per account/profile)
                cache_key = (account_id, profile.id, lang, lemma, pos)
                if cache_key not in lexeme_cache:
                    lexeme_cache[cache_key] = _resolve_lexeme(
                        db, lang, lemma, pos,
                        account_id=account_id, profile_id=profile.id
                    )
                lexeme = lexeme_cache[cache_key]
                
                if event_type in ("lookup", "click"):
                    srs_click(
                        db, 
                        account_id=account_id, 
                        lang=lang,
                        lemma=lemma,
                        pos=pos,
                        surface=word_surface,
                        context_hash=context_hash,
                        profile=profile,
                        lexeme=lexeme
                    )
                elif event_type == "exposure":
                    srs_exposure(
                        db,
                        account_id=account_id,
                        lang=lang,
                        lemma=lemma,
                        pos=pos,
                        surface=word_surface,
                        context_hash=context_hash,
                        profile=profile,
                        lexeme=lexeme,
                        session_start_time=session_start_time
                    )
            
            # Explicit flush at end of batch
            db.flush()
            return True
            
        except Exception as e:
            print(f"[SESSION_PROCESSING] Error updating SRS data: {e}")
            return False
    
    def _record_progress(self, 
                        db: Session, 
                        account_id: int, 
                        text_id: int, 
                        session_data: Dict) -> bool:
        """Record user progress metrics."""
        try:
            # Update text state to mark as read
            from ..models import ReadingText
            text = db.query(ReadingText).filter(
                ReadingText.id == text_id,
                ReadingText.account_id == account_id
            ).first()
            
            if text:
                text.is_read = True
                text.read_at = datetime.utcnow()
                
                # Store additional metrics if available
                if "time_spent" in session_data:
                    # Could add a time_spent column to ReadingText in future
                    pass
                
                if "completion_percentage" in session_data:
                    # Could track how much of text was actually read
                    pass
            
            # Update profile statistics
            profile = db.query(Profile).filter(Profile.account_id == account_id).first()
            if profile:
                # Increment texts read counter (could add to profile model)
                pass
            
            return True
            
        except Exception as e:
            print(f"[SESSION_PROCESSING] Error recording progress: {e}")
            return False
    
    def _update_user_level(self, db: Session, account_id: int) -> None:
        """Update user's estimated proficiency level."""
        try:
            from ..level import update_level_if_stale
            # Get all profiles for this account and update each
            profiles = db.query(Profile).filter(Profile.account_id == account_id).all()
            for prof in profiles:
                update_level_if_stale(db, account_id, prof.lang)
            
        except Exception as e:
            print(f"[SESSION_PROCESSING] Error updating user level: {e}")
    
    def generate_summary_report(self, 
                               db: Session, 
                               account_id: int, 
                               text_id: int) -> Dict:
        """Generate a summary of the session processing for this text."""
        try:
            from ..models import ReadingText, ReadingLookup
            
            # Get text info
            text = db.query(ReadingText).filter(
                ReadingText.id == text_id,
                ReadingText.account_id == account_id
            ).first()
            
            if not text:
                return {"error": "Text not found"}
            
            # Count lookups
            lookup_count = db.query(ReadingLookup).filter(
                ReadingLookup.account_id == account_id,
                ReadingLookup.text_id == text_id
            ).count()
            
            # Get unique words
            unique_words = db.query(ReadingLookup.surface).filter(
                ReadingLookup.account_id == account_id,
                ReadingLookup.text_id == text_id
            ).distinct().count()
            
            return {
                "text_id": text_id,
                "text_length": len(text.content) if text.content else 0,
                "lookup_count": lookup_count,
                "unique_words_count": unique_words,
                "read_at": text.read_at.isoformat() if text.read_at else None,
                "opened_at": text.opened_at.isoformat() if text.opened_at else None,
            }
            
        except Exception as e:
            print(f"[SESSION_PROCESSING] Error generating summary: {e}")
            return {"error": str(e)}
    
    def _adapt_ci_preference(
        self,
        db: Session,
        account_id: int,
        text_id: int,
        session_data: Dict[str, Any],
    ) -> None:
        """
        Adapt CI preference based on session performance.
        
        Rules:
        - High lookup rate (>15%) → increase CI (make easier)
        - Low lookup rate (<5%) → decrease CI (make harder)
        - Also adapt topic weights based on engagement
        """
        try:
            from ..models import ReadingText, ReadingLookup
            
            # Get text and profile
            text = db.query(ReadingText).filter(
                ReadingText.id == text_id,
                ReadingText.account_id == account_id
            ).first()
            
            if not text or not text.content:
                return
            
            lang = text.lang
            profile = db.query(Profile).filter(
                Profile.account_id == account_id,
                Profile.lang == lang
            ).first()
            
            if not profile:
                return
            
            # Calculate lookup rate
            text_length = len(text.content)
            lookup_count = db.query(ReadingLookup).filter(
                ReadingLookup.account_id == account_id,
                ReadingLookup.text_id == text_id
            ).count()
            
            # Approximate word count (for non-Chinese, use space-based; for Chinese, char count / 1.5)
            if lang.startswith("zh"):
                approx_words = text_length / 1.5
            else:
                approx_words = len(text.content.split())
            
            lookup_rate = lookup_count / max(approx_words, 1)
            
            # Get current CI preference
            ci_pref = getattr(profile, 'ci_preference', None) or 0.92
            
            # Adapt CI based on lookup rate
            CI_ADJUST = 0.02
            if lookup_rate > 0.15:
                # Too many lookups - make easier
                new_ci = min(0.98, ci_pref + CI_ADJUST)
                print(f"[ADAPT] High lookup rate ({lookup_rate:.2%}), increasing CI: {ci_pref:.2f} -> {new_ci:.2f}")
            elif lookup_rate < 0.05:
                # Few lookups - can make harder
                new_ci = max(0.80, ci_pref - CI_ADJUST)
                print(f"[ADAPT] Low lookup rate ({lookup_rate:.2%}), decreasing CI: {ci_pref:.2f} -> {new_ci:.2f}")
            else:
                new_ci = ci_pref
            
            if new_ci != ci_pref:
                profile.ci_preference = new_ci
            
            # Adapt topic weights based on engagement (handles multiple comma-separated topics)
            text_topic_str = getattr(text, 'topic', None)
            if text_topic_str:
                text_topics = [t.strip() for t in text_topic_str.split(',') if t.strip()]
                topic_weights = getattr(profile, 'topic_weights', None) or {}
                read_time_ms = session_data.get("read_time_ms") or 0
                if not isinstance(read_time_ms, (int, float)):
                    read_time_ms = 0
                
                # Simple heuristic: if user spent good amount of time, boost topics
                # If they rushed through (< 30 seconds), slightly reduce
                for topic in text_topics:
                    if read_time_ms > 60000:  # > 1 minute
                        old_weight = topic_weights.get(topic, 1.0)
                        topic_weights[topic] = min(2.0, old_weight * 1.05)
                        print(f"[ADAPT] Good engagement with '{topic}', boosting weight")
                    elif read_time_ms > 0 and read_time_ms < 30000:  # < 30 seconds
                        old_weight = topic_weights.get(topic, 1.0)
                        topic_weights[topic] = max(0.2, old_weight * 0.95)
                        print(f"[ADAPT] Quick read of '{topic}', reducing weight slightly")
                
                profile.topic_weights = topic_weights
            
            db.commit()
            
        except Exception as e:
            print(f"[ADAPT] Error adapting CI preference: {e}")
            try:
                db.rollback()
            except Exception:
                pass

    def persist_session_state(self, db: Session, account_id: int, state: TextSessionState) -> bool:
        """Persist periodic session sync state (e.g., to ProfilePref or temp table)."""
        from ..models import ProfilePref
        
        # Find profile
        # We don't have lang in schema yet, assume it's derived or stored with text_id context
        # Ideally schema should include lang, but we can lookup text
        from ..models import ReadingText
        rt = db.get(ReadingText, state.text_id)
        if not rt:
            return False
            
        # Texts are now global - just verify profile exists for this lang
        prof = db.query(Profile).filter(Profile.account_id == account_id, Profile.lang == rt.lang).first()
        if not prof:
            return False
            
        # Store in ProfilePref for now (simple JSON storage per profile)
        # We use a specific key for current text session
        pref = db.query(ProfilePref).filter(ProfilePref.profile_id == prof.id).first()
        if not pref:
            pref = ProfilePref(profile_id=prof.id, data={})
            db.add(pref)
            
        # Update the session state for this text
        # We store it under "current_session" key
        current_data = dict(pref.data or {})
        current_data["current_session"] = state.model_dump(mode='json')
        pref.data = current_data
        
        try:
            # Mark as modified for SQLA to pick up JSON change
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(pref, "data")
            db.commit()
            return True
        except Exception as e:
            print(f"[SESSION_PROCESSING] Error persisting session state: {e}")
            db.rollback()
            return False

    def get_persisted_session_state(self, db: Session, account_id: int, text_id: int) -> Optional[Dict]:
        """Retrieve persisted session state for restoration."""
        from ..models import ReadingText, ProfilePref
        rt = db.get(ReadingText, text_id)
        if not rt:
            return None
            
        # Texts are now global - just verify profile exists for this lang
        prof = db.query(Profile).filter(Profile.account_id == account_id, Profile.lang == rt.lang).first()
        if not prof:
            return None
            
        pref = db.query(ProfilePref).filter(ProfilePref.profile_id == prof.id).first()
        if not pref or not pref.data:
            return None
            
        session_data = pref.data.get("current_session")
        if not session_data or session_data.get("text_id") != text_id:
            return None
            
        return session_data
