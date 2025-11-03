from __future__ import annotations

from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session
import logging
import json
import os
from pathlib import Path

from ..models import Profile, ReadingText, ReadingLookup

logger = logging.getLogger(__name__)


def _session_log_dir_root() -> Path:
    """Base directory for session logs, mirroring LLM stream logs structure"""
    base = os.getenv("ARC_OR_LOG_DIR", str(Path.cwd() / "data" / "session_logs"))
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _session_log_path(account_id: int, lang: str) -> Path:
    """Generate timestamped directory for a session log"""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    d = _session_log_dir_root() / str(int(account_id)) / lang / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


class ProgressService:
    def record_session(self, db: Session, account_id: int, payload) -> None:
        """Process complete session data including word lookups, translations, and analytics."""
        if not payload or not isinstance(payload, dict):
            return None
            
        try:
            logger.info(f"Recording session for account {account_id}: {payload.get('session_id')}")
            
            # Extract session metadata
            session_id = payload.get('session_id')
            text_id = payload.get('text_id')
            lang = payload.get('lang')
            target_lang = payload.get('target_lang')
            opened_at = payload.get('opened_at')
            analytics = payload.get('analytics', {})
            
            if not text_id or not session_id:
                logger.warning(f"Missing required session fields: {list(payload.keys())}")
                return None
                
            # Verify text belongs to account
            text = db.get(ReadingText, text_id)
            if not text or text.account_id != account_id:
                logger.warning(f"Text {text_id} not found for account {account_id}")
                return None
                
            # Process word lookups
            self._process_word_lookups(db, account_id, text_id, payload)
            
            # Store session analytics and save session data to files
            self._store_analytics(db, account_id, text_id, session_id, analytics)
            self._save_session_to_disk(account_id, lang, session_id, payload)
            
            logger.info(f"Successfully processed session {session_id} for account {account_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error recording session for account {account_id}: {e}", exc_info=True)
            return None
    
    def _process_word_lookups(self, db: Session, account_id: int, text_id: int, payload: dict) -> None:
        """Process word lookup data from session."""
        lookups = []
        
        # Extract word lookups from title
        title = payload.get('title', {})
        if title and title.get('words'):
            for word in title['words']:
                if word.get('looked_up_at'):
                    lookups.append({
                        'word': word,
                        'is_title': True
                    })
        
        # Extract word lookups from paragraphs
        paragraphs = payload.get('paragraphs', [])
        for para in paragraphs:
            sentences = para.get('sentences', [])
            for sentence in sentences:
                words = sentence.get('words', [])
                for word in words:
                    if word.get('looked_up_at'):
                        lookups.append({
                            'word': word,
                            'is_title': False,
                            'sentence_text': sentence.get('text', ''),
                            'paragraph_text': para.get('text', '')
                        })
        
        # Create ReadingLookup records
        for lookup in lookups:
            word_data = lookup['word']
            try:
                record = ReadingLookup(
                    account_id=account_id,
                    text_id=text_id,
                    surface=word_data.get('surface'),
                    lemma=word_data.get('lemma'),
                    pos=word_data.get('pos'),
                    translations=[word_data.get('translation')] if word_data.get('translation') else [],
                    target_lang=payload.get('target_lang', 'en'),
                    span_start=None,  # Not available in new structure
                    span_end=None,    # Not available in new structure
                    created_at=datetime.fromtimestamp(word_data.get('looked_up_at', 0) / 1000.0)
                )
                db.add(record)
            except Exception as e:
                logger.debug(f"Error creating lookup record: {e}")
                continue
                
        try:
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving lookup records: {e}")
    
    def _store_analytics(self, db: Session, account_id: int, text_id: int, session_id: str, analytics: dict) -> None:
        """Store session analytics data."""
        # For now, just log analytics - could be expanded to store in database
        logger.info(f"Session Analytics - Text {text_id}, Account {account_id}:")
        logger.info(f"  Total words: {analytics.get('total_words', 0)}")
        logger.info(f"  Words looked up: {analytics.get('words_looked_up', 0)}")
        logger.info(f"  Lookup rate: {analytics.get('lookup_rate', 0):.3f}")
        logger.info(f"  Reading time (ms): {analytics.get('reading_time_ms', 0)}")
        logger.info(f"  Reading speed (WPM): {analytics.get('average_reading_speed_wpm', 0)}")
        logger.info(f"  Completion status: {analytics.get('completion_status', 'unknown')}")
        
        # Analytics could be stored in a dedicated table for reporting
        # For now, just logging for debugging purposes
    
    def _save_session_to_disk(self, account_id: int, lang: str, session_id: str, payload: dict) -> None:
        """Save complete session data to disk similar to LLM stream logs"""
        try:
            session_dir = _session_log_path(account_id, lang)
            
            # Save meta information
            meta = {
                "account_id": account_id,
                "lang": lang,
                "session_id": session_id,
                "text_id": payload.get('text_id'),
                "created_at": datetime.utcnow().isoformat()
            }
            meta_file = session_dir / "meta.json"
            meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            
            # Save complete session data
            session_file = session_dir / "session.json"
            session_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            
            # Save analytics separately for easy access
            analytics_file = session_dir / "analytics.json"
            analytics = payload.get('analytics', {})
            if analytics:
                analytics_file.write_text(json.dumps(analytics, ensure_ascii=False, indent=2), encoding="utf-8")
            
            # Save just the word lookups for analysis
            lookups_file = session_dir / "word_lookups.json"
            lookups = self._extract_all_lookups(payload)
            if lookups:
                lookups_file.write_text(json.dumps(lookups, ensure_ascii=False, indent=2), encoding="utf-8")
            
            # Save timestamps for all interactions
            interactions_file = session_dir / "interactions.json"
            interactions = self._extract_all_interactions(payload)
            if interactions:
                interactions_file.write_text(json.dumps(interactions, ensure_ascii=False, indent=2), encoding="utf-8")
            
            logger.debug(f"Saved session data to {session_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save session data to disk: {e}", exc_info=True)
    
    def _extract_all_lookups(self, payload: dict) -> list:
        """Extract all word lookups from session payload"""
        lookups = []
        
        # Title word lookups
        title = payload.get('title', {})
        if title and title.get('words'):
            for word in title['words']:
                if word.get('looked_up_at'):
                    lookups.append({
                        'location': 'title',
                        'surface': word.get('surface'),
                        'lemma': word.get('lemma'),
                        'pos': word.get('pos'),
                        'translation': word.get('translation'),
                        'looked_up_at': word.get('looked_up_at')
                    })
        
        # Body word lookups
        paragraphs = payload.get('paragraphs', [])
        for para_idx, para in enumerate(paragraphs):
            sentences = para.get('sentences', [])
            for sent_idx, sentence in enumerate(sentences):
                words = sentence.get('words', [])
                for word in words:
                    if word.get('looked_up_at'):
                        lookups.append({
                            'location': 'body',
                            'paragraph_index': para_idx,
                            'sentence_index': sent_idx,
                            'surface': word.get('surface'),
                            'lemma': word.get('lemma'),
                            'pos': word.get('pos'),
                            'translation': word.get('translation'),
                            'looked_up_at': word.get('looked_up_at')
                        })
        
        # Sort by timestamp
        return sorted(lookups, key=lambda x: x.get('looked_up_at', 0))
    
    def _extract_all_interactions(self, payload: dict) -> list:
        """Extract all interaction events with timestamps"""
        interactions = []
        
        # Session start
        if payload.get('opened_at'):
            interactions.append({
                'event_type': 'session_start',
                'timestamp': payload.get('opened_at'),
                'text_id': payload.get('text_id')
            })
        
        # Title translation
        title = payload.get('title', {})
        if title and title.get('translated_at'):
            interactions.append({
                'event_type': 'title_translation',
                'timestamp': title.get('translated_at'),
                'translation': title.get('full_translation')
            })
        
        # Sentence translations
        paragraphs = payload.get('paragraphs', [])
        for para_idx, para in enumerate(paragraphs):
            if para.get('translated_at'):
                interactions.append({
                    'event_type': 'paragraph_translation',
                    'timestamp': para.get('translated_at'),
                    'paragraph_index': para_idx
                })
            
            sentences = para.get('sentences', [])
            for sent_idx, sentence in enumerate(sentences):
                if sentence.get('translated_at'):
                    interactions.append({
                        'event_type': 'sentence_translation',
                        'timestamp': sentence.get('translated_at'),
                        'paragraph_index': para_idx,
                        'sentence_index': sent_idx,
                        'original_text': sentence.get('text'),
                        'translation': sentence.get('translation')
                    })
        
        # Word lookups
        lookups = self._extract_all_lookups(payload)
        for lookup in lookups:
            interactions.append({
                'event_type': 'word_lookup',
                'timestamp': lookup.get('looked_up_at'),
                'location': lookup.get('location'),
                'surface': lookup.get('surface')
            })
        
        # Full text translation if available
        if payload.get('full_translation'):
            # Add as final interaction (approximate time)
            opened_at = payload.get('opened_at', 0)
            analytics = payload.get('analytics', {})
            reading_time = analytics.get('reading_time_ms', 0)
            interactions.append({
                'event_type': 'full_text_translation',
                'timestamp': opened_at + reading_time,
                'full_translation': payload.get('full_translation')
            })
        
        # Sort by timestamp
        return sorted(interactions, key=lambda x: x.get('timestamp', 0))

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
