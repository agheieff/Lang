"""
Background worker for pool management and maintenance tasks.
Runs independently of HTTP requests to keep the system healthy.
"""
import logging
import threading
import time
from typing import List, Tuple

from ..db import GlobalSessionLocal
from ..utils.session_manager import db_manager
from .generation_orchestrator import GenerationOrchestrator

logger = logging.getLogger(__name__)


class BackgroundWorker:
    """
    Background thread that handles:
    - Pool generation and backfill
    - Translation retries with exponential backoff
    - Cleanup of permanently failed texts
    """
    
    def __init__(self, interval_seconds: int = 30):
        self.interval = interval_seconds
        self.running = False
        self.thread = None
        self.orchestrator = GenerationOrchestrator()
    
    def start(self):
        """Start the background worker thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(
            target=self._run,
            name="background-worker",
            daemon=True
        )
        self.thread.start()
        logger.info(f"[WORKER] Background worker started (interval={self.interval}s)")
    
    def stop(self):
        """Stop the background worker thread."""
        self.running = False
        logger.info("[WORKER] Background worker stopping...")
    
    def _run(self):
        """Main worker loop."""
        while self.running:
            try:
                self._do_maintenance()
            except Exception as e:
                logger.error(f"[WORKER] Maintenance cycle error: {e}", exc_info=True)
            
            # Sleep in small increments to allow faster shutdown
            for _ in range(self.interval):
                if not self.running:
                    break
                time.sleep(1)
        
        logger.info("[WORKER] Background worker stopped")
    
    def _do_maintenance(self):
        """Run one maintenance cycle for all active profiles."""
        profiles = self._get_active_profiles()
        
        if not profiles:
            return
        
        logger.debug(f"[WORKER] Running maintenance for {len(profiles)} profile(s)")
        
        for account_id, lang in profiles:
            try:
                with db_manager.transaction(account_id) as db:
                    # Retry incomplete texts first
                    retried = self.orchestrator.retry_incomplete_texts(db, account_id, lang)
                    if retried > 0:
                        logger.info(f"[WORKER] Queued {retried} text(s) for retry (account={account_id}, lang={lang})")
                    
                    # Then ensure pool is filled
                    self.orchestrator.ensure_text_available(db, account_id, lang)
            except Exception as e:
                logger.error(f"[WORKER] Error processing account={account_id} lang={lang}: {e}")
    
    def _get_active_profiles(self) -> List[Tuple[int, str]]:
        """Get all account_id, lang pairs that have profiles."""
        from ..auth.models import Account
        from ..models import Profile
        
        results = []
        
        try:
            # Get all accounts from global DB
            global_db = GlobalSessionLocal()
            try:
                accounts = global_db.query(Account.id).all()
                account_ids = [a.id for a in accounts]
            finally:
                global_db.close()
            
            # For each account, get their profiles
            for account_id in account_ids:
                try:
                    with db_manager.read_only(account_id) as db:
                        profiles = db.query(Profile.lang).filter(
                            Profile.account_id == account_id
                        ).all()
                        for (lang,) in profiles:
                            results.append((account_id, lang))
                except Exception as e:
                    logger.debug(f"[WORKER] Could not read profiles for account {account_id}: {e}")
        except Exception as e:
            logger.error(f"[WORKER] Error getting active profiles: {e}")
        
        return results


# Singleton instance
_worker_instance = None


def get_background_worker(interval_seconds: int = 30) -> BackgroundWorker:
    """Get or create the singleton background worker."""
    global _worker_instance
    if _worker_instance is None:
        _worker_instance = BackgroundWorker(interval_seconds)
    return _worker_instance
