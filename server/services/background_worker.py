"""
Background worker for pool management and maintenance tasks.
Runs independently of HTTP requests to keep the system healthy.
"""
import logging
import threading
import time
from typing import List, Tuple

from ..db import GlobalSessionLocal
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
            global_db = GlobalSessionLocal()
            try:
                # Retry incomplete texts first (uses global DB for reading_texts)
                retried = self.orchestrator.retry_incomplete_texts(global_db, account_id, lang)
                if retried > 0:
                    logger.info(f"[WORKER] Queued {retried} text(s) for retry (account={account_id}, lang={lang})")
                
                # Then ensure pool is filled (uses global DB for reading_texts)
                self.orchestrator.ensure_text_available(global_db, account_id, lang)
            except Exception as e:
                logger.error(f"[WORKER] Error processing account={account_id} lang={lang}: {e}")
            finally:
                global_db.close()
    
    def _get_active_profiles(self) -> List[Tuple[int, str]]:
        """Get all account_id, lang pairs that have profiles."""
        from ..models import Profile
        
        results = []
        
        try:
            # Profiles are in the global DB, not per-account DB
            global_db = GlobalSessionLocal()
            try:
                profiles = global_db.query(Profile.account_id, Profile.lang).all()
                for account_id, lang in profiles:
                    results.append((account_id, lang))
            finally:
                global_db.close()
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
