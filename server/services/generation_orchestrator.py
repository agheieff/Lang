from __future__ import annotations

from sqlalchemy.orm import Session

from .gen_queue import ensure_text_available as _queue_ensure, retry_failed_components
from .retry_service import RetryService
from .readiness_service import ReadinessService
import threading


class GenerationOrchestrator:
    def __init__(self):
        self.retry_service = RetryService()
        self.readiness_service = ReadinessService()
    
    def ensure_text_available(self, db: Session, account_id: int, lang: str, prefetch: bool = False) -> None:
        return _queue_ensure(db, int(account_id), str(lang), prefetch=prefetch)
    
    def check_and_retry_failed_texts(self, db: Session, account_id: int, lang: str) -> list:
        """Check for failed texts and attempt retries if allowed."""
        try:
            print(f"[RETRY] Checking for failed texts to retry for account_id={account_id} lang={lang}")
        except Exception:
            pass
        
        failed_texts = self.retry_service.get_failed_texts_for_retry(db, account_id, lang, limit=3)
        
        # Trigger retries in background threads to avoid blocking
        for text in failed_texts:
            def retry_in_background():
                try:
                    result = retry_failed_components(account_id, text.id)
                    if result.get("error"):
                        try:
                            print(f"[RETRY] Retry failed for text_id={text.id}: {result['error']}")
                        except Exception:
                            pass
                    else:
                        try:
                            print(f"[RETRY] Successfully initiated retry for text_id={text.id}")
                        except Exception:
                            pass
                            
                except Exception as e:
                    try:
                        print(f"[RETRY] Error retrying text_id={text.id}: {e}")
                    except Exception:
                        pass
            
            thread = threading.Thread(target=retry_in_background, daemon=True, name=f"retry-{account_id}-{text.id}")
            thread.start()
        
        return [{"text_id": t.id, "status": "retry_initiated"} for t in failed_texts]
