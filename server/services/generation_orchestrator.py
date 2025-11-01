from __future__ import annotations

from sqlalchemy.orm import Session

from .gen_queue import ensure_text_available as _queue_ensure


class GenerationOrchestrator:
    def ensure_text_available(self, db: Session, account_id: int, lang: str, prefetch: bool = False) -> None:
        try:
            print(f"[GEN] ensure_text_available called account_id={int(account_id)} lang={str(lang)} prefetch={bool(prefetch)}")
        except Exception:
            pass
        return _queue_ensure(db, int(account_id), str(lang), prefetch=prefetch)
