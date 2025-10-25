from __future__ import annotations

from sqlalchemy.orm import Session

from ..models import SubscriptionTier


def ensure_default_tiers(db: Session) -> None:
    existing = {t.name for t in db.query(SubscriptionTier).all()}
    defaults = [
        ("free", "Free plan"),
        ("premium", "Premium plan"),
        ("pro", "Pro plan"),
        ("admin", "Administrator (no limits)"),
    ]
    created = False
    for name, desc in defaults:
        if name not in existing:
            db.add(SubscriptionTier(name=name, description=desc))
            created = True
    if created:
        db.commit()
