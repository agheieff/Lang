from __future__ import annotations

from sqlalchemy.orm import Session

from ..models import SubscriptionTier


def ensure_default_tiers(db: Session) -> None:
    rows = db.query(SubscriptionTier).all()
    existing = {t.name for t in rows}
    legacy_map = {"free": ("Free", "Free plan"), "premium": ("Standard", "Standard plan"), "pro": ("Pro", "Pro plan")}
    renamed = False
    for old, (new, new_desc) in legacy_map.items():
        if old in existing and new not in existing:
            for r in rows:
                if r.name == old:
                    r.name = new
                    if not r.description:
                        r.description = new_desc
                    renamed = True
                    break
    if renamed:
        db.commit()
        rows = db.query(SubscriptionTier).all()
        existing = {t.name for t in rows}
    defaults = [
        ("Free", "Free plan"),
        ("Standard", "Standard plan"),
        ("Pro", "Pro plan"),
        ("Pro+", "Pro Plus plan"),
        ("BYOK", "Bring Your Own Key"),
        ("admin", "Administrator (no limits)"),
    ]
    created = False
    for name, desc in defaults:
        if name not in existing:
            db.add(SubscriptionTier(name=name, description=desc))
            created = True
    if created:
        db.commit()
