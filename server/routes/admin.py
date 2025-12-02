"""Admin routes for monitoring and management."""
from __future__ import annotations

from typing import Optional, List
from collections import defaultdict

from fastapi import APIRouter, Depends, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import func

from server.auth import Account
from ..deps import get_current_account
from ..db import get_global_db
from ..models import ReadingText, Profile

router = APIRouter(prefix="/admin", tags=["admin"])


def _get_templates():
    from pathlib import Path
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    
    templates_dir = Path(__file__).resolve().parents[1] / "templates"
    try:
        env = Environment(
            loader=FileSystemLoader([str(templates_dir)]),
            autoescape=select_autoescape(["html", "xml"]),
        )
        return Jinja2Templates(env=env)
    except Exception:
        return Jinja2Templates(directory=str(templates_dir))


def _require_admin(account: Account) -> None:
    """Raise 403 if account is not admin or system."""
    if account.subscription_tier not in ("admin", "system"):
        raise HTTPException(status_code=403, detail="Admin access required")


@router.get("/texts", response_class=HTMLResponse)
def admin_texts_page(
    request: Request,
    db: Session = Depends(get_global_db),
    account: Account = Depends(get_current_account),
):
    """Admin page showing all texts and their states."""
    _require_admin(account)
    
    t = _get_templates()
    
    # Get all texts ordered by created_at desc
    texts = db.query(ReadingText).order_by(ReadingText.created_at.desc()).all()
    
    # Calculate summary stats
    total_count = len(texts)
    ready_count = sum(1 for t in texts if t.content and t.words_complete and t.sentences_complete)
    failed_count = sum(1 for t in texts if t.translation_attempts >= 3)
    pending_count = total_count - ready_count - failed_count
    
    # By language stats
    by_lang = defaultdict(lambda: {"total": 0, "ready": 0})
    for text in texts:
        by_lang[text.lang]["total"] += 1
        if text.content and text.words_complete and text.sentences_complete:
            by_lang[text.lang]["ready"] += 1
    
    context = {
        "title": "Admin - Text Pool",
        "texts": texts,
        "total_count": total_count,
        "ready_count": ready_count,
        "pending_count": pending_count,
        "failed_count": failed_count,
        "by_lang": dict(by_lang),
    }
    
    return t.TemplateResponse(request, "pages/admin_texts.html", context)


@router.get("/texts/{text_id}", response_class=HTMLResponse)
def admin_text_detail(
    text_id: int,
    request: Request,
    db: Session = Depends(get_global_db),
    account: Account = Depends(get_current_account),
):
    """Admin page showing detail for a single text."""
    _require_admin(account)
    
    text = db.query(ReadingText).filter(ReadingText.id == text_id).first()
    if not text:
        raise HTTPException(status_code=404, detail="Text not found")
    
    # Get word glosses count
    from ..models import ReadingWordGloss, ReadingTextTranslation
    word_count = db.query(ReadingWordGloss).filter(ReadingWordGloss.text_id == text_id).count()
    sentence_count = db.query(ReadingTextTranslation).filter(
        ReadingTextTranslation.text_id == text_id,
        ReadingTextTranslation.unit == "sentence"
    ).count()
    
    t = _get_templates()
    context = {
        "title": f"Text #{text_id}",
        "text": text,
        "word_count": word_count,
        "sentence_count": sentence_count,
    }
    
    return t.TemplateResponse(request, "pages/admin_text_detail.html", context)


# Available tiers for dropdown (unified access control + features)
AVAILABLE_TIERS = ["Free", "Standard", "Pro", "Pro+", "BYOK", "admin", "system"]


@router.get("/accounts", response_class=HTMLResponse)
def admin_accounts_page(
    request: Request,
    db: Session = Depends(get_global_db),
    account: Account = Depends(get_current_account),
):
    """Admin page showing all accounts."""
    _require_admin(account)
    
    t = _get_templates()
    
    # Get all accounts
    accounts = db.query(Account).order_by(Account.created_at.desc()).all()
    
    # Get profiles for each account
    profiles_by_account = defaultdict(list)
    all_profiles = db.query(Profile).all()
    for profile in all_profiles:
        profiles_by_account[profile.account_id].append(profile)
    
    # Attach profiles to accounts
    for acc in accounts:
        acc.profiles = profiles_by_account.get(acc.id, [])
    
    # Stats
    total_count = len(accounts)
    admin_count = sum(1 for a in accounts if a.subscription_tier in ("admin", "system"))
    with_profile_count = sum(1 for a in accounts if profiles_by_account.get(a.id))
    total_profiles = len(all_profiles)
    
    context = {
        "title": "Admin - Accounts",
        "accounts": accounts,
        "tiers": AVAILABLE_TIERS,
        "total_count": total_count,
        "admin_count": admin_count,
        "with_profile_count": with_profile_count,
        "total_profiles": total_profiles,
    }
    
    return t.TemplateResponse(request, "pages/admin_accounts.html", context)


@router.post("/accounts/{account_id}/tier", response_class=HTMLResponse)
def update_account_tier(
    account_id: int,
    tier: str = Form(...),
    db: Session = Depends(get_global_db),
    account: Account = Depends(get_current_account),
):
    """Update an account's subscription tier."""
    _require_admin(account)
    
    if tier not in AVAILABLE_TIERS:
        raise HTTPException(status_code=400, detail="Invalid tier")
    
    target_account = db.query(Account).filter(Account.id == account_id).first()
    if not target_account:
        raise HTTPException(status_code=404, detail="Account not found")
    
    target_account.subscription_tier = tier
    db.commit()
    
    return RedirectResponse(url="/admin/accounts", status_code=303)


@router.get("/accounts/{account_id}", response_class=HTMLResponse)
def admin_account_detail(
    account_id: int,
    request: Request,
    db: Session = Depends(get_global_db),
    account: Account = Depends(get_current_account),
):
    """Admin page showing detail for a single account."""
    _require_admin(account)
    
    target = db.query(Account).filter(Account.id == account_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="Account not found")
    
    profiles = db.query(Profile).filter(Profile.account_id == account_id).all()
    
    t = _get_templates()
    context = {
        "title": f"Account #{account_id}",
        "target": target,
        "profiles": profiles,
    }
    
    return t.TemplateResponse(request, "pages/admin_account_detail.html", context)
