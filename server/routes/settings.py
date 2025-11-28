from __future__ import annotations

from typing import Optional, List
from fastapi import APIRouter, Depends, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from server.auth import Account
from server.deps import get_current_account
from server.account_db import get_db
from server.db import get_global_db
from server.models import UserProviderConfig, Profile, SubscriptionTier
from server.services.usage_service import get_usage_service
from server.config import TIER_SPENDING_LIMITS, FREE_TIER_TEXT_LIMIT, FREE_TIER_CHAR_LIMIT, TOPICS, DEFAULT_TOPIC_WEIGHTS

router = APIRouter(tags=["settings"])

# --- Schemas ---
class ProviderConfigCreate(BaseModel):
    provider_name: str
    provider_type: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    default_model: Optional[str] = None
    app_title: Optional[str] = None
    referer: Optional[str] = None

# --- Routes ---

@router.get("/settings/providers", response_class=HTMLResponse)
def list_providers(
    request: Request,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    """HTMX: Returns the list of configured providers as HTML rows/cards."""
    configs = db.query(UserProviderConfig).filter(
        UserProviderConfig.account_id == account.id
    ).all()
    
    # Simple HTML template for the list items
    # In a real app, this might use a Jinja2 partial.
    # For simplicity here, we'll generate safe HTML string or use a small inline template.
    
    if not configs:
        return """
        <div class="text-gray-500 italic p-4 text-center">
            No custom providers configured. Using system defaults.
        </div>
        """
    
    html_parts = []
    for conf in configs:
        # Mask API key for display
        masked_key = "****" + conf.api_key[-4:] if conf.api_key and len(conf.api_key) > 4 else "Not set"
        
        html = f"""
        <div class="border rounded p-4 mb-2 flex justify-between items-center bg-white" id="provider-{conf.id}">
            <div>
                <div class="font-bold">{conf.provider_name} <span class="text-xs text-gray-500 bg-gray-100 px-2 py-0.5 rounded ml-2">{conf.provider_type}</span></div>
                <div class="text-sm text-gray-600">Url: {conf.base_url or 'Default'}</div>
                <div class="text-sm text-gray-600">Key: {masked_key}</div>
                <div class="text-sm text-gray-600">Default Model: {conf.default_model or 'Auto'}</div>
                {f'<div class="text-xs text-gray-500 mt-1">App Title: {conf.app_title} | Referer: {conf.referer}</div>' if conf.app_title or conf.referer else ''}
            </div>
            <div class="space-x-2">
                <button 
                    hx-delete="/settings/providers/{conf.id}"
                    hx-confirm="Are you sure you want to delete this provider?"
                    hx-target="#provider-{conf.id}"
                    hx-swap="outerHTML"
                    class="text-red-600 hover:text-red-800 text-sm font-medium">
                    Delete
                </button>
            </div>
        </div>
        """
        html_parts.append(html)
        
    return "\n".join(html_parts)


@router.post("/settings/providers", response_class=HTMLResponse)
def create_provider(
    provider_name: str = Form(...),
    provider_type: str = Form(...),
    base_url: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    default_model: Optional[str] = Form(None),
    app_title: Optional[str] = Form(None),
    referer: Optional[str] = Form(None),
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    # Check uniqueness
    existing = db.query(UserProviderConfig).filter(
        UserProviderConfig.account_id == account.id,
        UserProviderConfig.provider_name == provider_name
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Provider name already exists")
    
    new_config = UserProviderConfig(
        account_id=account.id,
        provider_name=provider_name,
        provider_type=provider_type,
        base_url=base_url,
        api_key=api_key,
        default_model=default_model,
        app_title=app_title,
        referer=referer
    )
    db.add(new_config)
    db.commit()
    db.refresh(new_config)
    
    # Return the updated list
    # We can just trigger a reload of the list via HTMX headers or return the new item
    # Ideally, we return the whole list or the new item appended.
    # Let's return the trigger to reload the list
    response = HTMLResponse('<span class="text-green-600">Provider added successfully!</span>')
    response.headers["HX-Trigger"] = "reload-providers"
    return response


@router.delete("/settings/providers/{provider_id}", response_class=HTMLResponse)
def delete_provider(
    provider_id: int,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    config = db.query(UserProviderConfig).filter(
        UserProviderConfig.id == provider_id,
        UserProviderConfig.account_id == account.id
    ).first()
    
    if not config:
        raise HTTPException(status_code=404, detail="Provider not found")
        
    db.delete(config)
    db.commit()
    
    return ""  # Return empty string to remove the element from DOM


@router.put("/settings/preferences", response_class=HTMLResponse)
def update_global_preferences(
    preferred_generation_model: Optional[str] = Form(None),
    preferred_translation_model: Optional[str] = Form(None),
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    profile = db.query(Profile).filter(Profile.account_id == account.id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
        
    # Update settings JSON
    settings = dict(profile.settings or {})
    
    if preferred_generation_model is not None:
        settings["preferred_generation_model"] = preferred_generation_model
        
    if preferred_translation_model is not None:
        settings["preferred_translation_model"] = preferred_translation_model
        
    profile.settings = settings
    db.commit()
    
    return '<span class="text-green-600">Preferences saved</span>'


# --- Tier Management Routes ---

TIER_DESCRIPTIONS = {
    "Free": "Basic access with usage limits",
    "Standard": "Extended limits, more models",
    "Pro": "Higher limits, premium models",
    "Pro+": "Maximum limits, all models",
    "BYOK": "Bring your own API keys",
}

TIER_COLORS = {
    "Free": ("gray", "gray"),
    "Standard": ("blue", "blue"),
    "Pro": ("purple", "purple"),
    "Pro+": ("amber", "yellow"),
    "BYOK": ("green", "green"),
}


@router.get("/settings/tier", response_class=HTMLResponse)
def get_tier_section(
    account: Account = Depends(get_current_account),
    account_db: Session = Depends(get_db),
    global_db: Session = Depends(get_global_db),
):
    """HTMX: Returns the tier info section."""
    current_tier = account.subscription_tier or "Free"
    
    # Get usage stats
    usage_service = get_usage_service()
    usage_stats = usage_service.get_usage_stats(account_db, account.id, current_tier)
    
    # Get available tiers
    from server.repos.tiers import ensure_default_tiers
    ensure_default_tiers(global_db)
    tiers = global_db.query(SubscriptionTier).all()
    tier_order = ["Free", "Standard", "Pro", "Pro+", "BYOK"]
    tiers = sorted([t for t in tiers if t.name != "admin"], 
                   key=lambda t: tier_order.index(t.name) if t.name in tier_order else 99)
    
    # Build tier badge
    color = TIER_COLORS.get(current_tier, ("gray", "gray"))
    tier_badge = f'<span class="bg-{color[0]}-100 text-{color[1]}-800 px-3 py-1 rounded-full text-sm font-medium">{current_tier}</span>'
    
    # Build usage section for Free tier
    usage_html = ""
    if current_tier == "Free" and usage_stats.get("texts_generated") is not None:
        texts_pct = min(100, (usage_stats["texts_generated"] / FREE_TIER_TEXT_LIMIT) * 100)
        chars_pct = min(100, (usage_stats["chars_generated"] / FREE_TIER_CHAR_LIMIT) * 100)
        
        usage_html = f"""
        <div class="mt-4 p-4 bg-gray-50 rounded-lg">
            <h4 class="text-sm font-medium text-gray-700 mb-3">Monthly Usage</h4>
            <div class="space-y-3">
                <div>
                    <div class="flex justify-between text-xs text-gray-600 mb-1">
                        <span>Texts Generated</span>
                        <span>{usage_stats["texts_generated"]} / {FREE_TIER_TEXT_LIMIT}</span>
                    </div>
                    <div class="w-full bg-gray-200 rounded-full h-2">
                        <div class="bg-blue-600 h-2 rounded-full transition-all" style="width: {texts_pct}%"></div>
                    </div>
                </div>
                <div>
                    <div class="flex justify-between text-xs text-gray-600 mb-1">
                        <span>Characters Generated</span>
                        <span>{usage_stats["chars_generated"]:,} / {FREE_TIER_CHAR_LIMIT:,}</span>
                    </div>
                    <div class="w-full bg-gray-200 rounded-full h-2">
                        <div class="bg-green-600 h-2 rounded-full transition-all" style="width: {chars_pct}%"></div>
                    </div>
                </div>
            </div>
            <p class="text-xs text-gray-500 mt-2">Resets monthly. Upgrade for higher limits.</p>
        </div>
        """
    elif current_tier != "Free":
        limit = TIER_SPENDING_LIMITS.get(current_tier)
        if limit:
            usage_html = f"""
            <div class="mt-4 p-4 bg-gray-50 rounded-lg">
                <p class="text-sm text-gray-600">
                    Monthly spending limit: <span class="font-medium">${limit}</span>
                </p>
                <p class="text-xs text-gray-500 mt-1">Usage tracked via OpenRouter.</p>
            </div>
            """
        elif current_tier == "BYOK":
            usage_html = """
            <div class="mt-4 p-4 bg-gray-50 rounded-lg">
                <p class="text-sm text-gray-600">Using your own API keys - no platform limits.</p>
            </div>
            """
    
    # Build tier selector
    tier_options = []
    for t in tiers:
        selected = "selected" if t.name == current_tier else ""
        desc = TIER_DESCRIPTIONS.get(t.name, "")
        tier_options.append(f'<option value="{t.name}" {selected}>{t.name} - {desc}</option>')
    
    return f"""
    <div class="flex items-center justify-between mb-4">
        <div>
            <span class="text-gray-600 mr-2">Current Tier:</span>
            {tier_badge}
        </div>
    </div>
    
    {usage_html}
    
    <div class="mt-6 pt-4 border-t border-gray-200">
        <label class="block text-sm font-medium text-gray-700 mb-2">Change Tier</label>
        <div class="flex gap-2">
            <select id="tier-select" class="flex-1 px-3 py-2 border border-gray-300 rounded-md text-sm">
                {''.join(tier_options)}
            </select>
            <button 
                hx-post="/settings/tier"
                hx-include="#tier-select"
                hx-target="#tier-section"
                hx-swap="innerHTML"
                class="px-4 py-2 bg-blue-600 text-white rounded-md text-sm hover:bg-blue-700">
                Update
            </button>
        </div>
        <p class="text-xs text-gray-500 mt-2">Note: Tier changes may affect available models and API key provisioning.</p>
        <div id="tier-status" class="mt-2"></div>
    </div>
    """


@router.post("/settings/tier", response_class=HTMLResponse)
async def set_tier_html(
    request: Request,
    account: Account = Depends(get_current_account),
    account_db: Session = Depends(get_db),
    global_db: Session = Depends(get_global_db),
):
    """HTMX: Update tier and return refreshed section."""
    form = await request.form()
    new_tier = form.get("tier-select", "Free")
    
    # Validate tier
    from server.repos.tiers import ensure_default_tiers
    ensure_default_tiers(global_db)
    allowed = {t.name for t in global_db.query(SubscriptionTier).all()}
    if new_tier not in allowed:
        return '<div class="text-red-600 text-sm">Invalid tier selected</div>'
    
    # Get account from global DB to update
    acc = global_db.get(Account, account.id)
    if not acc:
        return '<div class="text-red-600 text-sm">Account not found</div>'
    
    old_tier = acc.subscription_tier or "Free"
    
    if old_tier == new_tier:
        # No change, just refresh
        return get_tier_section(account=account, account_db=account_db, global_db=global_db)
    
    # Update tier
    acc.subscription_tier = new_tier
    global_db.commit()
    
    # Handle OpenRouter key provisioning
    from server.config import PAID_TIERS
    from server.services.openrouter_key_service import get_openrouter_key_service, OpenRouterKeyError
    
    message = None
    try:
        key_service = get_openrouter_key_service()
        
        if new_tier in PAID_TIERS and old_tier not in PAID_TIERS:
            # Upgrading to paid - provision key
            await key_service.create_user_key(global_db, acc, new_tier)
            global_db.refresh(acc)
            if acc.openrouter_key_id:
                limit = TIER_SPENDING_LIMITS.get(new_tier)
                message = f"OpenRouter key provisioned (${limit}/mo limit)"
                
        elif old_tier in PAID_TIERS and new_tier not in PAID_TIERS:
            # Downgrading - revoke key
            if acc.openrouter_key_id:
                await key_service.revoke_user_key(global_db, acc)
                message = "OpenRouter key revoked"
                
        elif new_tier in PAID_TIERS and old_tier in PAID_TIERS:
            # Changing between paid tiers - update limit
            if acc.openrouter_key_id:
                new_limit = TIER_SPENDING_LIMITS.get(new_tier)
                await key_service.update_key_limit(global_db, acc, new_limit)
                message = f"Key limit updated to ${new_limit}/mo"
                
    except OpenRouterKeyError as e:
        message = f"Tier updated, key operation pending: {e}"
    except Exception:
        pass  # Key provisioning failed but tier still updated
    
    # Sync system models
    from server.services.user_model_service import get_user_model_service
    model_service = get_user_model_service()
    model_service.sync_system_models_for_tier(account_db, account.id, new_tier)
    
    # Refresh account for template
    global_db.refresh(acc)
    
    # Return updated section with success message
    html = get_tier_section(account=acc, account_db=account_db, global_db=global_db)
    
    if message:
        html += f'<div class="mt-2 text-sm text-green-600">{message}</div>'
    else:
        html += f'<div class="mt-2 text-sm text-green-600">Tier updated to {new_tier}</div>'
    
    return html


# --- Topic Interests Routes ---

TOPIC_LABELS = {
    "fiction": "Fiction",
    "news": "News",
    "science": "Science",
    "technology": "Technology",
    "history": "History",
    "daily_life": "Daily Life",
    "culture": "Culture",
    "sports": "Sports",
    "business": "Business",
}

WEIGHT_STEP = 0.15  # Amount to change weight per click


def _weight_to_color(weight: float) -> str:
    """Convert weight to a color. Green for positive (>1), red for negative (<1), gray for neutral."""
    if weight >= 1.5:
        return "bg-green-500 text-white"
    elif weight >= 1.2:
        return "bg-green-400 text-white"
    elif weight >= 1.0:
        return "bg-green-200 text-green-800"
    elif weight >= 0.8:
        return "bg-red-200 text-red-800"
    elif weight >= 0.5:
        return "bg-red-400 text-white"
    else:
        return "bg-red-500 text-white"


def _render_topic_chip(topic: str, weight: float, lang: str) -> str:
    """Render a single topic chip with weight and color."""
    label = TOPIC_LABELS.get(topic, topic.replace("_", " ").title())
    color_class = _weight_to_color(weight)
    weight_display = f"{weight:.2f}"
    
    return f"""
    <div class="inline-flex items-center gap-1 rounded-full px-3 py-1.5 text-sm cursor-pointer transition-all hover:scale-105 {color_class}"
         hx-post="/settings/topics/{topic}/adjust?lang={lang}&delta={WEIGHT_STEP}"
         hx-target="#topics-section"
         hx-swap="innerHTML"
         title="Click to increase weight">
        <span class="font-medium">{label}</span>
        <span class="text-xs opacity-75">({weight_display})</span>
    </div>
    """


@router.get("/settings/topics", response_class=HTMLResponse)
def get_topics_section(
    lang: Optional[str] = None,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    """HTMX: Returns the topic interests section."""
    # Get user's profile
    profile = db.query(Profile).filter(Profile.account_id == account.id).first()
    if not profile:
        return '<div class="text-gray-500 italic">No profile found. Create a profile first.</div>'
    
    effective_lang = lang or profile.lang
    
    # Get topic weights (or defaults)
    weights = profile.topic_weights if profile.topic_weights else DEFAULT_TOPIC_WEIGHTS.copy()
    
    # Ensure all topics have weights
    for topic in TOPICS:
        if topic not in weights:
            weights[topic] = 1.0
    
    # Split into positive (>=1) and negative (<1)
    positive = [(t, w) for t, w in weights.items() if w >= 1.0 and t in TOPICS]
    negative = [(t, w) for t, w in weights.items() if w < 1.0 and t in TOPICS]
    
    # Sort by weight (highest first for positive, lowest first for negative)
    positive.sort(key=lambda x: -x[1])
    negative.sort(key=lambda x: x[1])
    
    # Build HTML
    positive_chips = " ".join(_render_topic_chip(t, w, effective_lang) for t, w in positive)
    negative_chips = " ".join(_render_topic_chip(t, w, effective_lang) for t, w in negative)
    
    if not positive_chips:
        positive_chips = '<span class="text-gray-400 text-sm italic">No preferred topics</span>'
    if not negative_chips:
        negative_chips = '<span class="text-gray-400 text-sm italic">No avoided topics</span>'
    
    return f"""
    <div class="space-y-4">
        <div>
            <h4 class="text-sm font-medium text-gray-700 mb-2">Preferred Topics (click to boost)</h4>
            <div class="flex flex-wrap gap-2">
                {positive_chips}
            </div>
        </div>
        <div>
            <h4 class="text-sm font-medium text-gray-700 mb-2">Less Preferred Topics (click to boost)</h4>
            <div class="flex flex-wrap gap-2">
                {negative_chips}
            </div>
        </div>
        <div class="pt-2 border-t border-gray-200">
            <p class="text-xs text-gray-500">
                Click any topic to increase its weight. 
                <button hx-post="/settings/topics/reset?lang={effective_lang}" 
                        hx-target="#topics-section" 
                        hx-swap="innerHTML"
                        hx-confirm="Reset all topic weights to default?"
                        class="text-blue-600 hover:underline ml-1">Reset all</button>
            </p>
        </div>
    </div>
    """


@router.post("/settings/topics/{topic}/adjust", response_class=HTMLResponse)
def adjust_topic_weight(
    topic: str,
    lang: str,
    delta: float = WEIGHT_STEP,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    """HTMX: Adjust a topic's weight and return updated section."""
    if topic not in TOPICS:
        raise HTTPException(400, "Invalid topic")
    
    profile = db.query(Profile).filter(
        Profile.account_id == account.id,
        Profile.lang == lang
    ).first()
    
    if not profile:
        raise HTTPException(404, "Profile not found")
    
    # Get current weights
    weights = dict(profile.topic_weights) if profile.topic_weights else DEFAULT_TOPIC_WEIGHTS.copy()
    
    # Adjust weight (clamp between 0.1 and 2.0)
    current = weights.get(topic, 1.0)
    new_weight = min(2.0, max(0.1, current + delta))
    weights[topic] = round(new_weight, 2)
    
    # Save
    profile.topic_weights = weights
    db.commit()
    
    # Return updated section
    return get_topics_section(lang=lang, account=account, db=db)


@router.post("/settings/topics/reset", response_class=HTMLResponse)
def reset_topic_weights(
    lang: str,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    """HTMX: Reset all topic weights to default."""
    profile = db.query(Profile).filter(
        Profile.account_id == account.id,
        Profile.lang == lang
    ).first()
    
    if not profile:
        raise HTTPException(404, "Profile not found")
    
    profile.topic_weights = DEFAULT_TOPIC_WEIGHTS.copy()
    db.commit()
    
    return get_topics_section(lang=lang, account=account, db=db)


@router.post("/settings/topics/{topic}/decrease", response_class=HTMLResponse)
def decrease_topic_weight(
    topic: str,
    lang: str,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    """HTMX: Decrease a topic's weight."""
    return adjust_topic_weight(topic, lang, delta=-WEIGHT_STEP, account=account, db=db)
