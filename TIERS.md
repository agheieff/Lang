# Tier System with OpenRouter Provisioned Keys

## Overview

This document specifies how subscription tiers integrate with OpenRouter's Provisioned Keys API to provide per-user rate limiting, spending control, and usage tracking.

## Tier Definitions

| Tier | Monthly Limit | Key Type | Models Available |
|------|---------------|----------|------------------|
| Free | $2 | Shared key + usage tracking | Free models only (Kimi, DeepSeek) |
| Standard | $10 | Provisioned sub-key | Free + GPT-4o-mini, Claude Haiku |
| Pro | $50 | Provisioned sub-key | Free + Standard + GPT-4o, Claude Sonnet |
| Pro+ | $200 | Provisioned sub-key | All models including Claude Opus, o1 |
| BYOK | N/A | User's own keys | Whatever their keys support |

## OpenRouter Provisioned Keys

### What They Are
OpenRouter allows creating sub-keys programmatically via their API. These keys:
- Are tied to our master OpenRouter account (we pay the bill)
- Have individual spending limits with auto-reset
- Provide per-key usage tracking
- Can be revoked/updated at any time

### API Reference

**Create Key:**
```
POST https://openrouter.ai/api/v1/keys
Authorization: Bearer <PROVISIONING_KEY>
Content-Type: application/json

{
  "name": "arcadia-user-{account_id}",
  "limit": 10.00,
  "limit_reset": "monthly",
  "expires_at": null
}

Response:
{
  "key": "<GENERATED_API_KEY>",  // Only shown once!
  "hash": "abc123def456",
  "name": "arcadia-user-123",
  "limit": 10.00,
  "limit_reset": "monthly",
  "usage": 0.0,
  "created_at": "2024-01-01T00:00:00Z"
}
```

**Update Key Limit:**
```
PATCH https://openrouter.ai/api/v1/keys/{key_hash}
Authorization: Bearer <PROVISIONING_KEY>

{
  "limit": 50.00,
  "limit_reset": "monthly"
}
```

**Delete Key:**
```
DELETE https://openrouter.ai/api/v1/keys/{key_hash}
Authorization: Bearer <PROVISIONING_KEY>
```

**Get Key Usage:**
```
GET https://openrouter.ai/api/v1/keys/{key_hash}
Authorization: Bearer <PROVISIONING_KEY>
```

## Implementation Plan

### Phase 1: Infrastructure

1. **Environment Setup**
   - Create Provisioning Key in OpenRouter dashboard
   - Add `OPENROUTER_PROVISIONING_KEY` to env vars
   - Keep `OPENROUTER_API_KEY` for Free tier shared usage

2. **Database Changes**
   Add to `UserModelConfig` or new table:
   ```python
   class UserOpenRouterKey(Base):
       __tablename__ = "user_openrouter_keys"
       
       id: int  # PK
       account_id: int  # FK to account
       key_hash: str  # OpenRouter's key hash (for API calls)
       encrypted_key: str  # The actual API key, encrypted
       tier_at_creation: str  # Which tier when created
       limit_usd: float  # Current spending limit
       limit_reset: str  # "monthly", "weekly", etc.
       created_at: datetime
       updated_at: datetime
   ```

3. **Service: `OpenRouterKeyService`**
   ```python
   class OpenRouterKeyService:
       def create_key(account_id: int, tier: str) -> str:
           """Create provisioned key, store hash, return API key."""
           
       def update_key_limit(account_id: int, new_limit: float) -> bool:
           """Update spending limit when tier changes."""
           
       def delete_key(account_id: int) -> bool:
           """Revoke key (tier downgrade to Free or BYOK)."""
           
       def get_usage(account_id: int) -> dict:
           """Get current usage stats from OpenRouter."""
   ```

### Phase 2: Tier Change Integration

**On User Signup (Free tier):**
- No provisioned key created
- Use shared `OPENROUTER_API_KEY`
- Track usage in local DB for quota enforcement

**On Upgrade to Standard/Pro/Pro+:**
1. Call `OpenRouterKeyService.create_key(account_id, new_tier)`
2. Store encrypted key in DB
3. Update `UserModelConfig` system models to use new key
4. Sync available models for new tier

**On Downgrade:**
- Pro+ → Pro/Standard: Update key limit via API
- Any → Free: Delete provisioned key, revert to shared key
- Any → BYOK: Delete provisioned key (user provides own)

**On Tier Change (in `api/tiers.py`):**
```python
@router.post("/me/tier")
def set_my_tier(payload: TierIn, ...):
    old_tier = acc.subscription_tier
    new_tier = payload.name
    
    # Update tier in auth DB
    acc.subscription_tier = new_tier
    auth_db.commit()
    
    # Handle OpenRouter key lifecycle
    key_service = get_openrouter_key_service()
    
    if old_tier == "Free" and new_tier in ["Standard", "Pro", "Pro+"]:
        # Create provisioned key
        key_service.create_key(account.id, new_tier)
    elif old_tier in ["Standard", "Pro", "Pro+"] and new_tier == "Free":
        # Delete provisioned key
        key_service.delete_key(account.id)
    elif old_tier in ["Standard", "Pro", "Pro+"] and new_tier in ["Standard", "Pro", "Pro+"]:
        # Update limit
        key_service.update_key_limit(account.id, TIER_LIMITS[new_tier])
    elif new_tier == "BYOK":
        # Delete provisioned key if exists
        key_service.delete_key(account.id)
    
    # Sync system models
    model_service.sync_system_models_for_tier(account_db, account.id, new_tier)
```

### Phase 3: Free Tier Usage Tracking

For Free tier users (shared key), track usage locally:

```python
class UsageTracker(Base):
    __tablename__ = "usage_tracking"
    
    id: int
    account_id: int
    period_start: date  # First of month
    tokens_used: int
    estimated_cost_usd: float
    last_updated: datetime

class UsageService:
    def check_quota(account_id: int) -> bool:
        """Return True if user can make requests."""
        
    def record_usage(account_id: int, tokens: int, cost: float):
        """Called after each LLM request."""
        
    def get_remaining_quota(account_id: int) -> dict:
        """Return usage stats for UI."""
```

**Enforcement point** (in `model_resolution.py` or LLM client):
```python
def resolve_model_for_task(...):
    if user_tier == "Free":
        if not usage_service.check_quota(account_id):
            raise QuotaExceededError("Monthly limit reached")
    # ... continue with model resolution
```

### Phase 4: Key Security

1. **Encryption at rest**
   - Use Fernet symmetric encryption for stored keys
   - Encryption key from env var `ARCADIA_KEY_ENCRYPTION_SECRET`
   
2. **Key rotation**
   - If key is compromised, delete and recreate
   - Log all key operations for audit

3. **Never expose in API responses**
   - `UserModelConfig.is_key_visible = False` for system models
   - API responses show only masked version or "[system]"

## Configuration

```python
# settings.py or env vars
TIER_LIMITS = {
    "Free": 2.00,      # $2/month (tracked locally, shared key)
    "Standard": 10.00, # $10/month (provisioned key)
    "Pro": 50.00,      # $50/month (provisioned key)
    "Pro+": 200.00,    # $200/month (provisioned key)
    "BYOK": None,      # No limit (user's own keys)
}

OPENROUTER_PROVISIONING_KEY = os.getenv("OPENROUTER_PROVISIONING_KEY")
OPENROUTER_SHARED_KEY = os.getenv("OPENROUTER_API_KEY")  # For Free tier
```

## Error Handling

| Scenario | Handling |
|----------|----------|
| OpenRouter API down during key creation | Retry with exponential backoff, fall back to shared key temporarily |
| Key creation rate limited | Queue and process async |
| User hits spending limit | Return friendly error, suggest upgrade |
| Provisioned key compromised | Admin endpoint to force-rotate key |

## Future Enhancements

1. **Usage dashboard** - Show users their usage stats
2. **Alerts** - Email when approaching limit (80%, 100%)
3. **Overage handling** - Option to allow overage with extra charges
4. **Usage analytics** - Track which models users prefer, average costs
5. **Bulk key management** - Admin tools for managing all keys

## Files to Create/Modify

**New:**
- `server/services/openrouter_key_service.py`
- `server/services/usage_service.py`
- `server/models.py` - Add `UserOpenRouterKey`, `UsageTracking`

**Modify:**
- `server/api/tiers.py` - Add key lifecycle calls
- `server/services/model_resolution.py` - Add quota check for Free tier
- `server/services/user_model_service.py` - Use provisioned keys for system models
- `server/llm/client.py` - Record usage after each call
