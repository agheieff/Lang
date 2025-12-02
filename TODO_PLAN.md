# TODO Plan

## Critical: Database Architecture Fixes

The global pool architecture separates data into:
- **Global DB** (`data/arcadia_lang.db`): accounts, profiles, languages, reading_texts, translations, word_glosses
- **Per-Account DB** (`data/user_<id>.db`): lexemes, word_events, reading_history, user_configs

Many services still query tables from the wrong database.

### High Priority Fixes

#### 1. word_selection.py - Profile queries use wrong DB
**File:** `server/services/word_selection.py`
**Issue:** `_profile_for_lang()` queries Profile from `account_db` but Profile is in global DB
**Affected functions:**
- `urgent_words_detailed()` - calls `_profile_for_lang(db, user, lang)`
- `_variant_form_for_lang()` - calls `_profile_for_lang(db, user, lang)`
- `pick_words()` - calls `urgent_words_detailed()`

**Solution:** Refactor to accept both `global_db` and `account_db`, or pass profile as parameter

#### 2. level_service.py - Profile queries use wrong DB
**File:** `server/services/level_service.py`
**Issue:** `update_level_if_stale()` and `get_ci_target()` query Profile
**Solution:** These need global_db for Profile queries

#### 3. Other services with mixed DB needs
Review and fix these files:
- `server/services/translation_service.py`
- `server/services/retry_actions.py`
- `server/services/srs_service.py`
- `server/services/session_processing_service.py`
- `server/services/selection_service.py`
- `server/services/state_manager.py`

### Medium Priority

#### 4. System account not visible in admin
**Issue:** System account only created when `ARC_SYSTEM_API_KEY` env var is set
**File:** `server/services/startup_service.py`
**Decision needed:** Should system account always be created, or only when API key is configured?

#### 5. Text generation triggers without API key
**Issue:** Generation starts and creates placeholder texts that stay "Pending" forever
**Solution options:**
- Check for API key before triggering generation
- Show warning in UI when no API key configured
- Don't auto-trigger generation without valid LLM config

#### 6. _get_recent_read_titles returns empty
**File:** `server/services/llm_common.py`
**Issue:** Function was stubbed out because old schema references (account_id, read_at) don't exist
**Solution:** Implement properly using ProfileTextRead from account_db + ReadingText from global_db

## Testing

### Integration Tests Needed

#### 7. Create integration tests for basic user flows
**Priority:** High
**Tests to create:**
```
tests/integration/
├── test_auth_flow.py          # signup, login, logout
├── test_profile_flow.py       # create profile, view profile
├── test_admin_pages.py        # admin/texts, admin/accounts access
├── test_dashboard_flow.py     # dashboard, settings pages
└── test_reading_flow.py       # reading page (without actual LLM calls)
```

**Test scenarios:**
1. User signup → login → view dashboard
2. Create language profile → view settings
3. Admin user can access admin pages
4. Non-admin cannot access admin pages
5. Pages load without DB errors

## Architecture Documentation

### 8. Document DB architecture clearly
Create or update documentation explaining:
- Which tables are in global vs per-account DB
- How to properly query each
- Session management patterns (GlobalSessionLocal vs db_manager)

## Quick Reference: Table Locations

### Global DB (arcadia_lang.db)
- accounts
- profiles
- profile_prefs
- languages
- reading_texts
- reading_text_translations
- reading_word_glosses
- text_vocabulary
- generation_logs

### Per-Account DB (user_<id>.db)
- lexemes
- lexeme_variants
- user_lexeme_contexts
- word_events
- profile_text_reads
- profile_text_queue
- user_model_configs
- usage_tracking
- next_ready_overrides

## Session Patterns

```python
# Global DB access
from server.db import GlobalSessionLocal, get_global_db, open_global_session

# In routes (dependency injection):
db: Session = Depends(get_global_db)

# In background threads:
db = open_global_session()
try:
    # ... use db
finally:
    db.close()

# Per-Account DB access
from server.utils.session_manager import db_manager

# Transaction (read-write):
with db_manager.transaction(account_id) as db:
    # ... use db

# Read-only:
with db_manager.read_only(account_id) as db:
    # ... use db
```
