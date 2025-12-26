# Arcadia Lang - Systematic Audit & Fix Plan

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (HTML/JS)                  │
│  - Templates: /server/templates/pages/*.html                │
│  - Static JS: /server/static/app.js                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Routes                         │
│  /auth/* → server/auth.py (create_auth_router)          │
│  /profile, /settings, /stats, /words → server/routes/user.py      │
│  /reading/* → server/routes/reading.py                     │
│  /srs/* → server/routes/srs.py                            │
│  /admin/* → server/routes/admin.py                          │
│  /login, /signup, / → server/routes/pages.py              │
│  /health → server/routes/health.py                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Services Layer                         │
│  /server/services/learning.py                                │
│  /server/services/content.py                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Data Layer                             │
│  Models: /server/models.py (Base = declarative_base())    │
│  DB: /server/db.py (SessionLocal, engine, init_db)       │
└─────────────────────────────────────────────────────────────────┘
```

## Completed Fixes ✅

### Phase 1: Core Auth & Profile - COMPLETED

- ✅ Added `/me/profile` endpoint (alias to `/profile/api`)
- ✅ Added `/languages` endpoint (returns supported languages)
- ✅ Fixed Profile model reference (removed `updated_at` which doesn't exist)
- ✅ Removed broken static file references from base.html
- ✅ Fixed auth router return statement
- ✅ Added `create_sqlite_repo` function
- ✅ Fixed database Base import (now uses models.Base)
- ✅ Removed broken `auth_repo.py` file

### Phase 2: Minimal Reading Functionality - COMPLETED

- ✅ Created `/reading` endpoint with demo text
- ✅ Created `/reading/current` endpoint (returns HTML fragment)
- ✅ Created `/reading/next` endpoint (placeholder)
- ✅ Created `/reading/{text_id}/translations` endpoint
- ✅ Created `/reading/{text_id}/status` endpoint
- ✅ Created `/reading/word-click` endpoint (placeholder)
- ✅ Created `/reading/save-session` endpoint (placeholder)
- ✅ Created `/words` page endpoint
- ✅ Created `/stats` page endpoint
- ✅ Created `/srs/words` API endpoint
- ✅ Created reading.html template

## Current Working Features

| Feature | Status | Notes |
|---------|--------|-------|
| Authentication (signup/login) | ✅ WORKING | JWT tokens stored in cookies |
| Profile creation | ✅ WORKING | Creates Profile in DB |
| Dashboard | ✅ WORKING | Shows profile and options |
| Reading page | ✅ WORKING | Demo text, translation toggle |
| Words page | ✅ WORKING | Empty list (SRS not implemented) |
| Stats page | ✅ WORKING | Placeholder |
| Settings page | ✅ WORKING | Edit profile |
| Admin pages | ✅ WORKING | Text/account management |

## Remaining Issues

### 1. Database/Model Issues

- ⚠️ Mixed SQLAlchemy styles: Some models use `Column`, others use `Mapped`
- ⚠️ `Account` model has new columns (`openrouter_key_encrypted`, `openrouter_key_id`) - requires DB reset

### 2. Services Layer - Severely Broken

#### services/content.py (text generation, translation)
```python
# Issues:
- build_reading_prompt() called with wrong parameters
- chat_complete_with_raw() returns tuple, not awaitable
- split_sentences() parameter mismatch
- build_translation_contexts() doesn't exist
```

#### services/learning.py (SRS, level management)
```python
# Issues:
- get_ci_target() function exists but may not be imported correctly
- update_level_from_text() references non-existent imports
```

### 3. Missing Backend Infrastructure

**Completely missing (deleted in refactor but still needed):**
- ❌ Background worker for text generation
- ❌ Text pool management (real text generation)
- ❌ SSE (Server-Sent Events) for real-time updates
- ❌ Startup text pre-generation
- ❌ Rate limiting middleware (`install_rate_limit`)
- ❌ Word selection service for SRS
- ❌ Session processing for reading progress

### 4. Frontend Issues

- ⚠️ app.js still references deleted SSE endpoints
- ⚠️ Complex reading logic in app.js (836 lines) depends on broken services

## Recommended Next Steps

### Phase 3: Simplify & Consolidate (Future)

1. **Clean up app.js:**
   - Remove SSE-related code
   - Simplify word interaction tracking
   - Focus on basic text display

2. **Rebuild services layer (optional):**
   - Only if text generation is needed
   - Consider alternative: pre-generated texts only
   - Or use external API for LLM calls

3. **Database & Models Cleanup:**
   - Standardize to one style (`Mapped` or `Column`)
   - Add migration support

## How to Test Current State

1. Visit http://localhost:8000
2. Sign up for a new account
3. Create a profile (Spanish → English)
4. Go to reading (shows demo text)
5. Toggle translation
6. Visit words/stats pages
