# Word List Fix - Status

## What Was Done

### 1. SSE Unification (Complete)
- Removed orphaned `/reading/next/ready/sse` endpoint from `reading.py`
- Added `send_next_ready()` method to `NotificationService`
- Added `_notify_next_ready_if_backup()` to `GenerationOrchestrator`
- Added `next_ready` event handlers to client JS (`reading-sse.js`, `reading-controller.js`)
- Removed polling fallback `_startEmptyPoll()` from `reading-controller.js`
- Added initial backup text check on SSE connection

### 2. Client-Server Session Data Format Fix (Partial)
- **Done**: Updated `reading.js` with `extractWordEvents()` function that transforms nested session data into server-expected format:
  - Extracts `lookups` array from `paragraphs[].sentences[].words[]` where `looked_up_at` is set
  - Extracts `interactions` array with `event_type: 'click'` for lookups and `event_type: 'exposure'` for all words
  - Adds these to `sessionData` before sending to server

### 3. User-Specific Lexemes Fix (Partial)
- **Done**: Updated `lexeme_service.py`'s `resolve_lexeme()` to accept `account_id` and `profile_id` parameters
- **Done**: Updated `srs_service.py`'s `srs_click()`, `srs_exposure()`, `srs_nonlookup()` to pass user IDs
- **Done**: Updated `session_processing_service.py` to pass user IDs to `_resolve_lexeme()`

## What's Left

### 1. Fix remaining `_resolve_lexeme` call in `routes/srs.py:184`
Need to get profile and pass `account_id` and `profile_id`:
```python
# Around line 183 in routes/srs.py
# Current:
lex = _resolve_lexeme(db, req.lang, lemma, pos)
# Should be:
prof = db.query(Profile).filter(Profile.account_id == account.id, Profile.lang == req.lang).first()
lex = _resolve_lexeme(db, req.lang, lemma, pos, account_id=account.id, profile_id=prof.id if prof else 0)
```

### 2. Run tests to verify everything works
```bash
uv run pytest tests/ -v
```

### 3. Test the full flow manually
1. Open reading page
2. Click on some words
3. Click "Next" button
4. Verify in server logs that session data is processed
5. Check database that lexemes are created with correct `account_id` and `profile_id`

## Files Modified
- `server/routes/reading.py` - Removed SSE endpoint, cleaned imports
- `server/services/notification_service.py` - Added `send_next_ready()`
- `server/services/generation_orchestrator.py` - Added `_notify_next_ready_if_backup()`
- `server/static/reading-sse.js` - Added `next_ready` handler
- `server/static/reading-controller.js` - Added `next_ready` handler, removed polling
- `server/static/reading.js` - Added `extractWordEvents()` for format transformation
- `server/services/lexeme_service.py` - Updated `resolve_lexeme()` signature
- `server/services/srs_service.py` - Updated calls to pass user IDs
- `server/services/session_processing_service.py` - Updated calls to pass user IDs
- `server/models.py` - Fixed duplicate `completed_at` column (earlier cleanup)
- `server/services/translation_service.py` - Fixed `user_tier` scope issue (earlier cleanup)
