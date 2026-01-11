# Todo List

**Last Updated**: 2025-01-11

## Critical Fixes (Production Readiness)

### [x] Fix Database Transaction Safety ✅
**Priority**: CRITICAL
**Completed**: 2025-01-11
**Files**: `server/db.py`, `server/routes/reading.py`, `server/routes/reading_text_log.py`, `server/routes/user.py`

Implemented transaction wrapper context manager with automatic rollback on errors.

```python
@contextmanager
def db_transaction(db: Session):
    """Context manager for safe database transactions with automatic rollback on errors."""
    try:
        yield
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Transaction failed, rolled back: {e}", exc_info=True)
        raise
```

Applied to:
- `server/routes/reading.py:reading_next()` - wrapped text selection in transaction
- `server/routes/reading_text_log.py:log_text_state()` - wrapped SRS updates in transaction
- `server/routes/user.py:post_me_profile()` - wrapped profile updates in transaction

---

### [x] Strengthen JWT Token Validation ✅
**Priority**: CRITICAL
**Completed**: 2025-01-11
**File**: `server/deps.py`

Fixed authentication bypass by adding:
- ✅ Minimum token length validation (20+ characters)
- ✅ Invalid character detection (\n, \r, \0)
- ✅ Strict JWT algorithm checking (HS256)
- ✅ Expiration validation (defense in depth)

---

### [x] Fix Background Worker Race Condition ✅
**Priority**: CRITICAL
**Completed**: 2025-01-11
**File**: `server/services/background_worker.py`

Implemented in-memory distributed lock with:
- ✅ `_acquire_generation_lock()` - lock acquisition with timeout
- ✅ `_release_generation_lock()` - lock release in finally block
- ✅ `_get_locked_profile_ids()` - get currently locked profiles
- ✅ Lock filtering in `fill_gaps()` to skip locked profiles
- ✅ Automatic lock expiration (10 minute timeout)

---

### [x] Fix Rate Limiter Memory Leak ✅
**Priority**: CRITICAL
**Completed**: 2025-01-11
**File**: `server/middleware.py`

Implemented automatic cleanup of idle rate limiter buckets:
- ✅ Track last access time per bucket (`last_access` field)
- ✅ Periodic cleanup (every 100 requests)
- ✅ Stale bucket timeout (1 hour)
- ✅ Maximum bucket limit (10,000)
- ✅ OrderedDict for LRU eviction
- ✅ `is_stale()` method for timeout detection

---

### [x] Fix Timezone-Aware Datetime Comparisons ✅
**Priority**: HIGH
**Completed**: 2025-01-11
**Files**: `server/services/recommendation.py`, `server/services/srs.py`

Fixed timezone-aware datetime comparisons:
- ✅ Added `_ensure_timezone_aware()` helper function
- ✅ Fixed `recommendation.py:435` - use naive UTC for SQLite compatibility
- ✅ `srs.py` already handles timezone conversion properly

---

### [ ] Add CSRF Protection
**Priority**: HIGH
**Files**: All POST/PUT/DELETE endpoints

Implement CSRF tokens with validation on state-changing operations.

---

### [ ] Add Missing Input Validation
**Priority**: HIGH
**Files**: Multiple route files

Validate all path parameters, enum values, and ranges.

---

### [x] Improve SRS Batch Update Error Handling ✅
**Priority**: HIGH
**Completed**: 2025-01-11
**File**: `server/services/srs.py`

Wrapped individual lexeme updates in try/except:
- ✅ Individual error handling for each lexeme update
- ✅ Track failures with detailed error info
- ✅ Atomic commit of successful updates only
- ✅ Rollback on commit failure
- ✅ Return summary with failed count and error details

---

## Enhancement Tasks (Backlog)

### Text Generation & Selection
- [ ] Monitor LLM target word inclusion rates
- [ ] Add CI adjustment based on target word difficulty
- [ ] Implement adaptive target word count based on text length

### Performance Optimization
- [ ] Add database indexes for frequently queried fields
- [ ] Implement vocabulary overlap caching
- [ ] Optimize N+1 queries in text selection

### Monitoring & Observability
- [ ] Add structured logging for production
- [ ] Implement health check endpoints
- [ ] Add metrics for SRS effectiveness
- [ ] Set up error tracking (Sentry integration)

---

## Completed

### Phase 1: SRS Integration (2025-01-11)
- ✅ Implemented urgency score calculation (time + familiarity)
- ✅ Added vocabulary overlap scoring for text selection
- ✅ Enhanced text selection with urgency factor (weight 2.5)
- ✅ Implemented POS-diversified target word selection
- ✅ Added target words to text generation pipeline
- ✅ Fixed JSON serialization bug (prompt_words set→list)

### Phase 2: UI Improvements (2025-01-11)
- ✅ Cleaned up variable naming (p→clicks, n→exposures)
- ✅ Fixed Click % column to show actual probability
- ✅ Added color coding for familiarity (red/yellow/green)
- ✅ Muted variance columns (fam_var, decay_var) in gray
