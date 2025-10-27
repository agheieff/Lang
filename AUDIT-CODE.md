# Arcadia Lang — Concise Code Audit (Top 5)

Scope: FastAPI app under `server/*`; focus on architecture cohesion, modularity, dead code/duplication, boundaries (routes/services/repos), error handling, config/env, performance/security. Ordered by impact.

## 1) Rate limiting effectively disabled
- Why it matters: All endpoints are currently unthrottled, exposing the app to abuse, cost spikes, and denial-of-service.
- Concrete next step: Remove the early `return` in `rate_limit()` and enforce per-tier/IP limits using `ARC_RATE_*` from `server/config.py`; keep `install_rate_limit(app)` after router registration.
- Impact: High
- Effort: S
- References: `server/middleware/rate_limit.py:19–23` (early return), `server/main.py:84` (middleware installed), `server/config.py` (RATE_LIMITS, RATE_WINDOW_SEC).

## 2) Security defaults misconfigured (CORS + JWT secret)
- Why it matters: `allow_origins=["*"]` together with `allow_credentials=True` allows cross-site cookie use, and a hardcoded default JWT secret risks token forgery in non-dev environments.
- Concrete next step: Restrict CORS to an environment-driven whitelist and set `allow_credentials=True` only when origins are explicit; fail-fast at startup if `ARC_LANG_JWT_SECRET` is the default value.
- Impact: High
- Effort: S
- References: `server/main.py:33` (CORS `"*"` with credentials), `server/deps.py:13` (default secret), `server/middleware/auth.py` (secret sourcing).

## 3) Settings page update wiring is broken (route path + params)
- Why it matters: Profile updates from Settings do not reach the API due to a wrong path and parameter encoding, resulting in a non-functional UX.
- Concrete next step: In `settings.html`, change `hx-put` to `/me/profile?lang={{ current_profile.lang }}&target_lang={{ current_profile.target_lang }}` (or add a hidden form + `hx-include`), and keep JSON body for fields; alternatively, update the API to accept `lang`/`target_lang` in the request body.
- Impact: High
- Effort: S
- References: `server/templates/pages/settings.html:23,49` (uses `/api/me/profile`), `server/api/profile.py:26` (no router prefix), `server/api/profile.py:174` (`@router.put("/me/profile")`).

## 4) DB session lifecycle leaks in UI routes
- Why it matters: `next(get_db(request))` creates a Session that is never closed, leading to connection/resource leaks over time.
- Concrete next step: Inject `db: Session = Depends(get_db)` into `settings_page()` and `home_page()` and remove manual generator usage so FastAPI handles teardown.
- Impact: Medium–High
- Effort: S
- References: `server/routes/ui.py:74,103` (manual `next(get_db(request))`).

## 5) Duplicate function definitions in LLM module
- Why it matters: Maintaining two identical `build_word_translation_prompt()` implementations risks divergence and confuses readers.
- Concrete next step: Keep one definition, delete the duplicate, and ensure all imports reference the single source.
- Impact: Medium
- Effort: S
- References: `server/llm.py:222` and `server/llm.py:649` (duplicate definitions).

---

### Notes on boundaries and modularity (observed, not top-5):
- Routes sometimes mix concerns (e.g., `routes/reading.py:119+` returns HTML snippets from a data API). Consider standardizing: JSON from API routes; HTML via template/UI routes.
- Per-account DB table list drift: `server/account_db.py` includes legacy names (`lexeme_info`, `user_lexemes`, `language_word_lists`) no longer present in `server/models.py`; prune to reduce confusion and prevent accidental reliance on removed tables.
- Performance consideration: `llm.urgent_words_detailed()` loads all user lexemes and scores in Python; for large datasets, consider pushing filtering/bucketing to SQL or limiting candidates via indexed queries.

---

## Modified files: potential incomplete work
- `server/routes/ui.py`
  - DB session leak as noted above (`:74,103`).
  - Home page generation calls LLM with `provider="openrouter"` but `base_url="http://localhost:1234/v1"` (LM Studio style); ensure provider/base_url pairing is intentional.
- `server/templates/pages/settings.html`
  - Wrong endpoint path (`/api/me/profile`) and missing query params for `lang`/`target_lang` cause 404/422 on save; align with `PUT /me/profile` and pass required params via query or adjust the API to read them from body.

---

Prepared by: Audit subagent (Factory)
