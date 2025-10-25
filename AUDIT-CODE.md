# Backend Code & Database Audit

Scope: server/, Lang/, scripts/ (backend-only). Security intentionally excluded per brief. Primary focus: database logic and backend architecture for maintainability, performance, and correctness.

## Executive Summary

- server/main.py is a 2K+ line monolith mixing routing, business logic, and persistence concerns; extract modular routers (APIRouter) and service/repository layers to dramatically improve maintainability and testability.
- init_db() is invoked across many request handlers (e.g., server/main.py:703, 754, 765, …); move schema initialization and seeding to a FastAPI startup/lifespan hook to reduce per-request overhead and risk.
- Replace ad-hoc, best-effort migrations with lightweight versioned migrations (or Alembic). Current PRAGMA-based column adds will not scale and are hard to reason about.
- Leverage SQLAlchemy 2.0 upserts (ON CONFLICT) for idempotent writes (ReadingTextTranslation, ReadingLookup) instead of try/rollback loops; unify transaction boundaries to one commit per request.
- Consolidate duplicated DB access patterns (profiles, tiers, user lookup in middleware) behind small repository helpers and reuse request-scoped state to avoid repeated queries.
- Minor indexing and query-shaping opportunities exist; add a few composite indexes and push more filtering into SQL to reduce Python-side post-filtering.

---

## Prioritized Recommendations by Impact

### High Impact

1) Split server/main.py into routers + services + repositories

- Rationale:
  - server/main.py (~2,000+ LOC) combines HTTP routing, business logic (SRS, level estimation, reading/translation), and direct ORM queries. This tight coupling raises cognitive load, increases change risk, and complicates testing.
- Concrete steps:
  - Create submodules under server/:
    - server/api/srs.py (SRS endpoints + thin handlers)
    - server/api/reading.py (reading generation, read tracking)
    - server/api/translation.py (translate + retrieval)
    - server/api/profile.py (tiers, profiles, prefs, theme)
    - server/services/{srs.py, reading.py, translation.py, profile.py} (business logic)
    - server/repos/{users.py, profiles.py, lexemes.py, events.py, readings.py, lookups.py} (ORM calls)
  - Convert FastAPI routes to APIRouter per domain and include_routers in main.
  - Move helpers like _resolve_lexeme, _get_or_create_userlexeme into repos with focused unit tests.
- References:
  - Monolithic handlers and logic intermixed throughout server/main.py (e.g., SRS: ~lines 868–1515; translation: ~1528–1931).
- Expected payoff:
  - Maintainability and testability up; regression risk down; enables local refactors without touching unrelated endpoints.

2) Move DB initialization from per-request to startup (lifespan)

- Rationale:
  - init_db() is called in many endpoints (server/main.py lines: 98, 703, 754, 765, 784, 796, 806, 994, 1126, 1149, 1174, 1276, 1327, 1361, 1415, 1466, 1504, 1710, 1876, 1909, 1938, 1955). This introduces overhead every request and risks writing during read-only GETs.
- Concrete steps:
  - Use a FastAPI lifespan hook to run initialization once.
  - Also move optional seeding tasks (tiers, model catalog) here.
- Snippet:
  ```python
  # server/app.py or at top of main.py
  from contextlib import asynccontextmanager
  from fastapi import FastAPI
  from .db import init_db

  @asynccontextmanager
  async def lifespan(app: FastAPI):
      init_db()  # create tables/migrations once
      # optionally: seed default tiers/models here
      yield

  app = FastAPI(lifespan=lifespan, title="Arcadia Lang", version="0.1.0")
  ```
- Expected payoff:
  - Lower request latency; less lock contention on SQLite; clearer separation of startup side-effects vs request handling.

3) Adopt versioned migrations; retire ad-hoc PRAGMA-based alters

- Rationale:
  - server/db.py::_run_migrations() issues ALTER TABLEs opportunistically via PRAGMA table_info lookups. This becomes brittle as schema grows and offers no version tracking or downgrade path.
- Concrete steps:
  - Short-term: add a migrations/schema_version table; encode ordered migration steps and apply once per version with a simple runner.
  - Long-term: adopt Alembic and generate revisions; call upgrade() from startup.
- References:
  - server/db.py::_run_migrations()
- Expected payoff:
  - Safer schema evolution; easier rollouts and debugging; removes surprise writes in request code.

4) Use SQLAlchemy 2.0 upserts for idempotent writes (avoid rollback loops)

- Rationale:
  - ReadingTextTranslation writes catch IntegrityError and then rollback inside a per-row loop (server/main.py ~1739–1787). Similar patterns appear in ReadingLookup upserts (~294–345). This is wasteful and can rollback unrelated work in the same transaction.
- Concrete steps:
  - For SQLite, use INSERT ... ON CONFLICT DO UPDATE/NOTHING via SQLAlchemy 2.0’s on_conflict_do_*.
- Snippet (example for ReadingTextTranslation):
  ```python
  from sqlalchemy.dialects.sqlite import insert
  from .models import ReadingTextTranslation as RTT

  stmt = insert(RTT).values(
      user_id=user.id,
      text_id=payload.text_id,
      unit=unit,
      target_lang=(payload.target_lang or "en"),
      segment_index=(idx if unit != "text" else None),
      span_start=(base_offset + span[0]),
      span_end=(base_offset + span[1]),
      source_text=src,
      translated_text=tr,
      provider=(payload.provider or "openrouter"),
      model=payload.model,
  )
  stmt = stmt.on_conflict_do_update(
      index_elements=[RTT.user_id, RTT.text_id, RTT.target_lang, RTT.unit, RTT.segment_index, RTT.span_start, RTT.span_end],
      set_={"translated_text": stmt.excluded.translated_text, "source_text": stmt.excluded.source_text},
  )
  db.execute(stmt)
  ```
  - Apply the same to ReadingLookup (unique by user_id, text_id, target_lang, span_start, span_end).
- Expected payoff:
  - Eliminates noisy rollbacks; clearer intent; better throughput under contention.

5) Unify transaction boundaries (“unit of work per request”) and isolate logging

- Rationale:
  - db.commit()/rollback() occurs multiple times in handlers, sometimes just for logs (e.g., LLMRequestLog around ~1014–1076, ~1766–1787). Rollbacks for logging errors can inadvertently affect business writes.
- Concrete steps:
  - Structure handlers as: modify state -> commit once -> best-effort logging with a separate short-lived session (or after-commit hook). For very low-risk logging, consider fire-and-forget background task.
- References:
  - server/main.py commits: lines 639, 674, 738, 773, 812, 887, 919, 1038, 1072, 1119, 1168, 1296, 1346, 1375, 1508, 1780, 1822, 1863, 1949; rollbacks: 1040, 1074, 1782, 1824, 1847.
- Expected payoff:
  - Reduced transaction complexity and fewer cross-effects; cleaner error handling.

### Medium Impact

6) Consolidate profile and tier access; reuse request-scoped state

- Rationale:
  - Repeated patterns to fetch/create Profile and ProfilePref (_get_or_create_profile, _get_pref_row) and to ensure default tiers are scattered across endpoints. Rate-limit middleware performs a separate SessionLocal() to read user tier before handlers.
- Concrete steps:
  - Create small repos: profiles.get_or_create(user_id, lang), profiles.get_pref(profile_id), tiers.ensure_defaults().
  - In middleware, resolve user/tier once and stash to request.state; have _get_current_user prefer request.state when present to avoid re-querying.
- References:
  - server/main.py: _get_or_create_profile (~624–640), _get_pref_row (~642–654), _ensure_default_tiers (~566–587); rate limit middleware (~120–170) opens its own SessionLocal.
- Expected payoff:
  - Fewer queries; clearer ownership; smaller handlers.

7) Indexing: add composite index for frequent predicates

- Rationale:
  - Many queries filter UserLexeme by (user_id, profile_id); per-column indexes exist, but a composite index improves selectivity.
- Concrete steps:
  - Add CREATE INDEX IF NOT EXISTS idx_ul_user_profile ON user_lexemes(user_id, profile_id).
  - Consider idx_reading_texts_user_created ON reading_texts(user_id, created_at) if you add list endpoints.
- References:
  - UserLexeme filters appear in SRS and selection flows (server/main.py: ~1182–1218; 1276–1327; llm.py: pick_words/urgent_words_detailed).
- Expected payoff:
  - Better planner choices and fewer row scans on large datasets.

8) Push more filtering into SQL and define stable sort where paginating

- Rationale:
  - get_srs_words computes some filters in Python after fetching up to 1000 rows; though basic filters are pushed into SQL, a stable ORDER BY and full SQL-side filtering will reduce over-fetch.
- Concrete steps:
  - Add ORDER BY stability DESC, distinct_texts DESC (or relevant score) and apply all numeric bounds in SQL. Then LIMIT/OFFSET for pagination.
- References:
  - server/main.py: get_srs_words (~1218–1296).
- Expected payoff:
  - Predictable performance and easier client-side pagination.

9) Normalize configuration

- Rationale:
  - Many env-tunable constants are scattered in main.py and level.py. A small Config dataclass (per domain) improves discoverability.
- Concrete steps:
  - server/config.py with cohesive groups (SRS weights, CI target, level estimator). Inject into services instead of reading os.getenv in hot paths.
- Expected payoff:
  - Clearer configuration surface; simpler testing via explicit dependency injection.

10) Align model catalog naming and usage

- Rationale:
  - openrouter.seed_sqlite(... table="models") seeds a table named models, while the app defines LLMModel mapped to llm_models and never reads it. This is confusing and likely dead code.
- Concrete steps:
  - Either (a) use LLMModel consistently and seed into llm_models, or (b) drop LLMModel if not used.
- References:
  - server/main.py imports LLMModel but doesn’t use it; server/models.py: class LLMModel.
- Expected payoff:
  - Removes ambiguity and potential schema drift.

### Low Impact

11) Eliminate dead/placeholder models

- Rationale:
  - Card and LLMModel are currently unused.
- Concrete steps:
  - Remove Card and LLMModel until needed, or add feature usage soon. Update migrations accordingly.
- References:
  - server/models.py: class Card, class LLMModel. Grep shows no usage beyond import.
- Expected payoff:
  - Leaner schema and fewer distractions for new contributors.

12) Prefer SQLAlchemy 2.0 Core/ORM patterns consistently

- Rationale:
  - Codebase mixes Session.query (1.x style) with modern 2.0 semantics. Align on one style to reduce confusion.
- Concrete steps:
  - Use select(Model).where(...) patterns; keep relationship loading explicit.
- Expected payoff:
  - Clearer upgrade path and consistency across repos/services.

13) Reduce broad exception swallowing in non-critical paths

- Rationale:
  - Several places catch Exception broadly and continue (e.g., dictionary seeding, OpenCC, LLM seeding). While intentional for dev ergonomics, limit to well-known exceptions and log at debug to aid diagnostics.
- Concrete steps:
  - Narrow except clauses where possible; add debug logs.
- References:
  - server/main.py: many try/except blocks across helpers and endpoints; server/db.py pragma setup swallowing.
- Expected payoff:
  - Easier debugging without harming resilience.

14) Scripts polish (type hints and entry points)

- Rationale:
  - scripts/import_hsk_levels.py uses an undefined type name `string` in annotation (safe due to `from __future__ import annotations` but misleading).
- Concrete steps:
  - Change to `str`; add uv scripts entries to pyproject for discoverability (e.g., `uv run python -m scripts.import_hsk_levels`).
- References:
  - scripts/import_hsk_levels.py: def fetch_csv(url: string) -> str
- Expected payoff:
  - Minor clarity improvements for contributors.

---

## Quick Wins (< 1 hour)

- Move init_db() to a FastAPI lifespan handler; delete redundant calls in endpoints (server/main.py lines listed above).
- Add composite index: user_lexemes(user_id, profile_id) via a one-line migration.
- Replace IntegrityError try/rollback loop in translate with a single ON CONFLICT upsert for ReadingTextTranslation.
- Fix the `string` type annotation to `str` in scripts/import_hsk_levels.py.
- Remove unused Card and LLMModel models (or comment clearly as planned features) to reduce schema noise.

## Deeper Refactors (multi-file)

- Router/Service/Repo extraction from server/main.py into server/api, server/services, server/repos modules.
- Introduce a minimal migration system (schema_version table + ordered steps) or adopt Alembic.
- Transaction boundary normalization: one commit per request; separate session for logging and non-critical analytics writes.
- Configuration centralization: Config dataclasses injected into services (SRS/level/CI parameters).

---

## Anti-Patterns Observed and Replacements

- Monolithic route module with mixed concerns → Extract APIRouter modules + services + repos.
- Per-request schema initialization (init_db in handlers) → Startup/lifespan init.
- Rollback-heavy idempotency (try/except IntegrityError in loops) → SQL-level upserts (ON CONFLICT DO UPDATE/NOTHING).
- Broad except Exception with silent pass → Narrow exceptions with debug logs; avoid masking real failures.
- Multiple commits per handler including logging → Single commit for core state; logging in separate session after commit.

---

## Backend Dead Code / Duplications

- server/models.py: Card (unused).
- server/models.py: LLMModel appears unused; openrouter seeding writes to a different table name ("models"). Either wire it up or remove.

---

## Notes on DB Logic Strengths (for context)

- Good use of WAL and SQLite PRAGMAs to improve concurrency (server/db.py) and a simple index creation pass.
- Many bulk lookups are batched (e.g., lemma IN (...) and joins in level/llm paths) reducing N+1 risk.
- ReadingLookup and ReadingTextTranslation have appropriate uniqueness constraints to ensure idempotency.

---

## Reporting: Actions Taken

- Analyzed backend code in server/, Lang/, and scripts/; reviewed pyproject.toml, uv.toml, and run.sh for environment and execution context.
- Identified modularization, migration, transaction, and idempotency improvements; highlighted indexing and duplication cleanup.
- Produced this ordered Markdown report (AUDIT-CODE.md) with concrete steps and references.

## Blockers / Follow-ups

- Choosing between lightweight internal migrations vs Alembic requires maintainer preference. A minimal schema_version approach can be implemented quickly; Alembic is more scalable.
- Router/service/repo organization may align with future feature growth; a short design session to agree on module boundaries will smooth adoption.
