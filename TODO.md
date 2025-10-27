Tier-in-JWT rollout

- Goal: include the user's subscription tier in JWTs to avoid DB reads during rate limiting and feature gating.

Plan
- Auth issuance
  - Add `tier` claim to access tokens in `arcadia_auth` at login/refresh using the Account.subscription_tier string (default "Free").
  - Add short-lived access tokens only; avoid embedding tier in refresh tokens.
  - Ensure claim uses canonical names: Free, Standard, Pro, Pro+, BYOK, admin.
- Token consumption
  - Update rate limiter to read `tier` from JWT when present and fall back to "Free" if missing/unknown.
  - Keep DB fallback in `get_current_account` for endpoints that need user data; do not rely on token for persistent state.
- Tier changes
  - After `/me/tier` change, rotate access token so the `tier` claim updates immediately.
  - Optionally, add a minimal `/auth/refresh` convenience call in UI after tier change.
- Backwards compatibility
  - Accept tokens without `tier`; treat as "Free".
  - Keep legacy lowercase rate-limit keys in config as fallback mapping.
- Security
  - Treat `tier` as informational; server must still authorize sensitive routes via DB lookup when required (e.g., admin-only ops).
  - Validate signature and exp as usual; do not trust unsigned data.
- Testing
  - Unit: token creation contains `tier`; parsing in middleware selects correct bucket.
  - Integration: change tier → refresh → verify new limits apply.

Notes
- We currently assign default tier in `get_current_account` if missing to cover new signups.
- When JWT `tier` is live, we can simplify the rate limiter’s default path.
