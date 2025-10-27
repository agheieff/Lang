from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from ..models import Profile, ProfilePref, WordEvent, Lexeme
from .math.decay import decay_factor
from .math.kernel import gaussian_kernel_weights
from .math.bayesian import compute_level_from_counts


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


# Public constants (imported by server.level wrappers)
PRIOR_A = _f("ARC_LEVEL_PRIOR_A", 1.0)
PRIOR_B = _f("ARC_LEVEL_PRIOR_B", 1.0)
P_TARGET = _f("ARC_LEVEL_P_TARGET", 0.75)
HL_DAYS = _f("ARC_LEVEL_HL_DAYS", 45.0)
KERNEL_SIGMA = _f("ARC_LEVEL_KERNEL_SIGMA", 1.0)
SUCCESS_SHIFT = _f("ARC_LEVEL_SUCCESS_SHIFT", 0.5)
FAIL_SHIFT = _f("ARC_LEVEL_FAIL_SHIFT", 0.5)
W_NONLOOKUP = _f("ARC_LEVEL_W_NONLOOKUP", 1.5)
W_EXPOSURE = _f("ARC_LEVEL_W_EXPOSURE", 0.1)
W_CLICK = _f("ARC_LEVEL_W_CLICK", 1.0)
ETA0 = _f("ARC_LEVEL_ETA0", 1.0)
E0 = _f("ARC_LEVEL_E0", 200.0)
ETA_MIN = _f("ARC_LEVEL_ETA_MIN", 0.1)
ETA_MAX = _f("ARC_LEVEL_ETA_MAX", 1.0)
IDLE_DAYS = _f("ARC_LEVEL_IDLE_DAYS", 7.0)
IDLE_BUMP_MAX = _f("ARC_LEVEL_IDLE_BUMP_MAX", 2.0)
UPDATE_MIN_INTERVAL_SEC = int(_f("ARC_LEVEL_UPDATE_MIN_INTERVAL_SEC", 60.0))
STALE_SEC = int(_f("ARC_LEVEL_STALE_SEC", 300.0))

_FREQ_EDGES = os.getenv("ARC_LEVEL_FREQ_EDGES", "1000,3000,10000,30000,100000").strip()
try:
    FREQ_EDGES = [int(x) for x in _FREQ_EDGES.split(",") if x.strip()]
    if len(FREQ_EDGES) != 5:
        FREQ_EDGES = [1000, 3000, 10000, 30000, 100000]
except Exception:
    FREQ_EDGES = [1000, 3000, 10000, 30000, 100000]


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def get_ci_target(db: Session, account_id: int, lang: str) -> float:
    try:
        default = float(os.getenv("ARC_CI_DEFAULT", "0.95"))
    except Exception:
        default = 0.95
    ci_min = float(os.getenv("ARC_CI_MIN", "0.8"))
    ci_max = float(os.getenv("ARC_CI_MAX", "0.99"))
    prof = db.query(Profile).filter(Profile.account_id == account_id, Profile.lang == lang).first()
    if not prof:
        return _clamp(default, ci_min, ci_max)
    pref = db.query(ProfilePref).filter(ProfilePref.profile_id == prof.id).first()
    if not pref or not isinstance(pref.data, dict):
        return _clamp(default, ci_min, ci_max)
    ci = pref.data.get("ci_target")
    if ci is None:
        st = pref.data.get("settings") if isinstance(pref.data.get("settings"), dict) else None
        if st and isinstance(st.get("ci_target"), (int, float)):
            ci = st.get("ci_target")
    try:
        return _clamp(float(ci) if ci is not None else default, ci_min, ci_max)
    except Exception:
        return _clamp(default, ci_min, ci_max)


def _ensure_profile_pref(db: Session, profile_id: int) -> ProfilePref:
    pref = db.query(ProfilePref).filter(ProfilePref.profile_id == profile_id).first()
    if not pref:
        pref = ProfilePref(profile_id=profile_id, data={})
        db.add(pref)
        db.flush()
    if not isinstance(pref.data, dict):
        pref.data = {}
    return pref


def _state_for_profile(pref: ProfilePref) -> Dict:
    data = pref.data or {}
    st = data.get("level_estimator")
    if not isinstance(st, dict):
        st = {}
    st.setdefault("version", 1)
    st.setdefault("last_update_at", None)
    st.setdefault("last_activity_at", None)
    st.setdefault("last_event_id", None)
    bins = st.get("bins") or {}
    a = bins.get("a") or [0.0] * 6
    b = bins.get("b") or [0.0] * 6
    if len(a) != 6:
        a = [0.0] * 6
    if len(b) != 6:
        b = [0.0] * 6
    st["bins"] = {"a": a, "b": b}
    st.setdefault("ess", sum(a) + sum(b))
    pref.data["level_estimator"] = st
    return st


def _save_state(db: Session, pref: ProfilePref, st: Dict, level_value: float, level_var: float) -> None:
    st["ess"] = float(sum(st["bins"]["a"]) + sum(st["bins"]["b"]))
    st.setdefault("last_profile", {})
    st["last_profile"].update(
        {
            "level_value": float(max(0.0, min(6.0, level_value))),
            "level_var": float(max(0.0, level_var)),
            "computed_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
    )
    pref.data["level_estimator"] = st
    db.flush()


def _bin_from_level_code(lang: str, code: Optional[str]) -> Optional[int]:
    if not code:
        return None
    c = code.strip().upper()
    if lang.startswith("zh"):
        if c.startswith("HSK"):
            try:
                n = int(c[3:])
                if 1 <= n <= 6:
                    return n
            except Exception:
                return None
        return None
    m = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
    return m.get(c)


def _bin_from_freq_rank(rank: Optional[int]) -> Optional[int]:
    if not rank or rank <= 0:
        return None
    edges = FREQ_EDGES
    if rank <= edges[0]:
        return 1
    if rank <= edges[1]:
        return 2
    if rank <= edges[2]:
        return 3
    if rank <= edges[3]:
        return 4
    if rank <= edges[4]:
        return 5
    return 6


def _resolve_bin_for_lexeme(db: Session, lexeme_id: int, lang: str) -> Optional[int]:
    lx = db.query(Lexeme).filter(Lexeme.id == lexeme_id).first()
    if lx:
        # Use level_code from Lexeme directly
        b = _bin_from_level_code(lang, lx.level_code)
        if b:
            return b
        # Use frequency_rank from Lexeme directly
        b2 = _bin_from_freq_rank(lx.frequency_rank)
        if b2:
            return b2
        # Fallback for Chinese based on character length
        if lang.startswith("zh") and lx.lemma:
            L = len(lx.lemma)
            return 1 if L <= 2 else 3 if L <= 4 else 5
    return None


def update_level_for_profile(db: Session, profile: Profile, *, force: bool = False) -> None:
    pref = _ensure_profile_pref(db, profile.id)
    st = _state_for_profile(pref)
    now = datetime.utcnow()
    if not force and st.get("last_update_at"):
        try:
            last_up = datetime.fromisoformat(st["last_update_at"])  # type: ignore
            if (now - last_up).total_seconds() < UPDATE_MIN_INTERVAL_SEC:
                return
        except Exception:
            pass
    a = list(st["bins"]["a"])
    b = list(st["bins"]["b"])
    if st.get("last_update_at"):
        try:
            last_up = datetime.fromisoformat(st["last_update_at"])  # type: ignore
            k = decay_factor((now - last_up).total_seconds() / 86400.0, HL_DAYS)
            a = [ai * k for ai in a]
            b = [bi * k for bi in b]
        except Exception:
            pass

    last_eid = st.get("last_event_id")
    q = db.query(WordEvent).filter(WordEvent.profile_id == profile.id)
    if last_eid:
        q = q.filter(WordEvent.id > int(last_eid))
    q = q.order_by(WordEvent.id.asc())
    events = q.limit(50000).all()
    if not events and not force:
        st["bins"]["a"], st["bins"]["b"] = a, b
        st["last_update_at"] = now.isoformat(timespec="seconds")
        lvl, lvar = compute_level_from_counts(a, b, PRIOR_A, PRIOR_B, P_TARGET)
        profile.level_value = float(lvl)
        profile.level_var = float(lvar)
        _save_state(db, pref, st, lvl, lvar)
        return

    S = [0.0] * 6
    F = [0.0] * 6
    last_ts: Optional[datetime] = None
    last_id: Optional[int] = last_eid
    for ev in events:
        last_id = ev.id
        last_ts = max(last_ts or ev.ts, ev.ts)
        et = (ev.event_type or "").lower()
        if et == "nonlookup":
            succ = True; w0 = W_NONLOOKUP
        elif et == "exposure":
            succ = True; w0 = W_EXPOSURE
        elif et == "click":
            succ = False; w0 = W_CLICK
        else:
            continue
        b0 = _resolve_bin_for_lexeme(db, ev.lexeme_id, profile.lang)
        if not b0:
            continue
        try:
            dt_days = (now - ev.ts).total_seconds() / 86400.0
        except Exception:
            dt_days = 0.0
        w_t = decay_factor(dt_days, HL_DAYS)
        w = max(0.0, w0 * w_t * max(1, ev.count or 1))
        center = float(b0 + (SUCCESS_SHIFT if succ else -FAIL_SHIFT))
        center = max(1.0, min(6.0, center))
        Kj = gaussian_kernel_weights(center, KERNEL_SIGMA, bins=6)
        for j in range(6):
            if succ:
                S[j] += w * Kj[j]
            else:
                F[j] += w * Kj[j]

    ess = float(sum(a) + sum(b))
    lr = max(ETA_MIN, min(ETA_MAX, ETA0 * (E0 / (E0 + ess))))
    if st.get("last_activity_at"):
        try:
            gap_days = (now - datetime.fromisoformat(st["last_activity_at"])) .total_seconds() / 86400.0  # type: ignore
            if gap_days > IDLE_DAYS:
                bump = min(IDLE_BUMP_MAX, 1.0 + math.log1p((gap_days - IDLE_DAYS) / max(1e-6, IDLE_DAYS)))
                lr = min(ETA_MAX, lr * bump)
        except Exception:
            pass

    for j in range(6):
        a[j] += lr * S[j]
        b[j] += lr * F[j]

    st["bins"]["a"], st["bins"]["b"] = a, b
    st["last_update_at"] = now.isoformat(timespec="seconds")
    if last_id:
        st["last_event_id"] = int(last_id)
    if last_ts:
        st["last_activity_at"] = last_ts.isoformat(timespec="seconds")

    lvl, lvar = compute_level_from_counts(a, b, PRIOR_A, PRIOR_B, P_TARGET)
    profile.level_value = float(lvl)
    profile.level_var = float(lvar)
    _save_state(db, pref, st, lvl, lvar)


def update_level_if_stale(db: Session, user_id: int, lang: str, *, force: bool = False) -> None:
    prof = db.query(Profile).filter(Profile.account_id == user_id, Profile.lang == lang).first()
    if not prof:
        return
    pref = _ensure_profile_pref(db, prof.id)
    st = _state_for_profile(pref)
    now = datetime.utcnow()
    if force:
        update_level_for_profile(db, prof, force=True)
        return
    last = st.get("last_update_at")
    if not last:
        update_level_for_profile(db, prof, force=True)
        return
    try:
        last_dt = datetime.fromisoformat(last)  # type: ignore
        if (now - last_dt).total_seconds() >= STALE_SEC:
            update_level_for_profile(db, prof, force=False)
    except Exception:
        update_level_for_profile(db, prof, force=True)


def get_level_summary(db: Session, profile: Profile) -> Dict:
    pref = _ensure_profile_pref(db, profile.id)
    st = _state_for_profile(pref)
    a = list(st["bins"]["a"])
    b = list(st["bins"]["b"])
    mu: List[float] = []
    vr: List[float] = []
    for j in range(6):
        alpha = PRIOR_A + max(0.0, a[j])
        beta = PRIOR_B + max(0.0, b[j])
        s = alpha + beta
        mu.append(alpha / s)
        vr.append((alpha * beta) / (s * s * (s + 1.0)))
    return {
        "level_value": float(getattr(profile, "level_value", 0.0) or 0.0),
        "level_var": float(getattr(profile, "level_var", 1.0) or 1.0),
        "last_update_at": st.get("last_update_at"),
        "last_activity_at": st.get("last_activity_at"),
        "ess": float(sum(a) + sum(b)),
        "bins": {
            "a": a,
            "b": b,
            "mu": mu,
            "var": vr,
        },
    }
