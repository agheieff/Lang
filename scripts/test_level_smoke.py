from __future__ import annotations

import math
from datetime import datetime, timedelta

from Lang.server.db import init_db, SessionLocal
from Lang.server.models import User, Profile, Lexeme, LexemeInfo, WordEvent
from Lang.server.level import (
    _gaussian_kernel_weights,
    _decay_factor,
    update_level_for_profile,
    get_level_summary,
)


def approx(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(a - b) <= eps


def test_kernel_normalization() -> None:
    for c in [1.0, 2.5, 6.0]:
        w = _gaussian_kernel_weights(c, 1.0, bins=6)
        s = sum(w)
        assert approx(s, 1.0, 1e-6), f"kernel not normalized: {s}"


def test_decay_curve() -> None:
    d0 = _decay_factor(0.0)
    d_half = _decay_factor(45.0)  # default HL_DAYS in env
    assert 0.99 <= d0 <= 1.01
    assert 0.45 <= d_half <= 0.55


def test_end_to_end_level() -> None:
    init_db()
    s = SessionLocal()
    try:
        u = s.query(User).filter(User.email == 'pytest@example.com').first()
        if u:
            s.query(WordEvent).filter(WordEvent.user_id == u.id).delete()
            s.query(Profile).filter(Profile.user_id == u.id).delete()
            s.delete(u)
            s.commit()
        u = User(email='pytest@example.com', password_hash='x'); s.add(u); s.commit(); s.refresh(u)
        p = Profile(user_id=u.id, lang='zh'); s.add(p); s.commit(); s.refresh(p)
        # lexemes
        def mk(lemma, lvl, freq):
            L = s.query(Lexeme).filter(Lexeme.lang=='zh', Lexeme.lemma==lemma).first()
            if not L:
                L = Lexeme(lang='zh', lemma=lemma)
                s.add(L); s.commit(); s.refresh(L)
            LI = s.query(LexemeInfo).filter(LexemeInfo.lexeme_id==L.id).first()
            if not LI:
                LI = LexemeInfo(lexeme_id=L.id, level_code=lvl, freq_rank=freq)
                s.add(LI); s.commit()
            return L
        a = mk('你好', 'HSK1', 100)
        b = mk('苹果', 'HSK1', 200)
        c = mk('学习', 'HSK2', 800)
        d = mk('然而', 'HSK4', 2500)
        now = datetime.utcnow()
        def add(ev_type, lx, days_ago=0):
            s.add(WordEvent(ts=now - timedelta(days=days_ago), user_id=u.id, profile_id=p.id, lexeme_id=lx.id, event_type=ev_type, count=1))
        # successes at low bins, some failures at higher bin
        for i in range(8):
            add('exposure', a, days_ago=10-i)
            add('nonlookup', a, days_ago=5)
        for i in range(5):
            add('exposure', c, days_ago=5-i)
        for i in range(2):
            add('click', d, days_ago=i)
        s.commit()
        update_level_for_profile(s, p, force=True)
        s.commit(); s.refresh(p)
        summ = get_level_summary(s, p)
        lvl = summ['level_value']
        mu = summ['bins']['mu']
        assert lvl > 1.0, f"expected level > 1.0, got {lvl}"
        assert mu[0] >= mu[3], f"low-bin mean should be >= higher-bin mean: {mu}"
    finally:
        s.close()


if __name__ == "__main__":
    test_kernel_normalization()
    test_decay_curve()
    test_end_to_end_level()
    print("OK")
