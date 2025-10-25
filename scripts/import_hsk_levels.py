#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
from typing import List
from urllib.request import urlopen, Request

from server.db import init_db, SessionLocal
from server.models import Lexeme, LexemeInfo, LexemeVariant


RAW_URL = "https://raw.githubusercontent.com/ivankra/hsk30/master/hsk30.csv"


def fetch_csv(url: str) -> str:
    req = Request(url, headers={"User-Agent": "Arcadia-Lang/0.1 hsk-import"})
    with urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="ignore")


def split_variants(s: str) -> List[str]:
    if not s:
        return []
    # split on ASCII and full-width bar
    parts = []
    for p in s.replace("｜", "|").split("|"):
        p = p.strip()
        if p:
            parts.append(p)
    return parts


def upsert_level(db, lang: str, lemma: str, level_code: str):
    from opencc import OpenCC  # type: ignore
    hans = OpenCC("t2s").convert(lemma)
    lex = db.query(Lexeme).filter(Lexeme.lang == "zh", Lexeme.lemma == hans, Lexeme.pos == None).first()  # noqa: E711
    if not lex:
        lex = Lexeme(lang="zh", lemma=hans, pos=None)
        db.add(lex)
        db.flush()
    script = "Hant" if lang.endswith("Hant") else "Hans"
    if not db.query(LexemeVariant).filter(LexemeVariant.lexeme_id == lex.id, LexemeVariant.script == script, LexemeVariant.form == lemma).first():
        db.add(LexemeVariant(lexeme_id=lex.id, script=script, form=lemma))
        db.flush()
    li = db.query(LexemeInfo).filter(LexemeInfo.lexeme_id == lex.id).first()
    if not li:
        li = LexemeInfo(lexeme_id=lex.id)
        db.add(li)
        db.flush()
    li.level_code = level_code
    if not li.source:
        li.source = "hsk30"


def wipe_existing_levels(db, langs: List[str]):
    # unified zh only
    lex_ids = [row[0] for row in db.query(Lexeme.id).filter(Lexeme.lang == "zh").all()]
    if not lex_ids:
        return
    infos = db.query(LexemeInfo).filter(LexemeInfo.lexeme_id.in_(lex_ids)).all()
    for li in infos:
        li.level_code = None
    db.commit()


def main():
    ap = argparse.ArgumentParser(description="Import HSK 3.0 levels into LexemeInfo (full refresh)")
    ap.add_argument("--langs", nargs="+", default=["zh", "zh-Hans", "zh-Hant"], help="Langs to apply levels to")
    ap.add_argument("--url", default=RAW_URL, help="CSV URL (default: hsk30 master)")
    args = ap.parse_args()

    init_db()
    db = SessionLocal()
    try:
        print(f"Fetching {args.url} ...")
        text = fetch_csv(args.url)
        reader = csv.DictReader(io.StringIO(text))
        wipe_existing_levels(db, args.langs)
        count = 0
        for row in reader:
            simp = row.get("Simplified", "").strip()
            trad = row.get("Traditional", "").strip()
            level_raw = row.get("Level", "").strip()
            if not level_raw:
                continue
            code = f"HSK{level_raw}"
            # Simplified to zh/zh-Hans
            for l in split_variants(simp):
                if "zh" in args.langs:
                    upsert_level(db, "zh", l, code)
                if "zh-Hans" in args.langs:
                    upsert_level(db, "zh-Hans", l, code)
            # Traditional to zh-Hant
            for l in split_variants(trad):
                if "zh-Hant" in args.langs:
                    upsert_level(db, "zh-Hant", l, code)
            count += 1
            if count % 1000 == 0:
                db.commit()
                print(f"Committed {count} rows...")
        db.commit()
        print(f"Done. Processed {count} CSV rows.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
