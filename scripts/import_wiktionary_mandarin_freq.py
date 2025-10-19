#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
import argparse
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from typing import List, Tuple

from bs4 import BeautifulSoup  # type: ignore

from server.db import init_db, SessionLocal
from server.models import Lexeme, LexemeInfo, LexemeVariant


BASE = "https://en.wiktionary.org/wiki/Appendix:Mandarin_Frequency_lists/{}"
RANGES = [
    "1-1000",
    "1001-2000",
    "2001-3000",
    "3001-4000",
    "4001-5000",
    "5001-6000",
    "6001-7000",
    "7001-8000",
    "8001-9000",
    "9001-10000",
]


def fetch(url: str) -> str:
    req = Request(url, headers={"User-Agent": "Arcadia-Lang/0.1 frequency-import"})
    with urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="ignore")


def parse_rows(html: str) -> List[Tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    # Find first wikitable with headers Traditional | Simplified | Pinyin | Meaning
    tables = soup.find_all("table", class_="wikitable")
    rows: List[Tuple[str, str]] = []
    for tb in tables:
        # check header
        ths = [th.get_text(strip=True) for th in tb.find_all("th")]
        if len(ths) >= 4 and ths[0].startswith("Traditional") and ths[1].startswith("Simplified"):
            # iterate rows
            for tr in tb.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) >= 2:
                    trad = tds[0].get_text(strip=True)
                    simp = tds[1].get_text(strip=True)
                    if trad and simp:
                        rows.append((trad, simp))
            break
    return rows


def upsert_one(db, lang: str, lemma: str, rank: int):
    # Resolve lexeme via variants under unified zh key
    # Normalize using OpenCC
    from opencc import OpenCC  # type: ignore
    hans = OpenCC("t2s").convert(lemma)
    lex = db.query(Lexeme).filter(Lexeme.lang == "zh", Lexeme.lemma == hans, Lexeme.pos == None).first()  # noqa: E711
    if not lex:
        lex = Lexeme(lang="zh", lemma=hans, pos=None)
        db.add(lex)
        db.flush()
    script = "Hant" if lang.endswith("Hant") else "Hans"
    exists = db.query(LexemeVariant).filter(LexemeVariant.lexeme_id == lex.id, LexemeVariant.script == script, LexemeVariant.form == lemma).first()
    if not exists:
        db.add(LexemeVariant(lexeme_id=lex.id, script=script, form=lemma))
        db.flush()
    li = db.query(LexemeInfo).filter(LexemeInfo.lexeme_id == lex.id).first()
    if not li:
        li = LexemeInfo(lexeme_id=lex.id)
        db.add(li)
        db.flush()
    # Overwrite rank each run (full refresh semantics)
    li.freq_rank = rank
    li.source = "wiktionary-ckip"


def wipe_existing_freq(db, langs: List[str]):
    # Set freq_rank and freq_score to NULL for all lexemes in the given languages
    lex_ids = [row[0] for row in db.query(Lexeme.id).filter(Lexeme.lang.in_(langs)).all()]
    if not lex_ids:
        return
    infos = db.query(LexemeInfo).filter(LexemeInfo.lexeme_id.in_(lex_ids)).all()
    for li in infos:
        li.freq_rank = None
        li.freq_score = None
        li.source = None
    db.commit()


def main():
    ap = argparse.ArgumentParser(description="Import Mandarin frequency from Wiktionary (CKIP 10k)")
    ap.add_argument("--langs", nargs="+", default=["zh", "zh-Hans", "zh-Hant"], help="Language codes to attach (default: zh zh-Hans zh-Hant)")
    args = ap.parse_args()

    init_db()
    db = SessionLocal()
    try:
        rank = 0
        # wipe freq for zh only (unified)
        wipe_existing_freq(db, ["zh"]) 
        for seg in RANGES:
            url = BASE.format(seg)
            print(f"Fetching {url}...")
            try:
                html = fetch(url)
            except (URLError, HTTPError) as e:
                print(f"Failed to fetch {url}: {e}")
                continue
            pairs = parse_rows(html)
            print(f"Parsed {len(pairs)} entries in {seg}")
            batch = 0
            for trad, simp in pairs:
                rank += 1
                # Map by language: zh-Hans/zh -> simplified; zh-Hant -> traditional
                upsert_one(db, "zh-Hans", simp, rank)
                upsert_one(db, "zh-Hant", trad, rank)
                batch += 1
                if batch % 500 == 0:
                    db.commit()
            db.commit()
            time.sleep(1)
        print(f"Done. Last rank: {rank}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
