#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
import argparse
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from typing import List

from bs4 import BeautifulSoup  # type: ignore

from server.db import init_db, SessionLocal
from server.models import Lexeme, LexemeInfo


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


def parse_simplified_list(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    # Find first wikitable with headers Traditional | Simplified | Pinyin | Meaning
    tables = soup.find_all("table", class_="wikitable")
    simplified: List[str] = []
    seen: set[str] = set()
    for tb in tables:
        # check header
        ths = [th.get_text(strip=True) for th in tb.find_all("th")]
        if len(ths) >= 4 and ths[0].startswith("Traditional") and ths[1].startswith("Simplified"):
            # iterate rows
            for tr in tb.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) >= 2:
                    simp = tds[1].get_text(strip=True)
                    if simp and simp not in seen:
                        simplified.append(simp)
                        seen.add(simp)
            break
    return simplified


def upsert_lexeme_info(db, lang_codes: List[str], lemma: str, rank: int):
    for lang in lang_codes:
        lex = db.query(Lexeme).filter(Lexeme.lang == lang, Lexeme.lemma == lemma, Lexeme.pos == None).first()  # noqa: E711
        if not lex:
            lex = Lexeme(lang=lang, lemma=lemma, pos=None)
            db.add(lex)
            db.flush()
        li = db.query(LexemeInfo).filter(LexemeInfo.lexeme_id == lex.id).first()
        if not li:
            li = LexemeInfo(lexeme_id=lex.id)
            db.add(li)
            db.flush()
        li.freq_rank = rank
        li.source = "wiktionary-ckip"


def main():
    ap = argparse.ArgumentParser(description="Import Mandarin frequency from Wiktionary (CKIP 10k)")
    ap.add_argument("--langs", nargs="+", default=["zh", "zh-Hans", "zh-Hant"], help="Language codes to attach (default: zh zh-Hans zh-Hant)")
    args = ap.parse_args()

    init_db()
    db = SessionLocal()
    try:
        rank = 0
        global_seen: set[str] = set()
        for seg in RANGES:
            url = BASE.format(seg)
            print(f"Fetching {url}...")
            try:
                html = fetch(url)
            except (URLError, HTTPError) as e:
                print(f"Failed to fetch {url}: {e}")
                continue
            words = parse_simplified_list(html)
            print(f"Parsed {len(words)} entries in {seg}")
            batch = 0
            for w in words:
                if w in global_seen:
                    continue
                global_seen.add(w)
                rank += 1
                upsert_lexeme_info(db, args.langs, w, rank)
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
