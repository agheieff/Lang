#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from server.account_db import open_account_session
from server.utils.gloss import reconstruct_glosses_from_logs


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill ReadingWordGloss rows from words_*.json logs")
    p.add_argument("--account", type=int, required=True, help="Account ID")
    p.add_argument("--text", type=int, required=True, help="Reading text ID")
    p.add_argument("--lang", type=str, required=True, help="Language code (e.g., zh, es)")
    p.add_argument("--prefer-db", action="store_true", help="Prefer DB LLM logs over file logs")
    args = p.parse_args()

    db = open_account_session(args.account)
    try:
        # Fetch reading text content to compute spans reliably
        from server.models import ReadingText
        rt = db.get(ReadingText, int(args.text))
        if not rt or rt.account_id != int(args.account):
            print("error: reading text not found for this account", file=sys.stderr)
            return 1
        text = rt.content or ""
        if not text.strip():
            print("error: reading text has no content", file=sys.stderr)
            return 2
        n = reconstruct_glosses_from_logs(
            db,
            account_id=int(args.account),
            text_id=int(args.text),
            text=text,
            lang=str(args.lang),
            prefer_db=bool(args.prefer_db),
        )
        # Print result count; idempotent merges will return 0 if everything is already present
        print(f"inserted={n}")
        return 0
    finally:
        try:
            db.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
