"""Development utility to reset the database and per-account DBs.
Also removes local LLM generation logs and session logs under ARC_OR_LOG_DIR (or data/llm_stream_logs and data/session_logs).

Usage:
    python reset_db.py        # Reset users only (keep texts and shared data)
    python reset_db.py --full # Reset everything including texts
"""
import argparse
import os
import sys
import shutil
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from server.db import recreate_db, get_db_info, DB_PATH, global_engine

# LLM generation logs directory
LOG_DIR = Path(os.getenv("ARC_OR_LOG_DIR") or (Path.cwd() / "data" / "llm_stream_logs"))

# Session logs directory  
SESSION_LOG_DIR = Path(os.getenv("ARC_OR_LOG_DIR") or (Path.cwd() / "data" / "session_logs"))

# Tables that contain shared data (texts, translations, etc.) - preserved in partial reset
SHARED_TABLES = {
    "reading_texts",
    "reading_text_translations",
    "reading_word_glosses",
    "text_vocabulary",
    "languages",
    "llm_models",
    "subscription_tiers",
}

# Global tables with user data - cleared in partial reset
GLOBAL_USER_TABLES = {
    "accounts",
    "profiles",
    "profile_prefs",
    "generation_logs",
    "translation_logs",
    "llm_request_logs",
    "reading_lookups",
}

# Per-account DB tables (cleared by deleting account DB files)
PER_ACCOUNT_TABLES = {
    "lexemes",
    "lexeme_variants",
    "user_lexeme_contexts",
    "word_events",
    "profile_text_reads",
    "profile_text_queue",
    "user_model_configs",
    "usage_tracking",
    "next_ready_overrides",
    "generation_retry_attempts",
}


def clear_user_tables():
    """Clear user-related tables in global DB while preserving shared data."""
    from sqlalchemy import text
    
    with global_engine.connect() as conn:
        # Disable foreign key checks temporarily
        conn.execute(text("PRAGMA foreign_keys = OFF"))
        
        # Get existing tables
        result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        existing_tables = {row[0] for row in result}
        
        # Clear global user tables
        for table in GLOBAL_USER_TABLES:
            if table in existing_tables:
                try:
                    conn.execute(text(f"DELETE FROM {table}"))
                    print(f"✓ Cleared table: {table}")
                except Exception as e:
                    print(f"! Failed to clear {table}: {e}")
        
        # Clear generated_for_account_id in reading_texts (disassociate from users)
        if "reading_texts" in existing_tables:
            conn.execute(text("UPDATE reading_texts SET generated_for_account_id = NULL"))
            print("✓ Cleared account references from reading_texts")
        
        # Re-enable foreign key checks
        conn.execute(text("PRAGMA foreign_keys = ON"))
        conn.commit()


def delete_account_dbs():
    """Legacy function - now no-op since we use a single database."""
    print("(skipping account DB deletion - using single database mode)")


def delete_logs():
    """Delete LLM and session logs."""
    if LOG_DIR.exists():
        try:
            shutil.rmtree(LOG_DIR)
            print(f"✓ Deleted LLM logs directory: {LOG_DIR}")
        except Exception as e:
            print(f"! Failed to delete LLM logs: {e}")

    if SESSION_LOG_DIR.exists():
        try:
            shutil.rmtree(SESSION_LOG_DIR)
            print(f"✓ Deleted session logs directory: {SESSION_LOG_DIR}")
        except Exception as e:
            print(f"! Failed to delete session logs: {e}")


def delete_cookies():
    """Delete local cookie jar."""
    try:
        cj = Path.cwd() / 'cookies.txt'
        if cj.exists():
            cj.unlink()
            print("✓ Deleted local cookies.txt")
    except Exception as e:
        print(f"! Failed to delete cookies.txt: {e}")


def full_reset():
    """Full database reset - delete and recreate everything."""
    # Fully remove the global DB file
    try:
        for p in [DB_PATH, DB_PATH.with_suffix(DB_PATH.suffix + "-wal"), DB_PATH.with_suffix(DB_PATH.suffix + "-shm")]:
            if p.exists():
                p.unlink()
                print(f"✓ Deleted global DB file: {p.name}")
    except Exception as e:
        print(f"! Failed to delete global DB files: {e}")

    # Recreate all tables
    recreate_db()
    print("✓ Global database reset complete (recreated)")

    delete_account_dbs()
    delete_logs()
    delete_cookies()


def partial_reset():
    """Partial reset - clear user data but keep shared texts."""
    print("Clearing user data (keeping texts)...")
    clear_user_tables()
    delete_account_dbs()
    delete_logs()
    delete_cookies()
    print("✓ User data reset complete")


def show_db_info():
    """Display current database information."""
    print("Current database info:")
    info = get_db_info()
    print(f"  Global DB Path: {info['path']}")
    print(f"  Exists: {info['exists']}")
    print(f"  Size: {info['size_mb']:.2f} MB")
    print(f"  Tables: {', '.join(info['tables'])}")

    print("  Account databases: 0 (using single database mode)")

    print(f"  LLM logs dir: {LOG_DIR} {'(exists)' if LOG_DIR.exists() else '(missing)'}")
    print(f"  Session logs dir: {SESSION_LOG_DIR} {'(exists)' if SESSION_LOG_DIR.exists() else '(missing)'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset the database")
    parser.add_argument("--full", action="store_true", help="Full reset including texts (default: users only)")
    args = parser.parse_args()

    show_db_info()

    if args.full:
        prompt = "\nFull reset? This will DELETE ALL DATA including texts (y/N): "
    else:
        prompt = "\nReset user data? Texts will be preserved (y/N): "
    
    response = input(prompt)
    if response.lower() == 'y':
        if args.full:
            print("\nPerforming full reset...")
            full_reset()
        else:
            print("\nPerforming partial reset (users only)...")
            partial_reset()

        # Show updated info
        print()
        show_db_info()
        print("\nNote: You may still have a browser cookie set. Visit /auth/logout in the app or clear cookies for localhost to fully sign out.")
    else:
        print("Cancelled")