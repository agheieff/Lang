"""Development utility to reset the database and per-account DBs.
Also removes local LLM generation logs and session logs under ARC_OR_LOG_DIR (or data/llm_stream_logs and data/session_logs).
"""
import os
import sys
import shutil
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from server.db import recreate_db, get_db_info, DB_PATH
from server.account_db import _ACCOUNTS_DIR

# LLM generation logs directory
LOG_DIR = Path(os.getenv("ARC_OR_LOG_DIR") or (Path.cwd() / "data" / "llm_stream_logs"))

# Session logs directory  
SESSION_LOG_DIR = Path(os.getenv("ARC_OR_LOG_DIR") or (Path.cwd() / "data" / "session_logs"))

if __name__ == "__main__":
    print("Current database info:")
    info = get_db_info()
    print(f"  Global DB Path: {info['path']}")
    print(f"  Exists: {info['exists']}")
    print(f"  Size: {info['size_mb']:.2f} MB")
    print(f"  Tables: {', '.join(info['tables'])}")

    # Check for account databases
    account_dbs = list(_ACCOUNTS_DIR.glob("*.db")) if _ACCOUNTS_DIR.exists() else []
    print(f"  Account databases: {len(account_dbs)}")
    for db_file in account_dbs:
        print(f"    - {db_file.name} ({db_file.stat().st_size / 1024:.1f} KB)")

    # Show LLM logs dir
    print(f"  LLM logs dir: {LOG_DIR} {'(exists)' if LOG_DIR.exists() else '(missing)'}")
    print(f"  Session logs dir: {SESSION_LOG_DIR} {'(exists)' if SESSION_LOG_DIR.exists() else '(missing)'}")

    response = input("\nReset ALL databases? This will DELETE ALL DATA (y/N): ")
    if response.lower() == 'y':
        print("Resetting databases...")

        # Fully remove the global DB file (auth tables, sequences, everything)
        try:
            for p in [DB_PATH, DB_PATH.with_suffix(DB_PATH.suffix + "-wal"), DB_PATH.with_suffix(DB_PATH.suffix + "-shm")]:
                if p.exists():
                    p.unlink()
                    print(f"✓ Deleted global DB file: {p.name}")
        except Exception as e:
            print(f"! Failed to delete global DB files: {e}")

        # Recreate all tables (app + auth)
        recreate_db()
        print("✓ Global database reset complete (recreated)")

        # Delete all account databases
        if _ACCOUNTS_DIR.exists():
            # Remove *.db and associated SQLite sidecar files
            patterns = ["*.db", "*.db-wal", "*.db-shm"]
            removed = 0
            for pat in patterns:
                for f in _ACCOUNTS_DIR.glob(pat):
                    try:
                        f.unlink()
                        removed += 1
                        print(f"✓ Deleted account DB file: {f.name}")
                    except Exception as e:
                        print(f"! Failed to delete {f}: {e}")
            if removed == 0:
                print("(no account DB files found)")

        print("✓ All account databases deleted")

        # Delete LLM generation logs directory
        if LOG_DIR.exists():
            try:
                shutil.rmtree(LOG_DIR)
                print(f"✓ Deleted LLM logs directory: {LOG_DIR}")
            except Exception as e:
                print(f"! Failed to delete LLM logs: {e}")

        # Delete session logs directory
        if SESSION_LOG_DIR.exists():
            try:
                shutil.rmtree(SESSION_LOG_DIR)
                print(f"✓ Deleted session logs directory: {SESSION_LOG_DIR}")
            except Exception as e:
                print(f"! Failed to delete session logs: {e}")

        # Delete any local cookie jar used by CLI tools
        try:
            cj = Path.cwd() / 'cookies.txt'
            if cj.exists():
                cj.unlink()
                print("✓ Deleted local cookies.txt")
        except Exception as e:
            print(f"! Failed to delete cookies.txt: {e}")

        # Show updated info
        info = get_db_info()
        print(f"\nNew database info:")
        print(f"  Global DB Path: {info['path']}")
        print(f"  Exists: {info['exists']}")
        print(f"  Size: {info['size_mb']:.2f} MB")
        print(f"  Tables: {', '.join(info['tables'])}")

        # Check account databases after reset
        account_dbs = list(_ACCOUNTS_DIR.glob("*.db")) if _ACCOUNTS_DIR.exists() else []
        print(f"  Account databases: {len(account_dbs)}")
        print(f"  LLM logs dir exists: {LOG_DIR.exists()}")
        print("\nNote: You may still have a browser cookie set. Visit /auth/logout in the app or clear cookies for localhost to fully sign out.")
    else:
        print("Cancelled")