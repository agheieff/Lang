"""Development utility to reset the database"""
import os
import sys
import shutil
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from server.db import recreate_db, get_db_info
from server.account_db import _ACCOUNTS_DIR

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

    response = input("\nReset ALL databases? This will DELETE ALL DATA (y/N): ")
    if response.lower() == 'y':
        print("Resetting databases...")

        # Reset global database
        recreate_db()
        print("✓ Global database reset complete")

        # Delete all account databases
        if _ACCOUNTS_DIR.exists():
            for db_file in account_dbs:
                db_file.unlink()
                print(f"✓ Deleted account database: {db_file.name}")

        print("✓ All account databases deleted")

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
    else:
        print("Cancelled")