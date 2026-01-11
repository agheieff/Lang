"""Initialize SRS values for existing lexemes that don't have them yet."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.db import SessionLocal
from server.models import Lexeme
from server.services.srs import initialize_lexeme_srs


def main():
    """Initialize SRS values for all lexemes that need them."""
    db = SessionLocal()

    try:
        # Find all lexemes without familiarity values
        lexemes = db.query(Lexeme).filter(Lexeme.familiarity.is_(None)).all()

        print(f"Found {len(lexemes)} lexemes without SRS values")

        for i, lexeme in enumerate(lexemes):
            initialize_lexeme_srs(lexeme)
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(lexemes)} lexemes...")
                db.flush()  # Flush every 100 to avoid memory issues

        db.commit()
        print(f"✓ Initialized SRS values for {len(lexemes)} lexemes")

    except Exception as e:
        print(f"✗ Error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
