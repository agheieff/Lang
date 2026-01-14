"""Add queue_dirty tracking to Profile model."""

from sqlalchemy import text
from server.db import engine


def upgrade():
    """Add new fields to Profile table."""
    try:
        with engine.begin() as conn:
            # Add new columns
            conn.execute(text("""
                ALTER TABLE profiles
                ADD COLUMN queue_dirty BOOLEAN DEFAULT 0
            """))
            conn.execute(text("""
                ALTER TABLE profiles
                ADD COLUMN queue_built_at DATETIME
            """))
            conn.execute(text("""
                ALTER TABLE profiles
                ADD COLUMN queue_dirty_reason VARCHAR(64)
            """))

            # Create index
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_profile_queue_dirty
                ON profiles(queue_dirty, preferences_updating)
            """))

        print("✓ Migration completed: queue_dirty fields added to profiles table")

    except Exception as e:
        print(f"✗ Migration failed: {e}")
        raise


def downgrade():
    """Remove new fields from Profile table."""
    try:
        with engine.begin() as conn:
            # Drop the index
            conn.execute(text("DROP INDEX IF EXISTS ix_profile_queue_dirty"))

        print("⚠ Downgrade: Index dropped. Note: SQLite doesn't support DROP COLUMN, columns remain but unused.")

    except Exception as e:
        print(f"✗ Downgrade failed: {e}")
        raise


if __name__ == "__main__":
    upgrade()
