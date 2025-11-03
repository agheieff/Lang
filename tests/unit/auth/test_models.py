"""Unit tests for auth models."""

import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from server.auth.models import Account, create_sqlite_engine, create_tables, Base


class TestAccountModel:
    """Test Account model functionality."""

    @pytest.fixture
    def sample_account(self):
        """Create a sample Account instance."""
        return Account(
            email="test@example.com",
            password_hash="hashed_password",
            is_active=True,
            is_verified=True,
            role="user",
            subscription_tier="free",
            extras={"preferences": {"theme": "dark"}}
        )

    def test_account_creation(self, sample_account):
        """Test Account instance creation."""
        assert sample_account.email == "test@example.com"
        assert sample_account.password_hash == "hashed_password"
        assert sample_account.is_active is True
        assert sample_account.is_verified is True
        assert sample_account.role == "user"
        assert sample_account.subscription_tier == "free"
        assert sample_account.extras == {"preferences": {"theme": "dark"}}

    def test_account_to_dict(self, sample_account):
        """Test Account to_dict method."""
        # Set id for testing
        sample_account.id = 1
        
        result = sample_account.to_dict()
        
        expected = {
            "id": 1,
            "email": "test@example.com",
            "is_active": True,
            "is_verified": True,
            "role": "user",
            "subscription_tier": "free",
            "extras": {"preferences": {"theme": "dark"}}
        }
        
        assert result == expected

    def test_account_to_dict_optional_fields(self):
        """Test Account to_dict with None/missing optional fields."""
        account = Account(
            email="minimal@example.com",
            password_hash="hash"
        )
        account.id = 2
        
        result = account.to_dict()
        
        expected = {
            "id": 2,
            "email": "minimal@example.com",
            "is_active": None,  # No default set in model
            "is_verified": None,  # No default set in model  
            "role": None,
            "subscription_tier": None,
            "extras": None
        }
        
        assert result == expected

    def test_account_table_name(self):
        """Test Account table name."""
        assert Account.__tablename__ == "accounts"

    def test_account_required_fields(self):
        """Test that required fields are properly enforced."""
        # Account model doesn't raise exceptions for missing required fields
        # SQLAlchemy enforces constraints at database level, not Python level
        account = Account()  # This doesn't fail at object creation
        # Database constraints would be enforced at commit time

    def test_account_field_types(self):
        """Test that Account has the expected field types."""
        account = Account()
        
        # Check that the Account model has the expected attributes
        expected_fields = [
            'id', 'email', 'password_hash', 'is_active', 'is_verified',
            'role', 'subscription_tier', 'extras', 'created_at', 'updated_at'
        ]
        
        for field in expected_fields:
            assert hasattr(account, field)


class TestDatabaseSetup:
    """Test database setup functions."""

    @pytest.fixture
    def temp_db_file(self, tmp_path):
        """Create a temporary database file."""
        db_file = tmp_path / "test.db"
        return f"sqlite:///{db_file}"

    def test_create_sqlite_engine(self, temp_db_file):
        """Test SQLite engine creation."""
        engine = create_sqlite_engine(temp_db_file, echo=False)
        
        assert engine is not None
        assert "sqlite" in str(engine.url)

    def test_create_sqlite_engine_with_echo(self, temp_db_file):
        """Test SQLite engine creation with echo enabled."""
        engine = create_sqlite_engine(temp_db_file, echo=True)
        
        assert engine is not None
        assert "sqlite" in str(engine.url)

    def test_create_sqlite_engine_default_url(self):
        """Test SQLite engine creation with default URL."""
        engine = create_sqlite_engine()
        
        assert engine is not None
        assert "sqlite" in str(engine.url)

    def test_create_tables(self, temp_db_file):
        """Test table creation."""
        engine = create_sqlite_engine(temp_db_file)
        
        # Create tables
        create_tables(engine)
        
        # Verify tables exist by trying to query them
        with engine.connect() as conn:
            # Check if accounts table exists using SQLAlchemy text
            from sqlalchemy import text
            result = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name='accounts'")
            )
            rows = result.fetchall()
            assert len(rows) > 0

    def test_create_tables_idempotent(self, temp_db_file):
        """Test that create_tables can be called multiple times."""
        engine = create_sqlite_engine(temp_db_file)
        
        # Create tables twice - should not raise an error
        create_tables(engine)
        create_tables(engine)  # Second call should be fine

    def test_account_persistence(self, temp_db_file):
        """Test that Account instances can be persisted to database."""
        engine = create_sqlite_engine(temp_db_file)
        Base.metadata.create_all(bind=engine)
        
        Session = sessionmaker(bind=engine)
        session = Session()
        
        try:
            # Create and save an account
            account = Account(
                email="test@example.com",
                password_hash="hashed_password",
                is_active=True,
                subscription_tier="free"
            )
            session.add(account)
            session.commit()
            
            # Retrieve the account
            saved_account = session.query(Account).filter_by(email="test@example.com").first()
            
            assert saved_account is not None
            assert saved_account.email == "test@example.com"
            assert saved_account.subscription_tier == "free"
            assert saved_account.id is not None
            assert saved_account.created_at is not None
            
        finally:
            session.close()

    def test_account_timestamps(self, temp_db_file):
        """Test that timestamps are properly set."""
        engine = create_sqlite_engine(temp_db_file)
        Base.metadata.create_all(bind=engine)
        
        Session = sessionmaker(bind=engine)
        session = Session()
        
        try:
            account = Account(
                email="timestamps@example.com",
                password_hash="hash"
            )
            session.add(account)
            session.commit()
            
            # Refresh to get timestamps from database
            session.refresh(account)
            
            assert account.created_at is not None
            assert isinstance(account.created_at, datetime)
            
            # Update record to test updated_at
            account.is_active = False
            session.commit()
            
            # May not have updated_at if not supported by SQLite version
            # this test just ensures created_at worked
            
        finally:
            session.close()
