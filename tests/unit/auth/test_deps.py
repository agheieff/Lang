"""Unit tests for authentication dependencies."""

import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException

from server.deps import get_current_account, require_tier
from server.auth.models import Account


class TestGetCurrentAccount:
    """Test the get_current_account dependency behavior."""

    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI request."""
        request = Mock()
        request.cookies.clear()
        return request

    @pytest.fixture 
    def mock_db(self):
        """Mock database session."""
        db = Mock()
        return db

    @pytest.fixture
    def valid_account(self):
        """Sample valid account for testing."""
        account = Account()
        account.id = 1
        account.email = "test@example.com"
        account.is_active = True
        account.subscription_tier = "free"
        return account

    @patch('server.deps.decode_token')
    def test_successful_authentication_returns_account(self, mock_decode, mock_request, mock_db, valid_account):
        """Test that valid authentication returns the expected account."""
        mock_decode.return_value = {"sub": "1"}
        mock_db.get.return_value = valid_account
        
        result = get_current_account(
            request=mock_request,
            db=mock_db,
            authorization="Bearer some_token"
        )
        
        assert result == valid_account

    @patch('server.deps.decode_token')
    def test_authentication_preferences(self, mock_decode, mock_request, mock_db, valid_account):
        """Test that header tokens are preferred over cookies when both present."""
        mock_decode.return_value = {"sub": "1"}
        mock_db.get.return_value = valid_account
        
        # Set both cookie and header
        mock_request.cookies = {"access_token": "cookie_token"}
        authorization = "Bearer header_token"
        
        result = get_current_account(
            request=mock_request,
            db=mock_db,
            authorization=authorization
        )
        
        assert result == valid_account

    def test_authentication_fails_without_credentials(self, mock_request, mock_db):
        """Test that authentication fails when no credentials are provided."""
        mock_request.cookies.clear()
        
        with pytest.raises(HTTPException) as exc_info:
            get_current_account(
                request=mock_request,
                db=mock_db,
                authorization=None
            )
        
        assert exc_info.value.status_code == 401

    def test_authentication_fails_with_invalid_token_format(self, mock_request, mock_db):
        """Test that authentication fails with malformed token."""
        mock_request.cookies.clear()
        
        with pytest.raises(HTTPException) as exc_info:
            get_current_account(
                request=mock_request,
                db=mock_db,
                authorization="not_a_bearer_token"
            )
        
        assert exc_info.value.status_code == 401

    @patch('server.deps.decode_token')
    def test_authentication_fails_with_nonexistent_account(self, mock_decode, mock_request, mock_db):
        """Test that authentication fails for non-existent accounts."""
        mock_decode.return_value = {"sub": "999"}
        mock_db.get.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            get_current_account(
                request=mock_request,
                db=mock_db,
                authorization="Bearer valid_format_token"
            )
        
        assert exc_info.value.status_code == 401

    @patch('server.deps.decode_token')
    def test_authentication_fails_for_inactive_accounts(self, mock_decode, mock_request, mock_db):
        """Test that authentication fails for inactive accounts."""
        inactive_account = Account()
        inactive_account.id = 1
        inactive_account.is_active = False
        
        mock_decode.return_value = {"sub": "1"}
        mock_db.get.return_value = inactive_account
        
        with pytest.raises(HTTPException) as exc_info:
            get_current_account(
                request=mock_request,
                db=mock_db,
                authorization="Bearer valid_format_token"
            )
        
        assert exc_info.value.status_code == 401


class TestRequireTier:
    """Test tier-based authorization behavior."""

    @pytest.fixture
    def account(self):
        """Sample account for testing."""
        account = Account()
        account.id = 1
        account.email = "test@example.com"
        return account

    def test_tier_authorization_succeeds_for_allowed_tiers(self, account):
        """Test that authorization succeeds when account tier is allowed."""
        account.subscription_tier = "pro"
        allowed_tiers = {"free", "pro", "premium"}
        
        tier_dep = require_tier(allowed_tiers)
        result = tier_dep(account=account)
        
        assert result == account

    def test_tier_authorization_fails_for_insufficient_tier(self, account):
        """Test that authorization fails when account tier is not allowed."""
        account.subscription_tier = "free"
        allowed_tiers = {"pro", "premium"}
        
        tier_dep = require_tier(allowed_tiers)
        
        with pytest.raises(HTTPException) as exc_info:
            tier_dep(account=account)
        
        assert exc_info.value.status_code == 403

    def test_tier_authorization_fails_without_assigned_tier(self, account):
        """Test that authorization fails when account has no tier assigned."""
        account.subscription_tier = None
        allowed_tiers = {"free", "pro"}
        
        tier_dep = require_tier(allowed_tiers)
        
        with pytest.raises(HTTPException) as exc_info:
            tier_dep(account=account)
        
        assert exc_info.value.status_code == 403

    def test_tier_authorization_succeeds_when_all_tiers_allowed(self, account):
        """Test that authorization succeeds when all tiers are allowed."""
        account.subscription_tier = "basic"
        all_tiers = {"free", "basic", "pro", "premium"}
        
        tier_dep = require_tier(all_tiers)
        result = tier_dep(account=account)
        
        assert result == account
