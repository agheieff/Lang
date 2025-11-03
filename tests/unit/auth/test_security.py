"""Unit tests for auth security functions."""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from server.auth.security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_token,
    pwd_context
)


class TestPasswordHashing:
    """Test password hashing and verification."""

    def test_hash_password_returns_hash(self):
        """Test that password hashing returns a non-empty string."""
        password = "test_password_123"
        hashed = hash_password(password)
        
        assert isinstance(hashed, str)
        assert len(hashed) > 0
        assert hashed != password  # Should not be plaintext

    def test_verify_password_correct(self):
        """Test password verification with correct password."""
        password = "correct_password"
        hashed = hash_password(password)
        
        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password."""
        password = "correct_password"
        wrong_password = "wrong_password"
        hashed = hash_password(password)
        
        assert verify_password(wrong_password, hashed) is False

    def test_verify_password_invalid_hash(self):
        """Test password verification with invalid hash."""
        password = "test_password"
        invalid_hash = "invalid_hash_format"
        
        assert verify_password(password, invalid_hash) is False

    def test_password_hash_consistency(self):
        """Test that the same password generates different hashes (salt)."""
        password = "test_password"
        hash1 = hash_password(password)
        hash2 = hash_password(password)
        
        assert hash1 != hash2  # Should be different due to salt
        assert verify_password(password, hash1) is True
        assert verify_password(password, hash2) is True

    def test_unicode_passwords(self):
        """Test password hashing with unicode characters."""
        password = "pássword_émojis_🚀"
        hashed = hash_password(password)
        
        assert verify_password(password, hashed) is True
        assert verify_password("different_password", hashed) is False

    @patch('server.auth.security._argon2_available')
    def test_password_hashing_backend_fallback(self, mock_argon2):
        """Test fallback to PBKDF2 when Argon2 is not available."""
        mock_argon2.return_value = False
        
        # This should not raise an exception
        password = "test_password"
        hashed = hash_password(password)
        assert verify_password(password, hashed) is True


class TestJWTTokenHandling:
    """Test JWT token creation and validation."""

    @pytest.fixture
    def test_secret(self):
        """Test secret key."""
        return "test-secret-key-for-testing"

    def test_create_access_token_basic(self, test_secret):
        """Test basic JWT token creation."""
        subject = "user123"
        token = create_access_token(subject, test_secret)
        
        assert isinstance(token, str)
        assert len(token) > 0
        # JWT tokens have 3 parts separated by dots
        assert token.count(".") == 2

    def test_create_access_token_with_integer_subject(self, test_secret):
        """Test JWT token creation with integer subject."""
        subject = 123
        token = create_access_token(subject, test_secret)
        
        assert isinstance(token, str)
        
        decoded = decode_token(token, test_secret)
        assert decoded is not None
        assert decoded["sub"] == "123"

    def test_decode_token_valid(self, test_secret):
        """Test decoding a valid JWT token."""
        subject = "user123"
        token = create_access_token(subject, test_secret)
        
        decoded = decode_token(token, test_secret)
        
        assert decoded is not None
        assert decoded["sub"] == subject
        assert "iat" in decoded
        assert "exp" in decoded

    def test_decode_token_invalid_secret(self, test_secret):
        """Test decoding token with wrong secret."""
        subject = "user123"
        token = create_access_token(subject, test_secret)
        wrong_secret = "wrong-secret"
        
        decoded = decode_token(token, wrong_secret)
        
        assert decoded is None

    def test_decode_token_invalid_token(self, test_secret):
        """Test decoding an invalid token."""
        invalid_tokens = [
            "not.a.jwt",
            "invalid.token.here",
            "",
            "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJzdWIiOiIxMjMiLCJpYXQiOjE2MDAwMDAwMDJ9."  # Missing signature
        ]
        
        for token in invalid_tokens:
            decoded = decode_token(token, test_secret)
            assert decoded is None

    def test_token_expiration(self, test_secret):
        """Test that tokens expire correctly."""
        subject = "user123"
        # Create token that expires in 1 second
        token = create_access_token(subject, test_secret, expires_minutes=1/60)
        
        # Should be valid immediately
        decoded = decode_token(token, test_secret)
        assert decoded is not None
        
        # Wait for expiration
        import time
        time.sleep(2)
        
        # Should be expired now
        decoded = decode_token(token, test_secret)
        assert decoded is None

    def test_token_with_custom_algorithm(self, test_secret):
        """Test creating token with custom algorithm."""
        subject = "user123"
        token = create_access_token(subject, test_secret, algorithm="HS256")
        
        decoded = decode_token(token, test_secret, algorithms=["HS256"])
        assert decoded is not None
        assert decoded["sub"] == subject

    def test_decode_token_wrong_algorithm(self, test_secret):
        """Test decoding token with wrong algorithm list."""
        subject = "user123"
        token = create_access_token(subject, test_secret, algorithm="HS256")
        
        # Try to decode with wrong algorithm list
        decoded = decode_token(token, test_secret, algorithms=["HS512"])
        assert decoded is None

    def test_token_payload_structure(self, test_secret):
        """Test that token payload contains expected fields."""
        subject = "user456"
        token = create_access_token(subject, test_secret)
        
        decoded = decode_token(token, test_secret)
        
        assert "sub" in decoded
        assert "iat" in decoded  
        assert "exp" in decoded
        assert decoded["sub"] == subject
        
        # Check that timestamps are reasonable
        now = datetime.now(timezone.utc).timestamp()
        assert abs(decoded["iat"] - now) < 10  # Within 10 seconds
        assert decoded["exp"] > decoded["iat"]  # Expiration is after issued

    def test_token_subject_types(self, test_secret):
        """Test token creation with different subject types."""
        test_cases = [
            ("123", "123"),  # String
            (123, "123"),   # Integer
            (0, "0"),       # Zero
            (999999, "999999")  # Large number
        ]
        
        for subject_input, expected_sub in test_cases:
            token = create_access_token(subject_input, test_secret)
            decoded = decode_token(token, test_secret)
            
            assert decoded is not None
            assert decoded["sub"] == expected_sub
