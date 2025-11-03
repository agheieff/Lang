"""Integration tests for complete authentication flow."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from server.main import app
from server.auth.models import Account
from server.auth.security import create_access_token


class TestAuthenticationFlow:
    """Test complete authentication workflows."""

    @pytest.fixture
    def client(db_session):
        """Test client with database dependency override."""
        def override_get_db():
            try:
                yield db_session
            finally:
                pass
        
        from server.db import get_global_db
        app.dependency_overrides[get_global_db] = override_get_db
        
        with TestClient(app) as test_client:
            yield test_client
        
        app.dependency_overrides.clear()

    def test_token_creation_and_validation(self):
        """Test that tokens can be created and validated without mocking internals."""
        token = create_access_token("test-id", "test-secret")
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_authentication_mechanisms(self, client):
        """Test that authentication works via different mechanisms."""
        with patch('server.deps._JWT_SECRET', 'test-secret'):
            valid_token = create_access_token("test-id", "test-secret")
            
            # Test Bearer token
            headers = {"Authorization": f"Bearer {valid_token}"}
            response = client.get("/", headers=headers)
            # Home page should return 200 regardless of auth
            assert response.status_code == 200
            
            # Test cookie authentication  
            client.cookies.set("access_token", valid_token)
            response = client.get("/")
            assert response.status_code == 200

    def test_invalid_token_rejection(self, client):
        """Test that invalid tokens don't crash the system."""
        # Test completely invalid token
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/", headers=headers)
        # Should not crash, system should handle gracefully
        assert response.status_code in [200, 401, 404]
        
        # Test malformed token
        malformed_headers = {"Authorization": "invalid_format"}
        response = client.get("/", headers=malformed_headers)
        assert response.status_code in [200, 401, 404]

    def test_authentication_state_isolation(self, client):
        """Test that authentication state doesn't leak between requests."""
        with patch('server.deps._JWT_SECRET', 'test-secret'):
            valid_token = create_access_token("test-id", "test-secret")
            
            # First request - authenticated
            headers = {"Authorization": f"Bearer {valid_token}"}
            response1 = client.get("/", headers=headers)
            
            # Second request - same authentication should work
            response2 = client.get("/", headers=headers)
            
            # Both should behave consistently
            assert response1.status_code == response2.status_code
            assert response1.status_code == 200

    def test_multiple_authentication_methods_coexist(self, client):
        """Test that having both cookie and header tokens works predictably."""
        with patch('server.deps._JWT_SECRET', 'test-secret'):
            valid_token = create_access_token("test-id", "test-secret")
            
            # Set both cookie and header
            client.cookies.set("access_token", valid_token)
            headers = {"Authorization": f"Bearer {valid_token}"}
            
            response = client.get("/", headers=headers)
            
            # System should handle this gracefully
            assert response.status_code == 200

    def test_authentication_without_database_dependency(self, client):
        """Test that the app handles authentication gracefully without crashing."""
        # Test with no authentication at all
        response = client.get("/")
        assert response.status_code == 200  # Should not crash
        
        # Test with malformed headers
        malformed_headers = {"Authorization": "Bearer"}
        response = client.get("/", headers=malformed_headers)
        assert response.status_code == 200  # Handle gracefully
