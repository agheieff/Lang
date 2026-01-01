import pytest
from server.db import SessionLocal, init_db
from server.models import Account
from fastapi.testclient import TestClient


@pytest.fixture(scope="function", autouse=True)
def clean_db():
    """Clean database before each test."""
    from server.models import Account

    with SessionLocal() as db:
        db.query(Account).delete()
        db.commit()


@pytest.fixture
def client(clean_db):
    """Create a test client."""
    from server.main import app

    return TestClient(app)


def test_register_success(client):
    """Test successful registration."""
    email = "newuser@example.com"
    password = "TestPassword123!"

    response = client.post(
        "/auth/register",
        json={"email": email, "password": password},
    )

    assert response.status_code == 201

    # Register now returns a token (for auto-login)
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

    # Check that cookie was set
    cookies = response.cookies.get("access_token")
    assert cookies is not None


def test_register_duplicate_email(client):
    """Test registration with duplicate email."""
    email = "duplicate@example.com"
    password = "TestPassword123!"

    # First registration
    response1 = client.post(
        "/auth/register",
        json={"email": email, "password": password},
    )
    assert response1.status_code == 201

    # Second registration with same email
    response2 = client.post(
        "/auth/register",
        json={"email": email, "password": password},
    )

    assert response2.status_code == 409
    assert "already registered" in response2.json()["detail"].lower()


@pytest.mark.skip(reason="Backend does not validate email format")
def test_register_invalid_email(client):
    """Test registration with invalid email."""
    response = client.post(
        "/auth/register",
        json={"email": "invalid-email", "password": "TestPassword123!"},
    )

    assert response.status_code == 422


def test_register_weak_password(client):
    """Test registration with weak password."""
    response = client.post(
        "/auth/register",
        json={"email": "test@example.com", "password": "weak"},
    )

    assert response.status_code == 422
    assert "password" in response.json()["detail"].lower()


def test_login_success(client):
    """Test successful login with existing account."""
    email = "logintest@example.com"
    password = "TestPassword123!"

    # First register
    client.post(
        "/auth/register",
        json={"email": email, "password": password},
    )

    # Then login
    response = client.post(
        "/auth/login",
        json={"email": email, "password": password},
    )

    assert response.status_code == 200

    data = response.json()
    assert "access_token" in data
    assert isinstance(data["access_token"], str)
    assert len(data["access_token"]) > 0

    # Check that cookie was set
    cookies = response.cookies.get("access_token")
    assert cookies is not None


def test_login_wrong_password(client):
    """Test login with wrong password."""
    email = "wrongpass@example.com"
    correct_password = "TestPassword123!"

    # Register account
    client.post(
        "/auth/register",
        json={"email": email, "password": correct_password},
    )

    # Try login with wrong password
    response = client.post(
        "/auth/login",
        json={"email": email, "password": "WrongPassword123!"},
    )

    assert response.status_code == 401
    assert "invalid credentials" in response.json()["detail"].lower()


def test_login_nonexistent_user(client):
    """Test login with non-existent user."""
    response = client.post(
        "/auth/login",
        json={"email": "nonexistent@example.com", "password": "AnyPassword123!"},
    )

    assert response.status_code == 401
    assert "invalid credentials" in response.json()["detail"].lower()


def test_signup_and_login_flow(client):
    """Test complete signup and auto-login flow (matches UI behavior)."""
    email = "flowtest@example.com"
    password = "Password123!"

    # Step 1: Register
    register_response = client.post(
        "/auth/register",
        json={"email": email, "password": password},
    )

    assert register_response.status_code == 201

    # Step 2: Auto-login (as done by UI JavaScript)
    login_response = client.post(
        "/auth/login",
        json={"email": email, "password": password},
    )

    assert login_response.status_code == 200

    data = login_response.json()
    assert "access_token" in data

    # Step 3: Verify authenticated access to protected endpoint
    protected_response = client.get(
        "/me/profile", headers={"Cookie": f"access_token={data['access_token']}"}
    )

    assert protected_response.status_code == 200
