"""
pytest configuration - shared fixtures and setup
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env
from dotenv import load_dotenv
env_path = project_root / ".env"
load_dotenv(env_path)


def pytest_configure(config):
    """Configure pytest with custom settings."""
    import sys
    # Ensure the project root is in the path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
