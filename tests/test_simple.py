"""Simple demonstration of Pareto efficient testing."""
import pytest
from server.models import Base


def test_base_import():
    """Test that we can import the models."""
    assert Base is not None
    assert hasattr(Base, 'metadata')
