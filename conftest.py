"""Shared pytest fixtures."""
import pytest
from app import app


@pytest.fixture
def client():
    """Provide a Flask test client."""
    with app.test_client() as test_client:
        yield test_client