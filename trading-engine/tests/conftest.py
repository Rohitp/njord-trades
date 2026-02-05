"""
Pytest configuration and shared fixtures for the trading engine test suite.

Test Structure:
- tests/unit/ - Fast tests with mocked dependencies (no external services)
- tests/integration/ - Tests requiring database, external APIs, etc.

Run unit tests only:
    pytest -m "not integration"

Run integration tests only:
    pytest -m integration

Run all tests:
    pytest
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires external services)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """
    Automatically mark tests based on their location.

    Tests in tests/integration/ are automatically marked with @pytest.mark.integration
    Tests in tests/test_database/test_pgvector.py are marked as integration
    """
    for item in items:
        # Mark tests in integration folder
        if "/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark pgvector tests as integration (requires running database)
        if "test_pgvector" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
