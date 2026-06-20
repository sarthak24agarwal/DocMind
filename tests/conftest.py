"""
Shared test configuration.

The test suite references a couple of convenience helpers off the `pytest`
module (`pytest.AsyncMock`, `pytest.any_str`) that aren't part of pytest's
public API. We attach them here so the existing tests can use them without
modification.
"""
import os
import unittest.mock as mock
import pytest

# Allow `pytest.AsyncMock(...)` as a shorthand for `unittest.mock.AsyncMock(...)`
pytest.AsyncMock = mock.AsyncMock

# Allow `pytest.any_str` as a wildcard matcher in `assert_called_once_with(...)`
pytest.any_str = mock.ANY

# Ensure Settings() can be instantiated without a real .env file during tests.
os.environ.setdefault("R2_ENDPOINT_URL", "https://mock.r2.cloudflarestorage.com")
os.environ.setdefault("R2_ACCESS_KEY_ID", "mock")
os.environ.setdefault("R2_SECRET_ACCESS_KEY", "mock")
os.environ.setdefault("R2_BUCKET_NAME", "mock")
os.environ.setdefault("OPENAI_API_KEY", "mock")
os.environ.setdefault("ANTHROPIC_API_KEY", "mock")
os.environ.setdefault("STRIPE_API_KEY", "sk_test_mock")
os.environ.setdefault("STRIPE_WEBHOOK_SECRET", "whsec_mock")
os.environ.setdefault("STRIPE_PRO_PRICE_ID", "price_mock")
