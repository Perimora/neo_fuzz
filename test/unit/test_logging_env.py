"""Unit tests for LoggingEnv enum."""

import pytest

from src.classes.log.logging_env import LoggingEnv


class TestLoggingEnv:
    """Test LoggingEnv enum functionality."""

    def test_logging_env_values(self) -> None:
        """Test that logging environments have correct values."""
        assert LoggingEnv.DEV.value == "development"
        assert LoggingEnv.PROD.value == "production"
        assert LoggingEnv.TEST.value == "test"

    def test_from_string_valid(self) -> None:
        """Test from_string with valid inputs."""
        assert LoggingEnv.from_string("dev") == LoggingEnv.DEV
        assert LoggingEnv.from_string("PROD") == LoggingEnv.PROD
        assert LoggingEnv.from_string("Test") == LoggingEnv.TEST

    def test_from_string_invalid(self) -> None:
        """Test from_string with invalid input."""
        with pytest.raises(ValueError, match="Invalid logging environment"):
            LoggingEnv.from_string("invalid")
