"""Unit tests for LoggingLevel enum."""

import pytest

from src.classes.log.logging_level import LoggingLevel


class TestLoggingLevel:
    """Test LoggingLevel enum functionality."""

    def test_logging_level_values(self) -> None:
        """Test that logging levels have correct values."""
        assert LoggingLevel.DEBUG.level == 10
        assert LoggingLevel.INFO.level == 20
        assert LoggingLevel.WARNING.level == 30
        assert LoggingLevel.ERROR.level == 40
        assert LoggingLevel.CRITICAL.level == 50

    def test_logging_level_keys(self) -> None:
        """Test that logging levels have correct keys."""
        assert LoggingLevel.DEBUG.key == "debug"
        assert LoggingLevel.INFO.key == "info"
        assert LoggingLevel.WARNING.key == "warning"
        assert LoggingLevel.ERROR.key == "error"
        assert LoggingLevel.CRITICAL.key == "critical"
        assert LoggingLevel.MASTER.key == "master"

    def test_from_string_valid(self) -> None:
        """Test from_string with valid inputs."""
        assert LoggingLevel.from_string("debug") == LoggingLevel.DEBUG
        assert LoggingLevel.from_string("INFO") == LoggingLevel.INFO
        assert LoggingLevel.from_string("Warning") == LoggingLevel.WARNING

    def test_from_string_invalid(self) -> None:
        """Test from_string with invalid input."""
        with pytest.raises(ValueError, match="Invalid logging level"):
            LoggingLevel.from_string("invalid")
