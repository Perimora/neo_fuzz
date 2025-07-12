from enum import Enum
from typing import Any


class LoggingLevel(Enum):
    """Enum for logging levels."""

    MASTER = (0, "master")  # Special level for master log
    DEBUG = (10, "debug")
    INFO = (20, "info")
    WARNING = (30, "warning")
    ERROR = (40, "error")
    CRITICAL = (50, "critical")

    def __init__(self, level_value: int, key: str) -> None:
        self.level_value = level_value
        self.key = key

    @property
    def level(self) -> int:
        """Return the logging level value."""
        return self.level_value

    @classmethod
    def from_string(cls, level_str: str) -> Any:
        """Convert a string to a LoggingLevel."""
        level_str = level_str.upper()
        if hasattr(cls, level_str):
            return getattr(cls, level_str)
        raise ValueError(f"Invalid logging level: {level_str}")
