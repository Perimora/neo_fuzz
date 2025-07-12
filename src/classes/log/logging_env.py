from enum import Enum
from typing import Any


class LoggingEnv(Enum):
    """Enum for Logging Environment."""

    DEV = "development"
    PROD = "production"
    TEST = "test"

    @classmethod
    def from_string(cls, env_str: str) -> Any:
        """Convert a string to a LoggingEnv."""
        env_str = env_str.lower()
        if hasattr(cls, env_str.upper()):
            return getattr(cls, env_str.upper())
        raise ValueError(f"Invalid logging environment: {env_str}")
