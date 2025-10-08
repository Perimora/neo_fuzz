import json
import logging
from typing import cast

from src.classes.log.logging_env import LoggingEnv
from src.classes.log.logging_handler import LoggingHandler


class Logger:
    """Logger class for handling logging operations."""

    PATH_CONFIG = "config/logging/logging_paths.json"
    FORMAT_CONFIG = "config/logging/logging_formatting.json"

    def __init__(self, env: LoggingEnv):
        """Initialize the Logger"""
        self.env = env
        self.handler = self.setup_logging_handler(env)
        self.logger = self.setup_logger()

    def setup_logging_handler(self, env: LoggingEnv) -> LoggingHandler:
        """Setup the logging handler based on the environment."""

        paths_cfg = self.load_paths_config()
        format_cfg = self.load_format_config()
        return LoggingHandler(env, paths_cfg, format_cfg)

    def load_format_config(self) -> dict[str, dict[str, str]]:
        """Load the logging format configuration."""

        with open(self.FORMAT_CONFIG, "r") as file:
            try:
                return cast(dict[str, dict[str, str]], json.load(file))
            except json.JSONDecodeError as e:
                raise ValueError(f"Error decoding JSON from {self.FORMAT_CONFIG}: {e}")

    def load_paths_config(self) -> dict[str, dict[str, str]]:
        """Load the logging paths configuration."""

        with open(self.PATH_CONFIG, "r") as file:
            try:
                return cast(dict[str, dict[str, str]], json.load(file))
            except json.JSONDecodeError as e:
                raise ValueError(f"Error decoding JSON from {self.PATH_CONFIG}: {e}")

    def setup_logger(self) -> logging.Logger:
        """Setup the logger with the handlers."""
        import logging

        logger = logging.getLogger(f"app.{self.env.value}")
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            logger.addHandler(self.handler.master_handler)
            logger.addHandler(self.handler.debug_handler)
            logger.addHandler(self.handler.info_handler)
            logger.addHandler(self.handler.warning_handler)
            logger.addHandler(self.handler.error_handler)
        return logger

    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)
