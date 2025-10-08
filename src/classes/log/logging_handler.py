import logging
import os

from src.classes.log.logging_env import LoggingEnv
from src.classes.log.logging_level import LoggingLevel


class LoggingHandler:
    """Handler for logging operations."""

    def __init__(
        self,
        env: LoggingEnv,
        paths_cfg: dict[str, dict[str, str]],
        format_cfg: dict[str, dict[str, str]],
    ):
        """Initialize the LoggingHandler."""
        self.env = env
        self.paths_cfg = paths_cfg
        self.format_cfg = format_cfg

        # setup handlers for each level
        self.master_handler = self.setup_file_handlers(LoggingLevel.MASTER)
        self.debug_handler = self.setup_file_handlers(LoggingLevel.DEBUG)
        self.info_handler = self.setup_file_handlers(LoggingLevel.INFO)
        self.warning_handler = self.setup_file_handlers(LoggingLevel.WARNING)
        self.error_handler = self.setup_file_handlers(LoggingLevel.ERROR)

    def get_path(self, level: LoggingLevel) -> str:
        """Get the logging path based on the environment."""
        try:
            return self.paths_cfg[self.env.value][level.key]
        except KeyError as e:
            raise ValueError(f"Invalid environment or log type: {e}")

    def get_format(self) -> dict[str, str]:
        """Get the logging format based on the environment."""
        try:
            return self.format_cfg[self.env.value]
        except KeyError as e:
            raise ValueError(f"Invalid environment: {e}")

    def setup_file_handlers(self, level: LoggingLevel) -> logging.FileHandler:
        """Setup logging handlers for given level."""
        try:
            path = self.get_path(level)
            format = self.get_format().get(
                "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            datefmt = self.get_format().get("datefmt", "%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            raise e

        self.ensure_path_exists(path)
        handler = logging.FileHandler(path)

        if level == LoggingLevel.MASTER:
            # Master handler captures all levels
            handler.setLevel(logging.DEBUG)
        else:
            # Other handlers only capture their specific level
            handler.setLevel(level.level)
            handler.addFilter(lambda record: record.levelno == level.level)

        handler.setFormatter(logging.Formatter(format, datefmt=datefmt))

        return handler

    def setup_master_handler(self) -> logging.FileHandler:
        """Setup master handler that captures all log levels."""
        try:
            path = self.paths_cfg[self.env.value]["master"]
            format = self.get_format().get(
                "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            datefmt = self.get_format().get("datefmt", "%Y-%m-%d %H:%M:%S")
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid master log configuration: {e}")

        self.ensure_path_exists(path)
        handler = logging.FileHandler(path)
        handler.setLevel(logging.DEBUG)  # Capture all levels
        handler.setFormatter(logging.Formatter(format, datefmt=datefmt))

        return handler

    @staticmethod
    def ensure_path_exists(path: str) -> None:
        """Ensure the directory for the given path exists."""
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
