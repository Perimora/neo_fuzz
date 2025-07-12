from src.classes.log.logger import Logger
from src.classes.log.logging_env import LoggingEnv


def test_logger_initialization() -> None:
    """Test the Logger initialization."""
    env = LoggingEnv.TEST
    logger = Logger(env)

    assert logger.env == env
