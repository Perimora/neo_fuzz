from app import greet

from src.classes.log.logger import Logger
from src.classes.log.logging_env import LoggingEnv

logger = Logger(LoggingEnv.TEST)


def test_greet_with_name() -> None:
    logger.info("Starting test_greet_with_name")
    result = greet("Alice")
    logger.info(f"greet('Alice') returned: {result}")
    assert result == "Hello, Alice!"
    logger.info("test_greet_with_name passed")


def test_greet_without_name() -> None:
    logger.info("Starting test_greet_without_name")
    result = greet("")
    logger.info(f"greet('') returned: {result}")
    assert result == "Hello, Stranger!"
    logger.info("test_greet_without_name passed")
