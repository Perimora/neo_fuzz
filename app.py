from src.classes.log.logger import Logger
from src.classes.log.logging_env import LoggingEnv

logger = Logger(LoggingEnv.DEV)


def greet(name: str) -> str:
    if not name:
        logger.warning("No name provided to greet()")
        return "Hello, Stranger!"
    logger.info(f"Greeting user: {name}")
    return f"Hello, {name}!"


if __name__ == "__main__":
    print(greet("World"))
    print(greet(""))  # Test with no name to trigger warning
