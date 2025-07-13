"""Controller module for managing application logic and flow."""

from src.classes.controller.magma_container import MagmaContainer
from src.classes.controller.training_env import TrainingEnv

__all__: list[str] = ["MagmaContainer", "TrainingEnv"]
