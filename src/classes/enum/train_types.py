from enum import Enum


class TrainType(Enum):
    SSL = "SSL"
    PPO_STRUCTURE = "PPO for Syntax & Semantic"
    PPO_COVERAGE = "ppo for target coverage"
