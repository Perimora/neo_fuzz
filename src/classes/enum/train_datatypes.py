from enum import Enum


class DataSetType(Enum):
    PPO_CLEAN = "ppo_clean"
    PPO = "ppo"
    SSL = "ssl"
    BACKUP = "backup"
