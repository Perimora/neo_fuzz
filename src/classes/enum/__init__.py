"""Enum classes for targets, training types and dataset types."""

from src.classes.enum.targets import Target
from src.classes.enum.train_datatypes import DataSetType
from src.classes.enum.train_types import TrainType

__all__: list[str] = ["Target", "TrainType", "DataSetType"]
