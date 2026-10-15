"""Abstract base class and contract for all evaluation metrics."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from evalguard.models import MetricScore, TestCase


class BaseMetric(ABC):
    """Abstract base class for all evaluation metrics in EvalGuard AI."""

    def __init__(self, name: str, threshold: float = 0.70, description: str = ""):
        self.name = name
        self.threshold = threshold
        self.description = description

    @abstractmethod
    def evaluate(self, test_case: TestCase) -> MetricScore:
        """Computes evaluation score and returns a MetricScore object."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', threshold={self.threshold})"
