from abc import ABC, abstractmethod
from collections import Counter
from typing import Any

import numpy as np

from src.tree.node import Node

NDIM = 2


class BaseTree(ABC):
    def __init__(self, random_seed: int = 42) -> None:
        self.rng = np.random.default_rng(random_seed)
        self.root: Node | None = None

    @abstractmethod
    def predict(self, sample: np.ndarray) -> Any: ...

    @abstractmethod
    def fit(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        features: list[int],
    ) -> None: ...

    def predict_many(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples)
        if samples.ndim != NDIM:
            raise ValueError("predict_many expects a 2D array of samples.")
        return np.fromiter(
            (self.predict(row) for row in samples),
            dtype=object,
            count=samples.shape[0],
        )

    @staticmethod
    def most_common_label(targets: np.ndarray) -> Any:
        targets = targets.ravel()
        if targets.size == 0:
            return None
        return Counter(targets).most_common(1)[0][0]
