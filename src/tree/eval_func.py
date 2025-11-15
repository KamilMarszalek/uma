from collections import Counter
from typing import Protocol

import numpy as np
from get_splits import get_splits


class EvalFunc(Protocol):
    def __call__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        feature: int,
    ) -> float: ...


class InformationGain:
    def __call__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        feature: int,
    ) -> float:
        current_entropy = self.entropy(targets)
        splits = get_splits(data, targets, feature)
        weighted_entropy = 0.0
        for subset in splits.values():
            if subset.size == 0:
                continue
            weight = subset.size / targets.size
            weighted_entropy += weight * self.entropy(subset)
        return float(current_entropy - weighted_entropy)

    def entropy(self, targets: np.ndarray) -> float:
        targets = targets.ravel()
        total = targets.size
        if total == 0:
            return 0.0
        probs = np.array(list(Counter(targets).values()), dtype=float) / total
        return float(-np.sum(probs * np.log2(probs)))


class GiniGain:
    def __call__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        feature: int,
    ) -> float:
        current_gini = self.gini(targets)
        splits = get_splits(data, targets, feature)
        weighted_gini = 0.0
        for subset in splits.values():
            if subset.size == 0:
                continue
            weight = subset.size / targets.size
            weighted_gini += weight * self.gini(subset)
        return float(current_gini - weighted_gini)

    def gini(self, targets: np.ndarray) -> float:
        targets = targets.ravel()
        total = targets.size
        if total == 0:
            return 0.0
        probs = np.array(list(Counter(targets).values()), dtype=float) / total
        return float(1.0 - np.sum(probs**2))
