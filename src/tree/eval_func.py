from collections import Counter
from enum import Enum
from typing import Any, Protocol

import numpy as np

from src.tree.get_splits import get_splits


def probabilities(targets: np.ndarray) -> np.ndarray:
    targets = targets.ravel()
    total = targets.size
    if total == 0:
        return np.array([], dtype=float)
    counts = np.array(list(Counter(targets).values()), dtype=float)
    return counts / total


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
        probs = probabilities(targets)
        if probs.size == 0:
            return 0.0
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
        probs = probabilities(targets)
        if probs.size == 0:
            return 0.0
        return float(1.0 - np.sum(probs**2))


class CARTGiniGain:
    def __call__(
        self,
        *,
        parent_pos: int,
        parent_total: int,
        left_pos: np.ndarray,
        left_total: np.ndarray,
    ) -> np.ndarray:
        p_parent = parent_pos / parent_total if parent_total else 0.0
        g_parent = 2.0 * p_parent * (1.0 - p_parent)
        right_total = parent_total - left_total
        right_pos = parent_pos - left_pos
        left_total = left_total.astype(np.float64)
        right_total = right_total.astype(np.float64)
        p_left = left_pos / left_total
        p_right = right_pos / right_total
        g_left = 2.0 * p_left * (1.0 - p_left)
        g_right = 2.0 * p_right * (1.0 - p_right)
        weighted_gini = (left_total / parent_total) * g_left + (
            right_total / parent_total
        ) * g_right
        return g_parent - weighted_gini


class GainRatio:
    def __init__(self) -> None:
        self.info_gain = InformationGain()

    def __call__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        feature: int,
    ) -> float:
        info_gain = self.info_gain(data, targets, feature)
        if info_gain == 0.0:
            return 0.0
        splits = get_splits(data, targets, feature)
        split_info = 0.0
        for subset in splits.values():
            if subset.size == 0:
                continue
            weight = subset.size / targets.size
            split_info -= weight * np.log2(weight)
        if split_info == 0.0:
            return 0.0
        return float(info_gain / split_info)


class ID3EvalFunc(Protocol):
    def __call__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        feature: int,
    ) -> float: ...


class CARTEvalFunc(Protocol):
    def __call__(
        self,
        *,
        parent_pos: int,
        parent_total: int,
        left_pos: np.ndarray,
        left_total: np.ndarray,
    ) -> np.ndarray: ...


class SKLearnEvalFuncEnum(Enum):
    SKLEARN_ENTROPY = "entropy"
    SKLEARN_GINI = "gini"


class EvalFunction(Enum):
    # ID3
    ID3_INFORMATION_GAIN = InformationGain
    ID3_GAIN_RATIO = GainRatio
    ID3_GINI_GAIN = GiniGain

    # CART
    CART_GINI_GAIN = CARTGiniGain

    SKLEARN_ENTROPY = "entropy"
    SKLEARN_GINI = "gini"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if isinstance(self.value, str):
            return self.value
        return self.value()(*args, **kwargs)
