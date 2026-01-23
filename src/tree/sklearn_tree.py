from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from src.tree.base_tree import BaseTree
from src.tree.config import SKLearnTreeConfig


class SklearnTreeWrapper(BaseTree):
    def __init__(self, config: SKLearnTreeConfig) -> None:
        super().__init__(config.random_seed)
        self.features: list[int] = []
        self.model = DecisionTreeClassifier(
            random_state=config.random_seed,
            max_depth=config.max_depth,
            criterion=config.eval_function.value,
        )

    def fit(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        features: list[int],
    ) -> None:
        self.features = features
        subset_data = data[:, features]
        self.model.fit(subset_data, targets)

    def predict(self, sample: np.ndarray) -> Any:
        if sample.ndim == 1:
            sample_subset = sample[self.features].reshape(1, -1)
        else:
            sample_subset = sample[:, self.features]

        prediction = self.model.predict(sample_subset)

        return prediction[0] if prediction.size == 1 else prediction
