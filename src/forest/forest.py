from collections import Counter
from typing import Any

import numpy as np

from src.forest.config import TournamentForestConfig
from src.tree.tree import CARTTree, ID3Tree


class TournamentForest:
    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        config: TournamentForestConfig,
    ) -> None:
        self.data = data
        self.targets = targets
        self.config = config
        self.rng = np.random.default_rng()
        self.forest: list[ID3Tree | CARTTree] = []

    def build(self) -> None:
        rows, cols = self.data.shape

        for _ in range(self.config.num_of_trees):
            n = int(rows * self.config.sample_ratio)
            indices = self.rng.choice(rows, size=n, replace=True)
            data_boot = self.data[indices]
            targets_boot = self.targets[indices]
            k = max(1, int(cols * self.config.feature_ratio))
            feature_boot = list(self.rng.choice(cols, size=k, replace=False))
            tree_config = self.config.tree_config_class(
                max_depth=self.config.max_depth,
                eval_function=self.config.eval_function,
                tournament_size=self.config.tournament_size,
            )
            tree = self.config.tree_class(
                data=data_boot,
                targets=targets_boot,
                features=feature_boot,
                config=tree_config,
            )
            self.forest.append(tree)

    def predict(self, sample: np.ndarray) -> Any:
        predictions = [t.predict(sample) for t in self.forest]
        return Counter(predictions).most_common(1)[0][0]
