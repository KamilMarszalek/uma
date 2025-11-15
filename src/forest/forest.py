from collections import Counter
from typing import Any

import numpy as np

from src.forest.config import TournamentForestConfig
from src.tree.config import TreeConfig
from src.tree.tree import Tree


class TournamentForest:
    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        config: TournamentForestConfig,
    ) -> None:
        self.eval_function = config.eval_function
        self.data = data
        self.targets = targets
        self.num_of_trees = config.num_of_trees
        self.sample_ratio = config.sample_ratio
        self.feature_ratio = config.feature_ratio
        self.max_depth = config.max_depth
        self.tournament_size = config.tournament_size

        self.rng = np.random.default_rng()
        self.forest: list[Tree] = []

    def build(self) -> None:
        num_of_rows = self.data.shape[0]
        total_features = self.data.shape[1]
        for _ in range(self.num_of_trees):
            sample_size = int(num_of_rows * self.sample_ratio)
            sample_indices = self.rng.choice(
                num_of_rows,
                size=sample_size,
                replace=True,
            )
            data_boot = self.data[sample_indices]
            targets_boot = self.targets[sample_indices]

            feature_boot_size = max(
                1,
                int(total_features * self.feature_ratio),
            )
            feature_boot = list(
                self.rng.choice(
                    total_features,
                    feature_boot_size,
                    replace=False,
                )
            )
            tree_config = TreeConfig()
            tree = Tree(
                data=data_boot,
                targets=targets_boot,
                features=feature_boot,
                config=tree_config,
            )
            self.forest.append(tree)

    def predict(self, sample: np.ndarray) -> Any:
        predictions = [tree.predict(sample) for tree in self.forest]
        return Counter(predictions).most_common(1)[0][0]
