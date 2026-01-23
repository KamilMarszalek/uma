import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
from experiments.logger.logger import logger

from src.forest.config import TournamentForestConfig
from src.tree.cart_tree import CARTTree
from src.tree.id3_tree import ID3Tree


def _build_single_tree(
    args: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> Any:
    data, targets, config, seed = args
    rng = np.random.default_rng(seed)
    rows, cols = data.shape

    n = int(rows * config.sample_ratio)
    indices = rng.choice(rows, size=n, replace=True)
    data_boot = data[indices]
    targets_boot = targets[indices]

    k = max(1, int(cols * config.feature_ratio))
    feature_boot = list(rng.choice(cols, size=k, replace=False))

    tree_config = config.tree_config_class(
        max_depth=config.max_depth,
        eval_function=config.eval_function,
        tournament_size=config.tournament_size,
    )

    tree = config.tree_class(
        config=tree_config,
    )
    tree.fit(
        data=data_boot,
        targets=targets_boot,
        features=feature_boot,
    )

    logger.info("Built tree with seed %d", seed)

    return tree


class TournamentForest:
    def __init__(
        self,
        config: TournamentForestConfig,
    ) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self.forest: list[ID3Tree | CARTTree] = []

    def fit(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        n_jobs: int | None = None,
    ) -> None:
        if n_jobs is None:
            n_jobs = os.cpu_count() or 1

        seeds = self.rng.integers(0, 2**32 - 1, size=self.config.num_of_trees)

        args = [(data, targets, self.config, int(seed)) for seed in seeds]

        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            self.forest = list(ex.map(_build_single_tree, args))

    def predict(self, sample: np.ndarray) -> Any:
        predictions = [t.predict(sample) for t in self.forest]
        return Counter(predictions).most_common(1)[0][0]
