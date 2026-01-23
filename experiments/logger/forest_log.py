from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from experiments.experiment.experiment_config import ExperimentConfig
from src.data.encoders import CatEncodingStrategy


@dataclass
class ForestLog:
    experiment: str
    forest_type: Literal["CART", "ID3"]
    eval_function: Literal[
        "ID3_INFORMATION_GAIN", "ID3_GAIN_RATIO", "ID3_GINI_GAIN", "CART_GINI_GAIN"
    ]
    num_trees: int
    sample_ratio: float
    feature_ratio: float
    tree_max_depth: int
    tree_tournament_size: int
    set_id: int
    train_size: float
    random_seed: int
    categorial_encoding: Literal["ONE_HOT", "CATEGORICAL"]
    time_of_building: float
    accuracy: float
    TP: int
    TN: int
    FP: int
    FN: int


def cat_endoding_to_string(strategy: CatEncodingStrategy) -> str:
    if strategy == CatEncodingStrategy.CATEGORICAL:
        return "CATEGORICAL"
    elif strategy == CatEncodingStrategy.ONE_HOT:
        return "ONE_HOT"
    else:
        raise ValueError(f"Unknown CatEncodingStrategy: {strategy}")


def convert_config_to_log(
    config: ExperimentConfig,
    time_of_building: float,
    accuracy: float,
    confusion_matrix: np.ndarray | None = None,
) -> ForestLog:
    return ForestLog(
        experiment=config.experiment_name,
        forest_type=cast(Literal["CART", "ID3"], config.forest_config.tree_class.name),
        eval_function=cast(
            Literal[
                "ID3_INFORMATION_GAIN",
                "ID3_GAIN_RATIO",
                "ID3_GINI_GAIN",
                "CART_GINI_GAIN",
            ],
            config.forest_config.eval_function.name,
        ),
        num_trees=config.forest_config.num_of_trees,
        sample_ratio=config.forest_config.sample_ratio,
        feature_ratio=config.forest_config.feature_ratio,
        tree_max_depth=config.forest_config.max_depth
        if config.forest_config.max_depth is not None
        else -1,
        tree_tournament_size=config.forest_config.tournament_size,
        set_id=config.set_id,
        train_size=config.train_size,
        random_seed=config.forest_config.random_seed,
        categorial_encoding=cast(
            Literal["ONE_HOT", "CATEGORICAL"],
            cat_endoding_to_string(config.categorial_encoding),
        ),
        time_of_building=round(time_of_building, 4),
        accuracy=round(accuracy, 4),
        TP=confusion_matrix[1, 1] if confusion_matrix is not None else 0,
        TN=confusion_matrix[0, 0] if confusion_matrix is not None else 0,
        FP=confusion_matrix[0, 1] if confusion_matrix is not None else 0,
        FN=confusion_matrix[1, 0] if confusion_matrix is not None else 0,
    )
