from dataclasses import dataclass

from pyparsing import Literal

from src.forest.config import TournamentForestConfig


@dataclass
class ForestLog:
    experiment: str
    forest_type: Literal["CART", "ID3"]
    eval_function: Literal["ID3_INFORMATION_GAIN", "ID3_GAIN_RATIO", "ID3_GINI_GAIN", "CART_GINI_GAIN"]
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


def convert_config_to_log(
    experiment: str,
    config: TournamentForestConfig,
    set_id: int,
    train_size: float,
    random_seed: int,
    categorial_encoding: str,
    time_of_building: float,
    accuracy: float,
) -> ForestLog:
    return ForestLog(
        experiment=experiment,
        forest_type=config.tree_class.name,
        eval_function=config.eval_function.name,
        num_trees=config.num_of_trees,
        sample_ratio=config.sample_ratio,
        feature_ratio=config.feature_ratio,
        tree_max_depth=config.max_depth,
        tree_tournament_size=config.tournament_size,
        set_id=set_id,
        train_size=train_size,
        random_seed=random_seed,
        categorial_encoding=categorial_encoding,
        time_of_building=time_of_building,
        accuracy=accuracy,
    )