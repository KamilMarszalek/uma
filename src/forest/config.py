from dataclasses import dataclass

from src.tree.eval_func import EvalEnum


@dataclass
class TournamentForestConfig:
    num_of_trees: int
    sample_ratio: float
    feature_ratio: float
    tree_class: type
    tree_config_class: type
    max_depth: int = 5
    tournament_size: int = 2
    eval_function: EvalEnum = EvalEnum.ID3_INFORMATION_GAIN
