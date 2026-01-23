from dataclasses import dataclass

from src.tree.config import TreeConfig
from src.tree.eval_func import EvalFunction
from src.tree.tree_class import TreeClass


@dataclass
class TournamentForestConfig:
    num_of_trees: int
    sample_ratio: float
    feature_ratio: float
    tree_class: TreeClass
    tree_config_class: TreeConfig
    max_depth: int | None = None
    tournament_size: int = 2
    eval_function: EvalFunction = EvalFunction.ID3_INFORMATION_GAIN
    random_seed: int = 42
