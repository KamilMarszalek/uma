from dataclasses import dataclass

from src.tree.eval_func import EvalFunc, InformationGain


@dataclass
class TournamentForestConfig:
    num_of_trees: int
    sample_ratio: float
    feature_ratio: float
    max_depth: int = 5
    tournament_size: int = 2
    eval_function: EvalFunc = InformationGain()
