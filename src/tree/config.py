from dataclasses import dataclass

from src.tree.eval_func import EvalFunc, InformationGain


@dataclass
class TreeConfig:
    eval_function: EvalFunc = InformationGain()
    max_depth: int = 5
    tournament_size: int = 2
