from dataclasses import dataclass

from src.tree.eval_func import CARTEvalEnum, ID3EvalEnum


@dataclass
class ID3Config:
    eval_function: ID3EvalEnum = ID3EvalEnum.INFORMATION_GAIN
    max_depth: int = 5
    tournament_size: int = 2


@dataclass
class CARTConfig:
    max_depth: int = 5
    min_samples_split: int = 2
    tournament_size: int = 2
    eval_function: CARTEvalEnum = CARTEvalEnum.CART_GINI_GAIN
