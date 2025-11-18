from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.tree.eval_func import CARTEvalFunc, EvalFunction, ID3EvalFunc


@dataclass
class ID3Config:
    eval_function: ID3EvalFunc = EvalFunction.ID3_INFORMATION_GAIN
    max_depth: int = 5
    tournament_size: int = 2


@dataclass
class CARTConfig:
    max_depth: int = 5
    min_samples_split: int = 2
    tournament_size: int = 2
    eval_function: CARTEvalFunc = EvalFunction.CART_GINI_GAIN


class TreeConfig(Enum):
    ID3 = ID3Config
    CART = CARTConfig

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.value(*args, **kwargs)
