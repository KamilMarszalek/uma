from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.tree.eval_func import (
    CARTEvalFunc,
    EvalFunction,
    ID3EvalFunc,
    SKLearnEvalFuncEnum,
)


@dataclass
class ID3Config:
    eval_function: ID3EvalFunc = EvalFunction.ID3_INFORMATION_GAIN
    max_depth: int = 5
    tournament_size: int = 2
    random_seed: int = 42

    @classmethod
    def from_forest_config(cls, forest_cfg: Any) -> "ID3Config":
        return cls(
            eval_function=forest_cfg.eval_function,
            max_depth=forest_cfg.max_depth,
            tournament_size=forest_cfg.tournament_size,
            random_seed=forest_cfg.random_seed,
        )


@dataclass
class CARTConfig:
    max_depth: int = 5
    min_samples_split: int = 2
    tournament_size: int = 2
    eval_function: CARTEvalFunc = EvalFunction.CART_GINI_GAIN
    random_seed: int = 42

    @classmethod
    def from_forest_config(cls, forest_cfg: Any) -> "CARTConfig":
        return cls(
            max_depth=forest_cfg.max_depth,
            tournament_size=forest_cfg.tournament_size,
            eval_function=forest_cfg.eval_function,
            random_seed=forest_cfg.random_seed,
            min_samples_split=forest_cfg.min_samples_split,
        )


@dataclass
class SKLearnTreeConfig:
    max_depth: int = 5
    eval_function: SKLearnEvalFuncEnum = SKLearnEvalFuncEnum.SKLEARN_ENTROPY
    random_seed: int = 42

    @classmethod
    def from_forest_config(cls, forest_cfg: Any) -> "SKLearnTreeConfig":
        return cls(
            max_depth=forest_cfg.max_depth,
            eval_function=forest_cfg.eval_function,
            random_seed=forest_cfg.random_seed,
        )


class TreeConfig(Enum):
    ID3 = ID3Config
    CART = CARTConfig
    SKLEARN = SKLearnTreeConfig

    def __call__(self, forest_config: Any) -> Any:
        return self.value.from_forest_config(forest_config)
