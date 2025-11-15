from dataclasses import dataclass

from src.tree.eval_func import (
    CARTEvalFunc,
    CARTGiniGain,
    ID3EvalFunc,
    InformationGain,
)


@dataclass
class ID3Config:
    eval_function: ID3EvalFunc = InformationGain()
    max_depth: int = 5
    tournament_size: int = 2


@dataclass
class CARTConfig:
    max_depth: int = 5
    min_samples_split: int = 2
    tournament_size: int = 2
    eval_function: CARTEvalFunc = CARTGiniGain()
