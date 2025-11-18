from dataclasses import dataclass

from src.data.encoders import CatEncodingStrategy
from src.forest.config import TournamentForestConfig


@dataclass
class ExperimentConfig:
    experiment_name: str
    set_id: int
    train_size: float
    random_seed: int
    categorial_encoding: CatEncodingStrategy
    forest_config: TournamentForestConfig
