from dataclasses import dataclass

from src.forest.config import TournamentForestConfig


@dataclass
class ExperimentConfig:
    experiment_name: str
    set_id: int
    train_size: float
    random_seed: int
    categorial_encoding: str
    forest_config: TournamentForestConfig
