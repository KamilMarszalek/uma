import copy
import csv

import pandas as pd
from experiments.experiment.experiment_config import ExperimentConfig
from experiments.experiment.perform_experiment import perform_experiment
from src.data.encoders import CatEncodingStrategy
from src.data.uci_data_provider import download_uci_data
from src.forest.config import TournamentForestConfig
from src.tree.config import TreeConfig
from src.tree.eval_func import EvalFunction
from src.tree.tree_class import TreeClass


class ExperimentParser:
    def __init__(self, source_file: str):
        self.source_file: str = source_file
        self.configs: list[ExperimentConfig] = []
        self.parse()
        self.sort_configs()

    def parse(self) -> None:
        with open(self.source_file, newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                experiment_config = ExperimentConfig(
                    experiment_name=row["experiment"],
                    set_id=int(row["set_id"]),
                    train_size=float(row["train_size"]),
                    categorial_encoding=CatEncodingStrategy[row["categorial_encoding"]],
                    forest_config=TournamentForestConfig(
                        num_of_trees=int(row["num_trees"]),
                        sample_ratio=float(row["sample_ratio"]),
                        feature_ratio=float(row["feature_ratio"]),
                        eval_function=EvalFunction[row["eval_function"]],
                        max_depth=int(row["tree_max_depth"]),
                        tournament_size=int(row["tree_tournament_size"]),
                        min_samples_split=int(row["min_samples_split"]),
                        tree_class=TreeClass[row["forest_type"]],
                        tree_config_class=TreeConfig[row["forest_type"]],
                        random_seed=int(row["base_random_seed"]),
                    ),
                )
                times_repeat = int(row.get("times_repeat", 1))

                # No max depth is represented by non-positive values
                if (
                    experiment_config.forest_config.max_depth
                    and experiment_config.forest_config.max_depth <= 0
                ):
                    experiment_config.forest_config.max_depth = None

                for i in range(times_repeat):
                    cfg = copy.deepcopy(experiment_config)
                    cfg.forest_config.random_seed = (
                        experiment_config.forest_config.random_seed + i
                    )
                    self.configs.append(cfg)

    def sort_configs(self) -> None:
        self.configs.sort(key=lambda config: config.set_id)

    def perform_experiments(self) -> None:
        data: pd.DataFrame | None = None
        targets: pd.DataFrame | pd.Series | None = None
        previous_set_id = -1
        for config in self.configs:
            if config.set_id != previous_set_id:
                data, targets = download_uci_data(set_id=config.set_id)
                previous_set_id = config.set_id

            perform_experiment(config=config, data=data, targets=targets)
