import numpy as np
import pandas as pd
from experiments.experiment.experiment_config import ExperimentConfig
from experiments.logger.forest_log import convert_config_to_log
from experiments.logger.logger import logger, setup_experiment_csv
from experiments.timer import Timer
from src.data.uci_data_provider import get_uci_data
from src.forest.forest import TournamentForest

accuracy_type = float
time_type = float


def perform_experiment(
    config: ExperimentConfig,
    data: pd.DataFrame | None = None,
    targets: pd.DataFrame | pd.Series | None = None,
) -> tuple[accuracy_type, time_type]:
    train_data, test_data, train_targets, test_targets = get_uci_data(
        set_id=config.set_id,
        train_size=config.train_size,
        random_seed=config.forest_config.random_seed,
        encode=config.categorial_encoding,
        data=data,
        targets=targets,
    )

    forest = TournamentForest(config.forest_config)

    timer = Timer(forest.fit)
    timer.run(train_data, train_targets)
    time_of_building = timer.get_elapsed()

    y_pred_all = forest.predict(test_data)
    y_true = np.asarray(test_targets).ravel()
    y_pred = np.asarray(y_pred_all).ravel()
    correct = np.sum(y_pred == y_true)
    accuracy = correct / y_true.size

    labels = np.unique(np.concatenate([y_true, y_pred]))
    y_true_idx = np.searchsorted(labels, y_true)
    y_pred_idx = np.searchsorted(labels, y_pred)
    conf_matrix = np.zeros((labels.size, labels.size), dtype=int)
    np.add.at(conf_matrix, (y_true_idx, y_pred_idx), 1)

    logger.info(
        f"Experiment {config.experiment_name} completed: "
        f"Accuracy={accuracy:.4f}, "
        f"Time of building={time_of_building:.4f} seconds."
    )

    setup_experiment_csv(config.experiment_name)
    logger.data_trace(
        convert_config_to_log(
            config=config,
            time_of_building=time_of_building,
            accuracy=accuracy,
            confusion_matrix=conf_matrix,
        )
    )

    return accuracy, time_of_building
