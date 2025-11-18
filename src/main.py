from experiments.experiment_config import ExperimentConfig
from experiments.perform_experiment import perform_experiment

from src.data.encoders import CatEncodingStrategy
from src.data.uci_data_provider import download_uci_data
from src.forest.config import TournamentForestConfig
from src.tree.config import TreeConfig
from src.tree.eval_func import EvalFunction
from src.tree.tree import TreeClass

TRAIN_SIZE = 0.7
RANDOM_SEED = 42


def main() -> None:
    data, targets = download_uci_data(set_id=2)

    forest_config = TournamentForestConfig(
        num_of_trees=15,
        sample_ratio=0.8,
        # feature_ratio=np.sqrt(data_np.shape[1]) / data_np.shape[1],
        feature_ratio=0.8,
        eval_function=EvalFunction.CART_GINI_GAIN,
        max_depth=10,
        tournament_size=4,
        tree_class=TreeClass.CART,
        tree_config_class=TreeConfig.CART,
    )

    experiment_config = ExperimentConfig(
        experiment_name="Adult_Census_CART_Test_Experiment",
        set_id=2,
        train_size=TRAIN_SIZE,
        random_seed=RANDOM_SEED,
        categorial_encoding=CatEncodingStrategy.CATEGORICAL,
        forest_config=forest_config,
    )

    perform_experiment(
        config=experiment_config,
        data=data,
        targets=targets,
    )


if __name__ == "__main__":
    main()
