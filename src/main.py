import numpy as np

from src.data.data_encoding import prepare_data_one_hot
from src.forest.config import TournamentForestConfig
from src.forest.forest import TournamentForest
from src.tree.config import CARTConfig
from src.tree.eval_func import CARTGiniGain
from src.tree.tree import CARTTree

TRAIN_SIZE = 0.7
RANDOM_SEED = 42


def main() -> None:
    train_data, test_data, train_targets, test_targets = prepare_data_one_hot(
        73, TRAIN_SIZE, RANDOM_SEED
    )

    n_features = train_data.shape[1]
    print(f"Number of features: {n_features}")

    config = TournamentForestConfig(
        num_of_trees=25,
        sample_ratio=0.8,
        feature_ratio=0.8,
        eval_function=CARTGiniGain(),
        max_depth=10,
        tournament_size=np.sqrt(n_features).astype(int),
        tree_class=CARTTree,
        tree_config_class=CARTConfig,
    )

    forest = TournamentForest(train_data, train_targets, config)
    forest.build()

    correct = 0
    for x, y_true in zip(test_data, test_targets, strict=True):
        y_pred = forest.predict(x)
        if y_pred == y_true:
            correct += 1

    accuracy = correct / test_data.shape[0]
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
