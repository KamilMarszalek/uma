import numpy as np
import pandas as pd

from src.data.uci_data_provider import get_uci_data
from src.forest.config import TournamentForestConfig
from src.forest.forest import TournamentForest
from src.tree.config import CARTConfig
from src.tree.eval_func import CARTGiniGain
from src.tree.tree import CARTTree

TRAIN_SIZE = 0.6
RANDOM_SEED = 42


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    encoded = df.copy()
    for col in encoded.columns:
        encoded[col] = encoded[col].astype("category").cat.codes
    return encoded


def encode_targets(targets: pd.DataFrame | pd.Series) -> np.ndarray:
    if isinstance(targets, pd.DataFrame):
        series = targets.iloc[:, 0]
    else:
        series = targets
    return series.astype("category").cat.codes.to_numpy()


def main() -> None:
    data, targets = get_uci_data(2)

    data = encode_categorical(data)
    target_codes = encode_targets(targets)

    data_np = data.to_numpy()
    n_samples = data_np.shape[0]

    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.permutation(n_samples)

    train_size = int(TRAIN_SIZE * n_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_data = data_np[train_indices]
    train_targets = target_codes[train_indices]
    test_data = data_np[test_indices]
    test_targets = target_codes[test_indices]

    config = TournamentForestConfig(
        num_of_trees=50,
        sample_ratio=0.8,
        feature_ratio=np.sqrt(data_np.shape[1]) / data_np.shape[1],
        eval_function=CARTGiniGain(),
        max_depth=10,
        tournament_size=2,
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
