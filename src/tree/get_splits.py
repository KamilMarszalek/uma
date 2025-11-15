import numpy as np


def get_splits(
    data: np.ndarray, targets: np.ndarray, feature: int
) -> dict[int, np.ndarray]:
    col = data[:, feature]
    values = np.unique(col)
    splits: dict[int, np.ndarray] = {}
    for value in values:
        mask = col == value
        splits[int(value)] = targets[mask]
    return splits
