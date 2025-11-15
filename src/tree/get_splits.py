import numpy as np


def get_splits(
    data: np.ndarray, targets: np.ndarray, feature: int
) -> dict[int, np.ndarray]:
    values = np.unique(data[:, feature])
    splits = {}
    for value in values:
        subset_indices = data[:, feature] == value
        splits[value] = targets[subset_indices]
    return splits
