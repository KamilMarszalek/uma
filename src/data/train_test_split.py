import numpy as np
import pandas as pd


def train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.7,
    random_seed: int = 42,
    stratify: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(random_seed)

    def divide_indices(n_samples: int, n_train: int) -> tuple[np.ndarray, np.ndarray]:
        indices = rng.permutation(n_samples)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        return train_indices, test_indices

    if stratify is None:
        n_samples = X.shape[0]
        n_train = int(train_size * n_samples)
        train_indices, test_indices = divide_indices(n_samples, n_train)
    else:
        train_indices = []
        test_indices = []
        stratify_values = stratify.unique()
        for cls in stratify_values:
            class_indices = stratify[stratify == cls].index.to_numpy()
            n_class_samples = class_indices.shape[0]
            n_class_train = int(train_size * n_class_samples)
            class_train_indices, class_test_indices = divide_indices(
                n_class_samples, n_class_train
            )
            train_indices.extend(class_indices[class_train_indices])
            test_indices.extend(class_indices[class_test_indices])
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        rng.shuffle(train_indices)
        rng.shuffle(test_indices)

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    return X_train, X_test, y_train, y_test
