import numpy as np
import pandas as pd
from experiments.logger.logger import logger
from ucimlrepo import fetch_ucirepo

from src.data.encoders import (
    CatEncodingStrategy,
    encode_targets,
)
from src.data.train_test_split import train_test_split


def get_uci_data(  # noqa: PLR0913
    set_id: int = 73,
    train_size: float = 0.7,
    random_seed: int = 42,
    encode: CatEncodingStrategy = CatEncodingStrategy.CATEGORICAL,
    data: pd.DataFrame | None = None,
    targets: pd.DataFrame | pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if data is None or targets is None:
        data, targets = download_uci_data(set_id)

    cat_cols = data.select_dtypes(include=["object", "category"]).columns
    num_cols = data.columns.difference(cat_cols)
    data_cat = encode(data[cat_cols])
    data_encoded = pd.concat([data[num_cols], data_cat], axis=1)

    targets_encoded = encode_targets(targets)

    data_train, data_test, targets_train, targets_test = train_test_split(
        data_encoded,
        targets_encoded,
        train_size=train_size,
        random_seed=random_seed,
        stratify=targets_encoded,
    )

    return (
        data_train.to_numpy(),
        data_test.to_numpy(),
        targets_train.to_numpy(),
        targets_test.to_numpy(),
    )


def download_uci_data(set_id: int = 73) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = fetch_ucirepo(id=set_id)
    X = dataset.data.features
    Y = dataset.data.targets

    logger.info(f"Downloaded UCI dataset with set_id={set_id}")

    return X, Y
