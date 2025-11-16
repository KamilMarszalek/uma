import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from src.data.uci_data_provider import get_uci_data


def prepare_data_one_hot(
    set_id: int,
    train_size: float = 0.7,
    random_seed: int = 42,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    data, targets = get_uci_data(set_id)

    if isinstance(targets, pd.DataFrame):
        y = targets.iloc[:, 0]
    else:
        y = targets

    X_train_df, X_test_df, y_train_raw, y_test_raw = train_test_split(
        data,
        y,
        train_size=train_size,
        random_state=random_seed,
        stratify=y,
    )
    cat_cols = X_train_df.select_dtypes(include=["object", "category"]).columns
    num_cols = X_train_df.columns.difference(cat_cols)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                cat_cols,
            ),
            (
                "num",
                "passthrough",
                num_cols,
            ),
        ]
    )

    X_train_enc = preprocessor.fit_transform(X_train_df)
    X_test_enc = preprocessor.transform(X_test_df)

    X_train = X_train_enc.toarray().astype("float64")
    X_test = X_test_enc.toarray().astype("float64")

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    return X_train, X_test, y_train, y_test
