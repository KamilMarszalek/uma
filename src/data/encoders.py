from enum import Enum

import pandas as pd


def encode_targets(targets: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(targets, pd.DataFrame):
        series = targets.iloc[:, 0]
    else:
        series = targets
    return series.astype("category").cat.codes


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    encoded = df.copy()
    for col in encoded.columns:
        encoded[col] = encoded[col].astype("category").cat.codes
    return encoded


def encode_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or df.shape[1] == 0:
        return df.copy()
    return pd.get_dummies(df, prefix=df.columns, columns=df.columns, dtype=int)


class CatEncodingStrategy(Enum):
    CATEGORICAL = "categorical"
    ONE_HOT = "one_hot"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if self is CatEncodingStrategy.CATEGORICAL:
            return encode_categorical(df)
        if self is CatEncodingStrategy.ONE_HOT:
            return encode_one_hot(df)
