import re
from pathlib import Path

import numpy as np
import pandas as pd
from experiments.experiment.metrics import f1_score, precision, recall

PARAM_COLUMN_MAP = {
    "tree_count": "num_trees",
    "tournament_size": "tree_tournament_size",
    "sample_ratio": "sample_ratio",
    "feature_ratio": "feature_ratio",
    "max_depth": "tree_max_depth",
    "min_samples_split": "min_samples_split",
    "eval_function": "eval_function",
}

COLUMN_NAME_MAP = {
    "tree_count": "Liczba drzew",
    "tournament_size": "Rozmiar turnieju",
    "sample_ratio": "Proporcja próbkowania",
    "feature_ratio": "Proporcja cech",
    "max_depth": "Maksymalna głębokość",
    "min_samples_split": "Minimalna liczba próbek do podziału",
    "eval_function": "Funkcja oceny",
    "mean_accuracy": "Średnia",
    "std_accuracy": "Std",
    "max_accuracy": "Maks",
    "min_accuracy": "Min",
    "mean_precision": "Średnia",
    "std_precision": "Std",
    "max_precision": "Maks",
    "min_precision": "Min",
    "mean_recall": "Średnia",
    "std_recall": "Std",
    "max_recall": "Maks",
    "min_recall": "Min",
    "mean_f1": "Średnia",
    "std_f1": "Std",
    "max_f1": "Maks",
    "min_f1": "Min",
    "mean_time_of_building": "Średnia",
    "std_time_of_building": "Std",
    "max_time_of_building": "Maks",
    "min_time_of_building": "Min",
}

METRIC_LABELS = {
    "accuracy": "Dokładność",
    "precision": "Precyzja",
    "recall": "Czułość",
    "f1": "F1",
    "time_of_building": "Czas trenowania",
}

_METRIC_FUNCTIONS = {
    "precision": lambda tp, fp, _: precision(tp, fp),
    "recall": lambda tp, _, fn: recall(tp, fn),
    "f1": f1_score,
}


def extract_param_key(stem: str) -> str | None:
    stem = re.sub(r"_\d+$", "", stem)
    for prefix in ("CART_", "ID3_", "SKLEARN_"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return None


def get_param_info(csv_path: Path, df: pd.DataFrame) -> tuple[str, str] | None:
    param_key = extract_param_key(csv_path.stem)
    if not param_key:
        return None

    param_column = PARAM_COLUMN_MAP.get(param_key, param_key)
    if param_column not in df.columns:
        return None

    return param_key, param_column


def build_title_from_path(csv_path: Path, metric_key: str | None = None) -> str:
    param_key = extract_param_key(csv_path.stem)
    if not param_key:
        return csv_path.stem
    tree_variant = csv_path.stem.split("_")[0]
    dataset_id = csv_path.stem.split("_")[-1]
    readable_key = COLUMN_NAME_MAP.get(param_key, param_key)
    return f"{readable_key} na zbiorze {dataset_id} ({tree_variant})"


def build_metric_summary(
    csv_path: Path,
    metric_key: str,
    *,
    include_extremes: bool = True,
) -> pd.DataFrame | None:
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    return build_metric_summary_from_df(
        df,
        csv_path,
        metric_key,
        include_extremes=include_extremes,
    )


def build_metric_summary_from_df(
    df: pd.DataFrame,
    csv_path: Path,
    metric_key: str,
    *,
    include_extremes: bool = True,
) -> pd.DataFrame | None:
    param_info = get_param_info(csv_path, df)
    if not param_info:
        return None
    param_key, param_column = param_info

    values = _compute_metric_values(df, metric_key)
    if values is None:
        return None

    summary = _aggregate_metric(
        df,
        param_column,
        metric_key,
        values,
        include_extremes=include_extremes,
    )
    if summary is None:
        return None
    return summary.rename(columns={param_column: param_key})


def _compute_metric_values(
    df: pd.DataFrame,
    metric_key: str,
) -> pd.Series | None:
    values: pd.Series | None = None
    if metric_key == "accuracy":
        if "accuracy" in df.columns:
            values = pd.to_numeric(df["accuracy"], errors="coerce")
    elif metric_key == "time_of_building":
        if "time_of_building" in df.columns:
            values = pd.to_numeric(df["time_of_building"], errors="coerce")
    else:
        required_columns = {"TP", "TN", "FP", "FN"}
        if required_columns.issubset(df.columns):
            metric_func = _METRIC_FUNCTIONS.get(metric_key)
            if metric_func is not None:
                tp = (
                    pd.to_numeric(df["TP"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                    .to_numpy()
                )
                fp = (
                    pd.to_numeric(df["FP"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                    .to_numpy()
                )
                fn = (
                    pd.to_numeric(df["FN"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                    .to_numpy()
                )
                metric_values = np.fromiter(
                    (
                        metric_func(int(tp_i), int(fp_i), int(fn_i))
                        for tp_i, fp_i, fn_i in zip(tp, fp, fn, strict=False)
                    ),
                    dtype=float,
                    count=len(tp),
                )
                values = pd.Series(metric_values, index=df.index)
    return values


def _aggregate_metric(
    df: pd.DataFrame,
    param_column: str,
    metric_key: str,
    values: pd.Series,
    *,
    include_extremes: bool,
) -> pd.DataFrame | None:
    metric_df = pd.DataFrame(
        {
            param_column: df[param_column],
            metric_key: values,
        }
    ).dropna(subset=[metric_key])
    if metric_df.empty:
        return None

    aggregations: dict[str, str] = {
        f"mean_{metric_key}": "mean",
        f"std_{metric_key}": "std",
    }
    if include_extremes:
        aggregations[f"max_{metric_key}"] = "max"
        aggregations[f"min_{metric_key}"] = "min"

    return (
        metric_df.groupby(param_column, dropna=False)[metric_key]
        .agg(**aggregations)
        .reset_index()
        .sort_values(param_column)
    )
