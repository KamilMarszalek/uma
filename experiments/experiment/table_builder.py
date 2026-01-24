import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from experiments.experiment.metrics import f1_score, precision, recall


@dataclass(frozen=True, slots=True)
class TableBuilderOptions:
    caption: str | None = None
    label: str | None = None
    column_lines: bool = True
    header_hline: bool = True
    right_align_cells: list[tuple[int, str]] | None = None
    bold_rows: list[int] | None = None
    float_format: str = "%.2f"
    escape: bool | None = True


class TableBuilder:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        options: TableBuilderOptions | None = None,
    ) -> None:
        opts = options or TableBuilderOptions()
        self.dataframe = dataframe
        self.caption = opts.caption
        self.label = opts.label
        self.column_lines = opts.column_lines
        self.header_hline = opts.header_hline
        self.right_align_cells = opts.right_align_cells or []
        self.bold_rows = opts.bold_rows or []
        self.float_format = opts.float_format
        self.escape = opts.escape

    def _column_format(self) -> str:
        if self.dataframe.empty:
            return "l"
        alignments = ["r"] * len(self.dataframe.columns)
        if self.column_lines:
            return "|".join(alignments)
        return "".join(alignments)

    @staticmethod
    def _escape_underscores(value: object) -> object:
        if isinstance(value, str):
            return value.replace("_", "\\_")
        return value

    def _escape_caption(self) -> str | None:
        if not self.caption:
            return None
        return self.caption.replace("_", "\\_")

    def _format_cell_value(self, value: object) -> str:
        if pd.isna(value):
            return "NaN"
        if isinstance(value, (float, np.floating)):
            return self.float_format % value
        return str(value)

    def _prepare_dataframe(self) -> tuple[pd.DataFrame, bool]:
        df = self.dataframe.copy()
        needs_raw_latex = bool(self.right_align_cells or self.bold_rows)
        if not needs_raw_latex:
            return df, False

        df, rename_map, reverse_rename_map = self._rename_for_latex(df)
        df = self._escape_object_columns(df)
        self._apply_right_align(df, rename_map)
        self._apply_bold_rows(df, reverse_rename_map)
        return df, True

    def _rename_for_latex(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
        rename_map: dict[str, str] = {}
        for col in df.columns:
            if isinstance(col, str):
                escaped_col = self._escape_underscores(col)
                if escaped_col != col:
                    rename_map[col] = escaped_col
        if rename_map:
            df = df.rename(columns=rename_map)
        reverse_rename_map = {
            escaped: original for original, escaped in rename_map.items()
        }
        return df, rename_map, reverse_rename_map

    def _escape_object_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map(self._escape_underscores)
        return df

    def _apply_right_align(self, df: pd.DataFrame, rename_map: dict[str, str]) -> None:
        for row_idx, col_name in self.right_align_cells:
            escaped_col = rename_map.get(col_name, col_name)
            if escaped_col not in df.columns:
                raise KeyError(
                    f"Column not found for right alignment: {col_name}",
                )
            formatted_value = self._escape_underscores(
                self._format_cell_value(self.dataframe.at[row_idx, col_name])
            )
            df[escaped_col] = df[escaped_col].astype(object)
            df.at[row_idx, escaped_col] = (
                f"\\multicolumn{{1}}{{r}}{{{formatted_value}}}"
            )

    def _apply_bold_rows(
        self,
        df: pd.DataFrame,
        reverse_rename_map: dict[str, str],
    ) -> None:
        if not self.bold_rows:
            return
        for row_idx in self.bold_rows:
            if row_idx not in df.index:
                continue
            for col in df.columns:
                original_col = reverse_rename_map.get(col, col)
                formatted_value = self._escape_underscores(
                    self._format_cell_value(self.dataframe.at[row_idx, original_col])
                )
                df[col] = df[col].astype(object)
                df.at[row_idx, col] = f"\\textbf{{{formatted_value}}}"

    def _inject_table_style(self, latex_str: str) -> str:
        lines = latex_str.splitlines()
        if not lines or not lines[0].startswith("\\begin{table}"):
            return latex_str
        lines[1:1] = [
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{6pt}",
            "\\renewcommand{\\arraystretch}{1.2}",
        ]
        return "\n".join(lines)

    def _inject_header_hline(self, latex_str: str) -> str:
        if not self.header_hline:
            return latex_str
        lines = latex_str.splitlines()
        for idx, line in enumerate(lines):
            if line.strip() == "\\midrule":
                lines[idx] = "\\hline"
                break
        return "\n".join(lines)

    def to_latex(self, output_path: Path) -> None:
        latex_df, needs_raw_latex = self._prepare_dataframe()
        formatters = {col: self._format_cell_value for col in latex_df.columns}
        latex_str = latex_df.to_latex(
            index=False,
            column_format=self._column_format(),
            caption=self._escape_caption(),
            label=self.label,
            position="htbp",
            escape=False if needs_raw_latex else self.escape,
            decimal=",",
            formatters=formatters,
        )
        latex_str = self._inject_table_style(latex_str)
        latex_str = self._inject_header_hline(latex_str)
        with open(output_path, "w") as f:
            f.write(latex_str)


_PARAM_COLUMN_MAP = {
    "tree_count": "num_trees",
    "tournament_size": "tree_tournament_size",
    "sample_ratio": "sample_ratio",
    "max_depth": "tree_max_depth",
    "min_samples_split": "min_samples_split",
    "eval_function": "eval_function",
}

_COLUMN_NAME_MAP = {
    "tree_count": "Liczba drzew",
    "tournament_size": "Rozmiar turnieju",
    "sample_ratio": "Proporcja próbkowania",
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

_METRIC_LABELS = {
    "accuracy": "Dokładność",
    "precision": "Precyzja",
    "recall": "Czułość",
    "f1": "F1",
    "time_of_building": "Czas trenowania (s)",
}

_METRIC_FUNCTIONS = {
    "precision": lambda tp, fp, fn: precision(tp, fp),
    "recall": lambda tp, fp, fn: recall(tp, fn),
    "f1": f1_score,
}


def _extract_param_key(stem: str) -> str | None:
    stem = re.sub(r"_\d+$", "", stem)
    for prefix in ("CART_", "ID3_", "SKLEARN_"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return None


def _get_param_info(csv_path: Path, df: pd.DataFrame) -> tuple[str, str] | None:
    param_key = _extract_param_key(csv_path.stem)
    if not param_key:
        return None

    param_column = _PARAM_COLUMN_MAP.get(param_key, param_key)
    if param_column not in df.columns:
        return None

    return param_key, param_column


def _build_summary(csv_path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(csv_path)
    if df.empty or "accuracy" not in df.columns:
        return None

    param_info = _get_param_info(csv_path, df)
    if not param_info:
        return None
    param_key, param_column = param_info

    summary = (
        df.groupby(param_column, dropna=False)["accuracy"]
        .agg(
            mean_accuracy="mean",
            std_accuracy="std",
            max_accuracy="max",
            min_accuracy="min",
        )
        .reset_index()
        .sort_values(param_column)
        .rename(columns={param_column: param_key})
    )
    summary = summary.rename(columns=_COLUMN_NAME_MAP)
    return summary


def _build_time_summary(csv_path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(csv_path)
    if df.empty or "time_of_building" not in df.columns:
        return None

    param_info = _get_param_info(csv_path, df)
    if not param_info:
        return None
    param_key, param_column = param_info

    summary = (
        df.groupby(param_column, dropna=False)["time_of_building"]
        .agg(
            mean_time_of_building="mean",
            std_time_of_building="std",
            max_time_of_building="max",
            min_time_of_building="min",
        )
        .reset_index()
        .sort_values(param_column)
        .rename(columns={param_column: param_key})
    )
    summary = summary.rename(columns=_COLUMN_NAME_MAP)
    return summary


def _build_metric_summary(
    csv_path: Path,
    metric_key: str,
) -> pd.DataFrame | None:
    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    param_info = _get_param_info(csv_path, df)
    if not param_info:
        return None
    param_key, param_column = param_info

    required_columns = {"TP", "TN", "FP", "FN"}
    if not required_columns.issubset(df.columns):
        return None

    if metric_key not in _METRIC_FUNCTIONS:
        return None

    tp = pd.to_numeric(df["TP"], errors="coerce").fillna(0).astype(int).to_numpy()
    fp = pd.to_numeric(df["FP"], errors="coerce").fillna(0).astype(int).to_numpy()
    fn = pd.to_numeric(df["FN"], errors="coerce").fillna(0).astype(int).to_numpy()

    metric_func = _METRIC_FUNCTIONS[metric_key]
    metric_values = np.fromiter(
        (
            metric_func(int(tp_i), int(fp_i), int(fn_i))
            for tp_i, fp_i, fn_i in zip(tp, fp, fn, strict=False)
        ),
        dtype=float,
        count=len(tp),
    )
    df = df.copy()
    df[metric_key] = metric_values

    mean_col = f"mean_{metric_key}"
    std_col = f"std_{metric_key}"
    max_col = f"max_{metric_key}"
    min_col = f"min_{metric_key}"

    summary = (
        df.groupby(param_column, dropna=False)[metric_key]
        .agg(
            **{
                mean_col: "mean",
                std_col: "std",
                max_col: "max",
                min_col: "min",
            }
        )
        .reset_index()
        .sort_values(param_column)
        .rename(columns={param_column: param_key})
    )
    summary = summary.rename(columns=_COLUMN_NAME_MAP)
    return summary


def _bold_rows_for_mean(
    summary: pd.DataFrame, mean_column: str, max_or_min: str = "max"
) -> list[int]:
    if mean_column not in summary.columns:
        return []
    mean_series = summary[mean_column]
    if mean_series.dropna().empty:
        return []
    if max_or_min == "max":
        max_value = mean_series.max()
    else:
        max_value = mean_series.min()
    return mean_series[mean_series == max_value].index.tolist()


def _build_title_from_path(csv_path: Path, metric_key: str | None = None) -> str:
    param_key = _extract_param_key(csv_path.stem)
    if not param_key:
        return csv_path.stem
    tree_variant = csv_path.stem.split("_")[0]
    dataset_id = csv_path.stem.split("_")[-1]
    readable_key = _COLUMN_NAME_MAP.get(param_key, param_key)
    return f"{readable_key} na zbiorze {dataset_id} ({tree_variant})"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from experiment_output CSV files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("experiment_output"),
        help="Directory with experiment CSV outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for LaTeX tables (defaults to input dir).",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in sorted(input_dir.glob("*.csv")):
        summary = _build_summary(csv_path)
        if summary is None:
            continue
        bold_rows = _bold_rows_for_mean(summary, _COLUMN_NAME_MAP["mean_accuracy"])
        table_config = TableBuilderOptions(
            caption=_build_title_from_path(csv_path),
            label=f"tab:{csv_path.stem}",
            column_lines=True,
            header_hline=True,
            bold_rows=bold_rows,
            float_format="%.2f",
        )
        table = TableBuilder(summary, table_config)
        table.to_latex(output_dir / f"{csv_path.stem}.tex")

        time_summary = _build_time_summary(csv_path)
        if time_summary is not None:
            time_bold_rows = _bold_rows_for_mean(
                time_summary,
                _COLUMN_NAME_MAP["mean_time_of_building"],
                max_or_min="min",
            )
            time_title = _build_title_from_path(
                csv_path,
                metric_key="time_of_building",
            )
            time_table_config = TableBuilderOptions(
                caption=f"Czas trenowania - {time_title}",
                label=f"tab:{csv_path.stem}_time",
                column_lines=True,
                header_hline=True,
                bold_rows=time_bold_rows,
                float_format="%.2f",
            )
            time_table = TableBuilder(time_summary, time_table_config)
            time_table.to_latex(output_dir / f"{csv_path.stem}_time.tex")

        for metric_key in ("precision", "recall", "f1"):
            metric_summary = _build_metric_summary(csv_path, metric_key)
            if metric_summary is None:
                continue
            metric_bold_rows = _bold_rows_for_mean(
                metric_summary,
                _COLUMN_NAME_MAP[f"mean_{metric_key}"],
            )
            metric_title = _build_title_from_path(
                csv_path,
                metric_key=metric_key,
            )
            metric_table_config = TableBuilderOptions(
                caption=f"{_METRIC_LABELS[metric_key]} - {metric_title}",
                label=f"tab:{csv_path.stem}_{metric_key}",
                column_lines=True,
                header_hline=True,
                bold_rows=metric_bold_rows,
                float_format="%.2f",
            )
            metric_table = TableBuilder(metric_summary, metric_table_config)
            metric_table.to_latex(output_dir / f"{csv_path.stem}_{metric_key}.tex")


if __name__ == "__main__":
    main()
