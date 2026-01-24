import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from experiments.experiment import summary_utils as summaries


@dataclass(frozen=True, slots=True)
class TableBuilderOptions:
    caption: str | None = None
    label: str | None = None
    column_lines: bool = True
    header_hline: bool = True
    right_align_cells: list[tuple[int, str]] | None = None
    bold_rows: list[int] | None = None
    float_format: str = "%.2f"
    decimal: str = ","
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
        self.decimal = opts.decimal
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
            formatted = self.float_format % value
            if self.decimal != ".":
                formatted = formatted.replace(".", self.decimal)
            return formatted
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
            decimal=self.decimal,
            formatters=formatters,
        )
        latex_str = self._inject_table_style(latex_str)
        latex_str = self._inject_header_hline(latex_str)
        with open(output_path, "w") as f:
            f.write(latex_str)


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
        summary = summaries.build_metric_summary(
            csv_path,
            "accuracy",
            include_extremes=True,
        )
        if summary is None:
            continue
        bold_rows = _bold_rows_for_mean(summary, "mean_accuracy")
        summary = summary.rename(columns=summaries.COLUMN_NAME_MAP)
        table_config = TableBuilderOptions(
            caption=summaries.build_title_from_path(csv_path),
            label=f"tab:{csv_path.stem}",
            column_lines=True,
            header_hline=True,
            bold_rows=bold_rows,
            float_format="%.2f",
        )
        table = TableBuilder(summary, table_config)
        table.to_latex(output_dir / f"{csv_path.stem}.tex")

        time_summary = summaries.build_metric_summary(
            csv_path,
            "time_of_building",
            include_extremes=True,
        )
        if time_summary is not None:
            time_bold_rows = _bold_rows_for_mean(
                time_summary,
                "mean_time_of_building",
                max_or_min="min",
            )
            time_summary = time_summary.rename(columns=summaries.COLUMN_NAME_MAP)
            time_title = summaries.build_title_from_path(csv_path)
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
            metric_summary = summaries.build_metric_summary(
                csv_path,
                metric_key,
                include_extremes=True,
            )
            if metric_summary is None:
                continue
            metric_bold_rows = _bold_rows_for_mean(
                metric_summary,
                f"mean_{metric_key}",
            )
            metric_summary = metric_summary.rename(columns=summaries.COLUMN_NAME_MAP)
            metric_title = summaries.build_title_from_path(csv_path)
            metric_table_config = TableBuilderOptions(
                caption=f"{summaries.METRIC_LABELS[metric_key]} - {metric_title}",
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
