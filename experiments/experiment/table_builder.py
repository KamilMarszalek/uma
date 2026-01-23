from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class TableBuilderOptions:
    caption: str | None = None
    label: str | None = None
    column_lines: bool = True
    header_hline: bool = True
    right_align_cells: list[tuple[int, str]] | None = None
    float_format: str = "%.3f"
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
        needs_raw_latex = bool(self.right_align_cells)
        if not needs_raw_latex:
            return df, False

        rename_map = {}
        for col in df.columns:
            if isinstance(col, str):
                escaped_col = self._escape_underscores(col)
                if escaped_col != col:
                    rename_map[col] = escaped_col
        if rename_map:
            df = df.rename(columns=rename_map)

        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map(self._escape_underscores)

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

        return df, True

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
        latex_str = latex_df.to_latex(
            index=False,
            float_format=self.float_format,
            column_format=self._column_format(),
            caption=self._escape_caption(),
            label=self.label,
            position="htbp",
            escape=False if needs_raw_latex else self.escape,
            decimal=",",
        )
        latex_str = self._inject_table_style(latex_str)
        latex_str = self._inject_header_hline(latex_str)
        with open(output_path, "w") as f:
            f.write(latex_str)
