import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from experiments.experiment import summary_utils as summaries
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter


@dataclass(frozen=True, slots=True)
class PlotterOptions:
    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    output_path: str | None = None


@dataclass(frozen=True, slots=True)
class MetricPlotSpec:
    param_key: str
    param_label: str
    title: str
    output_path: Path


@dataclass(frozen=True, slots=True)
class CombinedPlotSpec:
    param_key: str
    param_label: str
    title: str
    output_path: Path
    time_summary: pd.DataFrame | None


def plot_line_chart(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    options: PlotterOptions | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data[x_col], data[y_col], marker="o")

    if options is not None:
        if options.title:
            ax.set_title(options.title)
        if options.x_label:
            ax.set_xlabel(options.x_label)
        if options.y_label:
            ax.set_ylabel(options.y_label)

    _apply_decimal_comma(ax)
    if pd.api.types.is_numeric_dtype(data[x_col]):
        _apply_decimal_comma(ax, axis="x")
    ax.grid(True)

    if options and options.output_path:
        fig.savefig(options.output_path)
    else:
        plt.show()

    plt.close(fig)


_METRIC_KEYS = ("accuracy", "precision", "recall", "f1")
_METRIC_COLORS = {
    "accuracy": "#27ae60",
    "precision": "#2980b9",
    "recall": "#f39c12",
    "f1": "#8e44ad",
}


def _format_decimal_label(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        text = f"{value:g}"
    else:
        text = str(value)
    return text.replace(".", ",")


_DECIMAL_COMMA_FORMATTER = FuncFormatter(
    lambda value, _: f"{value:g}".replace(".", ",")
)


def _apply_decimal_comma(ax: plt.Axes, axis: str = "y") -> None:
    if axis in {"y", "both"}:
        ax.yaxis.set_major_formatter(_DECIMAL_COMMA_FORMATTER)
    if axis in {"x", "both"}:
        ax.xaxis.set_major_formatter(_DECIMAL_COMMA_FORMATTER)


def _set_plot_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass


def _plot_metric(
    summary: pd.DataFrame,
    metric_key: str,
    spec: MetricPlotSpec,
) -> None:
    _set_plot_style()
    x_labels = summary[spec.param_key].map(_format_decimal_label)
    mean_col = f"mean_{metric_key}"
    std_col = f"std_{metric_key}"
    y_values = summary[mean_col].to_numpy()
    y_err = summary[std_col].fillna(0).to_numpy()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.errorbar(
        x_labels,
        y_values,
        yerr=y_err,
        fmt="-o",
        capsize=6,
        linewidth=2.5,
        color=_METRIC_COLORS.get(metric_key, "#2c3e50"),
        label=summaries.METRIC_LABELS.get(metric_key, metric_key),
    )
    _apply_decimal_comma(ax)
    ax.set_xlabel(f"Parametr: {spec.param_label}", fontsize=12)
    ax.set_ylabel("Wartość metryki", fontsize=12)
    ax.set_title(spec.title, fontsize=14, pad=20)
    ax.legend(loc="upper left", frameon=True, shadow=True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(spec.output_path, dpi=300)
    plt.close(fig)


def _plot_metrics_combined(
    metric_summaries: dict[str, pd.DataFrame],
    spec: CombinedPlotSpec,
) -> None:
    if "accuracy" not in metric_summaries:
        return

    base = metric_summaries["accuracy"][[spec.param_key]].copy()
    x_labels = base[spec.param_key].map(_format_decimal_label)

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    for metric_key in _METRIC_KEYS:
        summary = metric_summaries.get(metric_key)
        if summary is None:
            continue
        aligned = base.merge(summary, on=spec.param_key, how="left")
        mean_col = f"mean_{metric_key}"
        std_col = f"std_{metric_key}"
        y_values = aligned[mean_col].to_numpy()
        y_err = aligned[std_col].fillna(0).to_numpy()
        ax.errorbar(
            x_labels,
            y_values,
            yerr=y_err,
            fmt="-o",
            capsize=6,
            linewidth=2.5,
            color=_METRIC_COLORS.get(metric_key, "#2c3e50"),
            label=summaries.METRIC_LABELS.get(metric_key, metric_key),
        )

    _apply_decimal_comma(ax)
    ax.set_xlabel(f"Parametr: {spec.param_label}", fontsize=12)
    ax.set_ylabel("Wartość metryki", fontsize=12)
    ax.set_title(spec.title, fontsize=14, pad=20)

    if spec.time_summary is not None:
        aligned_time = base.merge(spec.time_summary, on=spec.param_key, how="left")
        ax2 = ax.twinx()
        ax2.errorbar(
            x_labels,
            aligned_time["mean_time_of_building"].to_numpy(),
            fmt="--s",
            capsize=6,
            linewidth=2.0,
            color="#e74c3c",
            label="Czas trenowania (s)",
        )
        _apply_decimal_comma(ax2)
        ax2.set_ylabel("Czas (sekundy)", fontsize=12, color="#e74c3c")
        ax2.tick_params(axis="y", labelcolor="#e74c3c")
        ax2.grid(False)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper left",
            frameon=True,
            shadow=True,
        )
    else:
        ax.legend(loc="upper left", frameon=True, shadow=True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(spec.output_path, dpi=300)
    plt.close(fig)


def _plot_compare_metrics(
    summary: pd.DataFrame, csv_path: Path, output_dir: Path
) -> None:
    if summary.empty:
        return

    dataset_id = csv_path.stem.split("_")[-1]
    title = f"Porównanie metryk na zbiorze {dataset_id}"

    metrics = [metric for metric in _METRIC_KEYS if metric in summary.columns]
    if not metrics:
        return

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 7))
    x_positions = np.arange(len(summary.index))
    bar_width = 0.8 / max(len(metrics), 1)
    offsets = (np.arange(len(metrics)) - (len(metrics) - 1) / 2) * bar_width

    for offset, metric_key in zip(offsets, metrics, strict=False):
        ax.bar(
            x_positions + offset,
            summary[metric_key].to_numpy(),
            width=bar_width,
            label=summaries.METRIC_LABELS.get(metric_key, metric_key),
            color=_METRIC_COLORS.get(metric_key, "#2c3e50"),
        )

    _apply_decimal_comma(ax)
    ax.set_xlabel("Typ lasu", fontsize=12)
    ax.set_ylabel("Wartość metryki", fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(summary.index.astype(str), rotation=0)
    ax.legend(loc="lower left", frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{csv_path.stem}_compare_metrics.png", dpi=300)
    plt.close(fig)


def plot_compare_metrics_for_csv(csv_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    if df.empty or "forest_type" not in df.columns:
        return

    metric_values: dict[str, pd.Series] = {}
    for metric_key in _METRIC_KEYS:
        values = summaries._compute_metric_values(df, metric_key)
        if values is not None:
            metric_values[metric_key] = values

    if not metric_values:
        return

    metrics_df = pd.DataFrame(metric_values)
    metrics_df["forest_type"] = df["forest_type"].astype(str)

    forest_order = metrics_df["forest_type"].dropna().unique().tolist()
    summary = (
        metrics_df.groupby("forest_type", dropna=False)[list(metric_values.keys())]
        .mean(numeric_only=True)
        .reindex(forest_order)
    )

    _plot_compare_metrics(summary, csv_path, output_dir)


def plot_metrics_for_csv(csv_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    param_info = summaries.get_param_info(csv_path, df)
    if not param_info:
        return
    param_key, _ = param_info
    param_label = summaries.COLUMN_NAME_MAP.get(param_key, param_key)

    metric_summaries: dict[str, pd.DataFrame] = {}
    for metric_key in _METRIC_KEYS:
        summary = summaries.build_metric_summary_from_df(
            df,
            csv_path,
            metric_key,
            include_extremes=False,
        )
        if summary is not None:
            metric_summaries[metric_key] = summary

    if not metric_summaries:
        return

    time_summary = summaries.build_metric_summary_from_df(
        df,
        csv_path,
        "time_of_building",
        include_extremes=False,
    )
    base_title = summaries.build_title_from_path(csv_path)
    combined_title = f"Metryki jakości - {base_title}"
    combined_spec = CombinedPlotSpec(
        param_key=param_key,
        param_label=param_label,
        title=combined_title,
        output_path=output_dir / f"{csv_path.stem}_metrics.png",
        time_summary=time_summary,
    )
    _plot_metrics_combined(metric_summaries, combined_spec)

    for metric_key, summary in metric_summaries.items():
        metric_label = summaries.METRIC_LABELS.get(metric_key, metric_key)
        metric_title = f"{metric_label} - {base_title}"
        metric_spec = MetricPlotSpec(
            param_key=param_key,
            param_label=param_label,
            title=metric_title,
            output_path=output_dir / f"{csv_path.stem}_{metric_key}.png",
        )
        _plot_metric(summary, metric_key, metric_spec)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate metric plots from experiment_output CSV files."
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
        help="Directory for charts (defaults to input dir).",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in sorted(input_dir.glob("*.csv")):
        if csv_path.stem.startswith("Compare_on"):
            plot_compare_metrics_for_csv(csv_path, output_dir)
        else:
            plot_metrics_for_csv(csv_path, output_dir)


if __name__ == "__main__":
    main()
